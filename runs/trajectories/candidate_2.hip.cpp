#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(cmd)                                                       \
  do {                                                                        \
    hipError_t _err = (cmd);                                                 \
    if (_err != hipSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d -- %s\n",                     \
                   __FILE__, __LINE__, hipGetErrorString(_err));             \
      std::printf("{\"kernel\":\"softmax\",\"correct\":false,"                \
                  "\"error\":\"%s\"}\n", hipGetErrorName(_err));             \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define THREADS_PER_BLOCK 256
#define WAVEFRONT_SIZE 64

// Wavefront-level max reduction
__device__ __forceinline__ float wavefront_max(float val) {
    val = __builtin_amdgcn_ds_swizzle(val, 0x8000 | (1 << 10) | 0); // not available, use shfl
    // Use shuffle-based reduction for wavefront of 64
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor(val, offset, WAVEFRONT_SIZE);
        val = fmaxf(val, other);
    }
    return val;
}

// Wavefront-level sum reduction
__device__ __forceinline__ float wavefront_sum(float val) {
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, WAVEFRONT_SIZE);
    }
    return val;
}

// Optimized softmax: one block per row
// Uses online softmax (single pass for max+sum) + vectorized loads
// Block size must be multiple of WAVEFRONT_SIZE (64)
__global__ void softmax_rows(const float* __restrict__ in,
                             float* __restrict__ out,
                             int rows, int cols) {
    // LDS for cross-wavefront reductions
    __shared__ float smax[THREADS_PER_BLOCK / WAVEFRONT_SIZE];
    __shared__ float ssum[THREADS_PER_BLOCK / WAVEFRONT_SIZE];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / WAVEFRONT_SIZE;   // wavefront index
    const int lane = tid % WAVEFRONT_SIZE;  // lane within wavefront
    const int num_wavefronts = blockDim.x / WAVEFRONT_SIZE;

    if (row >= rows) return;

    const float* row_in  = in  + static_cast<size_t>(row) * cols;
    float*       row_out = out + static_cast<size_t>(row) * cols;

    // Pass 1: Online softmax - compute max and sum in one pass
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    // Use float4 loads where possible
    int i = tid;
    // Process float4 chunks
    const int cols4 = (cols / 4) * 4;
    const float4* row_in4 = reinterpret_cast<const float4*>(row_in);
    int tid4 = tid;
    int cols_div4 = cols / 4;
    
    for (int j = tid4; j < cols_div4; j += blockDim.x) {
        float4 v = row_in4[j];
        // Update running max and sum with online algorithm
        float m_new = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
        if (m_new != local_max) {
            local_sum = local_sum * expf(local_max - m_new) + 
                        expf(v.x - m_new) + expf(v.y - m_new) + 
                        expf(v.z - m_new) + expf(v.w - m_new);
        } else {
            local_sum += expf(v.x - m_new) + expf(v.y - m_new) + 
                         expf(v.z - m_new) + expf(v.w - m_new);
        }
        local_max = m_new;
    }
    // Handle remainder
    for (int j = cols4 + tid; j < cols; j += blockDim.x) {
        float val = row_in[j];
        float m_new = fmaxf(local_max, val);
        if (m_new != local_max) {
            local_sum = local_sum * expf(local_max - m_new) + expf(val - m_new);
        } else {
            local_sum += expf(val - m_new);
        }
        local_max = m_new;
    }

    // Wavefront-level reduction for online softmax
    // Need to combine (max, sum) pairs across wavefront
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        float other_max = __shfl_xor(local_max, offset, WAVEFRONT_SIZE);
        float other_sum = __shfl_xor(local_sum, offset, WAVEFRONT_SIZE);
        if (other_max > local_max) {
            local_sum = local_sum * expf(local_max - other_max) + other_sum;
            local_max = other_max;
        } else {
            local_sum = local_sum + other_sum * expf(other_max - local_max);
        }
    }

    // Store wavefront results to LDS
    if (lane == 0) {
        smax[wid] = local_max;
        ssum[wid] = local_sum;
    }
    __syncthreads();

    // Final reduction across wavefronts (done by first wavefront)
    if (wid == 0) {
        float m = (lane < num_wavefronts) ? smax[lane] : -INFINITY;
        float s = (lane < num_wavefronts) ? ssum[lane] : 0.0f;
        
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
            float other_max = __shfl_xor(m, offset, WAVEFRONT_SIZE);
            float other_sum = __shfl_xor(s, offset, WAVEFRONT_SIZE);
            if (other_max > m) {
                s = s * expf(m - other_max) + other_sum;
                m = other_max;
            } else {
                s = s + other_sum * expf(other_max - m);
            }
        }
        
        if (lane == 0) {
            smax[0] = m;
            ssum[0] = s;
        }
    }
    __syncthreads();

    const float row_max = smax[0];
    const float inv_sum = 1.0f / ssum[0];

    // Pass 2: Normalize with vectorized stores
    for (int j = tid4; j < cols_div4; j += blockDim.x) {
        float4 v = row_in4[j];
        float4 res;
        res.x = expf(v.x - row_max) * inv_sum;
        res.y = expf(v.y - row_max) * inv_sum;
        res.z = expf(v.z - row_max) * inv_sum;
        res.w = expf(v.w - row_max) * inv_sum;
        reinterpret_cast<float4*>(row_out)[j] = res;
    }
    for (int j = cols4 + tid; j < cols; j += blockDim.x) {
        row_out[j] = expf(row_in[j] - row_max) * inv_sum;
    }
}

static void softmax_reference(const std::vector<float>& in,
                              std::vector<double>& out,
                              int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* row = in.data() + static_cast<size_t>(r) * cols;
    double m = -INFINITY;
    for (int c = 0; c < cols; ++c) m = std::fmax(m, static_cast<double>(row[c]));

    double s = 0.0;
    for (int c = 0; c < cols; ++c) s += std::exp(static_cast<double>(row[c]) - m);

    for (int c = 0; c < cols; ++c) {
      out[static_cast<size_t>(r) * cols + c] =
          std::exp(static_cast<double>(row[c]) - m) / s;
    }
  }
}

int main() {
  const int rows = 1024;
  const int cols = 1024;
  const size_t n = static_cast<size_t>(rows) * cols;
  const size_t bytes = n * sizeof(float);

  std::vector<float>  h_in(n), h_out(n);
  std::vector<double> h_ref(n);

  std::srand(42);
  for (size_t i = 0; i < n; ++i) {
    h_in[i] = static_cast<float>(std::rand()) / RAND_MAX * 10.0f - 5.0f;
  }

  softmax_reference(h_in, h_ref, rows, cols);

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(hipMalloc(&d_in, bytes));
  CUDA_CHECK(hipMalloc(&d_out, bytes));
  CUDA_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

  const int threads = THREADS_PER_BLOCK;
  const int blocks = rows;
  const size_t shmem = (threads / WAVEFRONT_SIZE) * 2 * sizeof(float);
  
  hipEvent_t ev_start, ev_stop;
  CUDA_CHECK(hipEventCreate(&ev_start));
  CUDA_CHECK(hipEventCreate(&ev_stop));

  for (int w = 0; w < 10; ++w) {
    softmax_rows<<<blocks, threads, shmem>>>(d_in, d_out, rows, cols);
  }
  CUDA_CHECK(hipDeviceSynchronize());

  const int NUM_ITERS = 20;
  CUDA_CHECK(hipEventRecord(ev_start));
  for (int i = 0; i < NUM_ITERS; ++i) {
    softmax_rows<<<blocks, threads, shmem>>>(d_in, d_out, rows, cols);
  }
  CUDA_CHECK(hipEventRecord(ev_stop));
  CUDA_CHECK(hipEventSynchronize(ev_stop));
  float total_ms = 0.0f;
  CUDA_CHECK(hipEventElapsedTime(&total_ms, ev_start, ev_stop));
  float kernel_ms = total_ms / NUM_ITERS;

  CUDA_CHECK(hipGetLastError());
  CUDA_CHECK(hipDeviceSynchronize());
  CUDA_CHECK(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));

  double max_abs_err = 0.0;
  bool   saw_nonfinite = false;
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(h_out[i])) { saw_nonfinite = true; break; }
    double err = std::fabs(static_cast<double>(h_out[i]) - h_ref[i]);
    if (err > max_abs_err) max_abs_err = err;
  }

  const bool correct = !saw_nonfinite && (max_abs_err <= 1e-6);

  CUDA_CHECK(hipFree(d_in));
  CUDA_CHECK(hipFree(d_out));

  const int SAMPLE_STRIDE = 7919;

  std::printf("{\"kernel\":\"softmax\",\"correct\":%s,"
              "\"rows\":%d,\"cols\":%d,\"max_abs_err\":%.3e",
              correct ? "true" : "false", rows, cols, max_abs_err);

  std::printf(",\"row_sums\":[");
  for (int r = 0; r < rows; ++r) {
    double s = 0.0;
    for (int c = 0; c < cols; ++c) {
      s += static_cast<double>(h_out[static_cast<size_t>(r) * cols + c]);
    }
    std::printf("%s%.9f", r ? "," : "", s);
  }
  std::printf("]");

  std::printf(",\"output\":[");
  bool first = true;
  for (size_t i = 0; i < n; i += SAMPLE_STRIDE) {
    std::printf("%s%.9e", first ? "" : ",", h_out[i]);
    first = false;
  }
  std::printf("]");

  std::printf(",\"sample_stride\":%d,\"kernel_ms\":%.6f}\n", SAMPLE_STRIDE, kernel_ms);

  return correct ? 0 : 2;
}
