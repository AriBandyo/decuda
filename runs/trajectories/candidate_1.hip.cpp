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

// Warp/wavefront size for CDNA3 is 64
#define WAVEFRONT_SIZE 64

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Fused single-pass online softmax kernel using Welford-style online algorithm
// One block per row, uses wavefront-level reductions
__global__ void softmax_rows_fused(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int rows, int cols) {
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    if (row >= rows) return;

    const float* row_in  = in  + static_cast<size_t>(row) * cols;
    float*       row_out = out + static_cast<size_t>(row) * cols;

    // Number of wavefronts in this block
    const int num_warps = bdim / WAVEFRONT_SIZE;
    const int warp_id = tid / WAVEFRONT_SIZE;
    const int lane_id = tid % WAVEFRONT_SIZE;

    // Use shared memory layout: first num_warps floats for max, next num_warps for sum
    float* smax = sdata;
    float* ssum = sdata + num_warps;

    // Online softmax: single pass to compute max and sum simultaneously
    // using the parallel online normalization algorithm
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    for (int i = tid; i < cols; i += bdim) {
        float val = row_in[i];
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    // Wavefront-level reduction for max and sum (online merge)
    // We need to merge (max, sum) pairs across lanes
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        float other_max = __shfl_down(local_max, offset);
        float other_sum = __shfl_down(local_sum, offset);
        if (other_max > local_max) {
            local_sum = local_sum * expf(local_max - other_max) + other_sum;
            local_max = other_max;
        } else {
            local_sum += other_sum * expf(other_max - local_max);
        }
    }

    // Lane 0 of each warp writes to shared memory
    if (lane_id == 0) {
        smax[warp_id] = local_max;
        ssum[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction across warps (done by first warp)
    if (warp_id == 0) {
        float wmax = (lane_id < num_warps) ? smax[lane_id] : -INFINITY;
        float wsum = (lane_id < num_warps) ? ssum[lane_id] : 0.0f;

        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
            float other_max = __shfl_down(wmax, offset);
            float other_sum = __shfl_down(wsum, offset);
            if (other_max > wmax) {
                wsum = wsum * expf(wmax - other_max) + other_sum;
                wmax = other_max;
            } else {
                wsum += other_sum * expf(other_max - wmax);
            }
        }

        if (lane_id == 0) {
            smax[0] = wmax;
            ssum[0] = wsum;
        }
    }
    __syncthreads();

    const float row_max = smax[0];
    const float row_sum = ssum[0];
    const float inv_sum = 1.0f / row_sum;

    // Normalize
    for (int i = tid; i < cols; i += bdim) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
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
  // Shared memory: 2 * num_warps floats (max and sum per warp)
  const int num_warps = threads / WAVEFRONT_SIZE;
  const size_t shmem = 2 * num_warps * sizeof(float);

  hipEvent_t ev_start, ev_stop;
  CUDA_CHECK(hipEventCreate(&ev_start));
  CUDA_CHECK(hipEventCreate(&ev_stop));

  for (int w = 0; w < 10; ++w) {
    softmax_rows_fused<<<blocks, threads, shmem>>>(d_in, d_out, rows, cols);
  }
  CUDA_CHECK(hipDeviceSynchronize());

  const int NUM_ITERS = 20;
  CUDA_CHECK(hipEventRecord(ev_start));
  for (int i = 0; i < NUM_ITERS; ++i) {
    softmax_rows_fused<<<blocks, threads, shmem>>>(d_in, d_out, rows, cols);
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
