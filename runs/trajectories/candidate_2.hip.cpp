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

// Online softmax: single pass for max+sum, one pass for normalize.
// This reduces global memory reads from 3 passes to 2 passes.
// Uses float4 vectorized loads for better memory throughput.
__global__ void softmax_rows(const float* __restrict__ in,
                             float* __restrict__ out,
                             int rows, int cols) {
  extern __shared__ float sdata[];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdim = blockDim.x;
  if (row >= rows) return;

  const float* row_in  = in  + static_cast<size_t>(row) * cols;
  float*       row_out = out + static_cast<size_t>(row) * cols;

  // Online softmax: compute max and sum in one pass using the
  // Milakov & Gimelshein algorithm.
  // local_m = running max, local_s = running sum adjusted for max
  float local_m = -INFINITY;
  float local_s = 0.0f;

  // Vectorized load with float4 where possible
  const int cols4 = cols / 4;
  const float4* row_in4 = reinterpret_cast<const float4*>(row_in);

  for (int i = tid; i < cols4; i += bdim) {
    float4 v = row_in4[i];
    // Process each element with online update
    float new_m = fmaxf(local_m, v.x);
    local_s = local_s * expf(local_m - new_m) + expf(v.x - new_m);
    local_m = new_m;

    new_m = fmaxf(local_m, v.y);
    local_s = local_s * expf(local_m - new_m) + expf(v.y - new_m);
    local_m = new_m;

    new_m = fmaxf(local_m, v.z);
    local_s = local_s * expf(local_m - new_m) + expf(v.z - new_m);
    local_m = new_m;

    new_m = fmaxf(local_m, v.w);
    local_s = local_s * expf(local_m - new_m) + expf(v.w - new_m);
    local_m = new_m;
  }

  // Handle remaining elements
  for (int i = cols4 * 4 + tid; i < cols; i += bdim) {
    float v = row_in[i];
    float new_m = fmaxf(local_m, v);
    local_s = local_s * expf(local_m - new_m) + expf(v - new_m);
    local_m = new_m;
  }

  // Store both max and sum in shared memory for reduction
  // Use two arrays: sdata[0..bdim-1] for max, sdata[bdim..2*bdim-1] for sum
  float* smax = sdata;
  float* ssum = sdata + bdim;

  smax[tid] = local_m;
  ssum[tid] = local_s;
  __syncthreads();

  // Reduce to find global max and sum using online algorithm
  for (int stride = bdim / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float m_a = smax[tid];
      float s_a = ssum[tid];
      float m_b = smax[tid + stride];
      float s_b = ssum[tid + stride];
      float new_m = fmaxf(m_a, m_b);
      float new_s = s_a * expf(m_a - new_m) + s_b * expf(m_b - new_m);
      smax[tid] = new_m;
      ssum[tid] = new_s;
    }
    __syncthreads();
  }

  const float row_max = smax[0];
  const float row_sum = ssum[0];
  const float inv_sum = 1.0f / row_sum;

  // Pass 2: normalize using vectorized stores
  float4* row_out4 = reinterpret_cast<float4*>(row_out);
  for (int i = tid; i < cols4; i += bdim) {
    float4 v = row_in4[i];
    float4 r;
    r.x = expf(v.x - row_max) * inv_sum;
    r.y = expf(v.y - row_max) * inv_sum;
    r.z = expf(v.z - row_max) * inv_sum;
    r.w = expf(v.w - row_max) * inv_sum;
    row_out4[i] = r;
  }

  // Handle remaining elements
  for (int i = cols4 * 4 + tid; i < cols; i += bdim) {
    row_out[i] = expf(row_in[i] - row_max) * inv_sum;
  }
}

// Host reference, computed in double precision so it is a stricter
// standard than the kernel it checks.
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

  // Deterministic input, fixed seed -- reruns must be comparable.
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
  // Need 2x shared memory for both max and sum arrays
  const size_t shmem = 2 * threads * sizeof(float);
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
