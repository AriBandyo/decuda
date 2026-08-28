// kernels/softmax/softmax.cu
//
// Row-wise softmax over an R x C float32 matrix, CUDA baseline.
//
// This is the numerically stable formulation: subtract the row max before
// exponentiating. Three passes over each row (max, sum, normalize). It is
// deliberately unfused -- this is the honest textbook baseline that HIPIFY
// translates and hipcc -O3 compiles, not a strawman.
//
// Self-contained by design: no framework headers, so HIPIFY sees only CUDA
// API calls and kernel syntax.

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

// One block per row. Three passes: max, sum of exp, normalize.
__global__ void softmax_rows(const float* __restrict__ in,
                             float* __restrict__ out,
                             int rows, int cols) {
  extern __shared__ float sdata[];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const float* row_in  = in  + static_cast<size_t>(row) * cols;
  float*       row_out = out + static_cast<size_t>(row) * cols;

  // Pass 1: row maximum.
  float local_max = -INFINITY;
  for (int i = tid; i < cols; i += blockDim.x) {
    local_max = fmaxf(local_max, row_in[i]);
  }
  sdata[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = sdata[0];
  __syncthreads();

  // Pass 2: sum of exp(x - max).
  float local_sum = 0.0f;
  for (int i = tid; i < cols; i += blockDim.x) {
    local_sum += expf(row_in[i] - row_max);
  }
  sdata[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  const float row_sum = sdata[0];
  __syncthreads();

  // Pass 3: normalize.
  const float inv_sum = 1.0f / row_sum;
  for (int i = tid; i < cols; i += blockDim.x) {
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
  const size_t shmem = threads * sizeof(float);

  softmax_rows<<<blocks, threads, shmem>>>(d_in, d_out, rows, cols);

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

  std::printf("{\"kernel\":\"softmax\",\"correct\":%s,"
              "\"rows\":%d,\"cols\":%d,\"max_abs_err\":%.3e}\n",
              correct ? "true" : "false", rows, cols, max_abs_err);

  return correct ? 0 : 2;
}