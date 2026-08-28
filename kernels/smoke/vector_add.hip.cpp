// kernels/smoke/vector_add.hip.cpp
//
// Minimal HIP smoke kernel: validates the device memory round trip and the
// build/run path. Emits one line of machine-readable JSON on stdout.
// Not a benchmark -- no timing here by design.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Every HIP call goes through this. On failure: human-readable detail to
// stderr, a machine-readable failure record to stdout, nonzero exit.
#define HIP_CHECK(cmd)                                                        \
  do {                                                                        \
    hipError_t _err = (cmd);                                                  \
    if (_err != hipSuccess) {                                                 \
      std::fprintf(stderr, "HIP error at %s:%d -- %s (%s)\n",                 \
                   __FILE__, __LINE__,                                        \
                   hipGetErrorName(_err), hipGetErrorString(_err));           \
      std::printf("{\"kernel\":\"vector_add\",\"correct\":false,"             \
                  "\"error\":\"%s\"}\n", hipGetErrorName(_err));              \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int    n     = 1 << 20;               // 1,048,576 elements
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  std::vector<float> h_a(n), h_b(n), h_c(n), h_ref(n);
  for (int i = 0; i < n; ++i) {
    h_a[i]   = static_cast<float>(i) * 0.5f;
    h_b[i]   = static_cast<float>(n - i) * 0.25f;
    h_ref[i] = h_a[i] + h_b[i];              // host reference
  }

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  HIP_CHECK(hipMalloc(&d_a, bytes));
  HIP_CHECK(hipMalloc(&d_b, bytes));
  HIP_CHECK(hipMalloc(&d_c, bytes));

  HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

  const int threads = 256;
  const int blocks  = (n + threads - 1) / threads;

  vector_add<<<dim3(blocks), dim3(threads), 0, 0>>>(d_a, d_b, d_c, n);

  // Kernel launches are asynchronous and fail silently. These two lines are
  // the difference between a real result and a plausible-looking lie.
  HIP_CHECK(hipGetLastError());        // catches bad launch configuration
  HIP_CHECK(hipDeviceSynchronize());   // catches faults during execution

  HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));

  double max_abs_err = 0.0;
  for (int i = 0; i < n; ++i) {
    double err = std::fabs(static_cast<double>(h_c[i]) -
                           static_cast<double>(h_ref[i]));
    if (err > max_abs_err) max_abs_err = err;
  }
  const bool correct = (max_abs_err <= 1e-5);

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));

  std::printf("{\"kernel\":\"vector_add\",\"correct\":%s,"
              "\"n\":%d,\"max_abs_err\":%.3e}\n",
              correct ? "true" : "false", n, max_abs_err);

  return correct ? 0 : 2;   // 0 ok, 1 HIP failure, 2 wrong answer
}