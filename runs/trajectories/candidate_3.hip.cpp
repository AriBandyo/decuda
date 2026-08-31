// kernels/softmax/softmax.cu
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

// Wavefront size on CDNA3 is 64
#define WAVEFRONT_SIZE 64

// Warp-level reduction using __shfl_xor (wavefront = 64 on CDNA3)
__device__ __forceinline__ float wavefront_max(float val) {
    val = fmaxf(val, __shfl_xor(val, 32));
    val = fmaxf(val, __shfl_xor(val, 16));
    val = fmaxf(val, __shfl_xor(val, 8));
    val = fmaxf(val, __shfl_xor(val, 4));
    val = fmaxf(val, __shfl_xor(val, 2));
    val = fmaxf(val, __shfl_xor(val, 1));
    return val;
}

__device__ __forceinline__ float wavefront_sum(float val) {
    val += __shfl_xor(val, 32);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}

// Fused two-pass softmax: pass1 = online max+sum, pass2 = normalize
// One block per row, uses shared memory for cross-wavefront reduction
__global__ void softmax_rows(const float* __restrict__ in,
                             float* __restrict__ out,
                             int rows, int cols) {
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % WAVEFRONT_SIZE;
    const int wid  = tid / WAVEFRONT_SIZE;
    const int num_waves = blockDim.x / WAVEFRONT_SIZE;

    if (row >= rows) return;

    const float* row_in  = in  + static_cast<size_t>(row) * cols;
    float*       row_out = out + static_cast<size_t>(row) * cols;

    // Shared memory layout: [0..num_waves) for max, [num_waves..2*num_waves) for sum
    float* smax = sdata;
    float* ssum = sdata + num_waves;

    // Pass 1: compute row maximum
    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, row_in[i]);
    }

    // Wavefront-level max reduction
    local_max = wavefront_max(local_max);
    if (lane == 0) smax[wid] = local_max;
    __syncthreads();

    // Block-level max reduction (only first wavefront)
    if (wid == 0) {
        float val = (lane < num_waves) ? smax[lane] : -INFINITY;
        val = wavefront_max(val);
        if (lane == 0) smax[0] = val;
    }
    __syncthreads();
    const float row_max = smax[0];
    __syncthreads();

    // Pass 2: compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_sum += expf(row_in[i] - row_max);
    }

    // Wavefront-level sum reduction
    local_sum = wavefront_sum(local_sum);
    if (lane == 0) ssum[wid] = local_sum;
    __syncthreads();

    // Block-level sum reduction (only first wavefront)
    if (wid == 0) {
        float val = (lane < num_waves) ? ssum[lane] : 0.0f;
        val = wavefront_sum(val);
        if (lane == 0) ssum[0] = val;
    }
    __syncthreads();
    const float inv_sum = 1.0f / ssum[0];
    __syncthreads();

    // Pass 3: normalize
    for (int i = tid; i < cols; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
    }
}

// Host reference, computed in double precision
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
    // Shared memory: 2 * num_waves floats
    const int num_waves = threads / WAVEFRONT_SIZE;
    const size_t shmem = 2 * num_waves * sizeof(float);

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
