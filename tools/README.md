# Vendored tools

## hipify-perl

AMD's CUDA-to-HIP source translator, vendored from
https://github.com/ROCm/HIPIFY (branch: amd-staging), MIT licensed.
Copyright (c) 2015-present Advanced Micro Devices, Inc.

Vendored rather than installed so the CUDA -> HIP translation step is
reproducible without a ROCm installation, and so the exact translator
version that produced the baseline is pinned alongside the results.
