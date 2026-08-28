"""Canonical repo paths. Import these instead of hardcoding relative paths."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
KERNELS_DIR = REPO_ROOT / "kernels"
BUILD_DIR = REPO_ROOT / "build"

SMOKE_VECTOR_ADD = KERNELS_DIR / "smoke" / "vector_add.hip.cpp"
SOFTMAX_CUDA = KERNELS_DIR / "softmax" / "softmax.cu"