import json
from pathlib import Path
from unittest.mock import patch

from backend.rocm import ROCmBackend
from paths import SMOKE_VECTOR_ADD
from paths import SMOKE_VECTOR_ADD, SOFTMAX_CUDA


def test_compile_builds_correct_hipcc_command():
    """The fair-baseline build is hipcc -O3. Locking the exact invocation
    means a silent flag change can't quietly move the baseline."""
    backend = ROCmBackend()

    source = Path("kernel.hip.cpp")
    output = Path("kernel")

    with patch("backend.rocm.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""

        result = backend.compile(source, output)

        mock_run.assert_called_once_with(
            [
                "hipcc",
                "-O3",
                "kernel.hip.cpp",
                "-o",
                "kernel",
            ],
            capture_output=True,
            text=True,
        )

    assert result == output


def test_run_preserves_stdout_when_kernel_reports_wrong_answer():
    """Exit code 2 means the kernel ran and self-reported a wrong answer.
    The JSON on stdout is evidence the referee needs -- it must survive."""
    backend = ROCmBackend()

    with patch("backend.rocm.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 2
        mock_run.return_value.stdout = (
            '{"kernel":"vector_add","correct":false,'
            '"n":1048576,"max_abs_err":1.250e-01}\n'
        )
        mock_run.return_value.stderr = ""

        result = backend.run(Path("./vector_add"))

    assert result.returncode == 2
    assert result.ok is False

    payload = json.loads(result.stdout)
    assert payload["kernel"] == "vector_add"
    assert payload["correct"] is False


def test_smoke_kernel_source_exists():
    """Unmocked. Catches the kernel being moved or renamed out from under us."""
    assert SMOKE_VECTOR_ADD.is_file()


def test_softmax_cuda_source_exists():
    assert SOFTMAX_CUDA.is_file()