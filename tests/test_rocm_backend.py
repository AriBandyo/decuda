from pathlib import Path
from unittest.mock import patch

from backend.rocm import ROCmBackend


def test_compile_builds_correct_hipcc_command():
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