import subprocess
from pathlib import Path

from .base import Backend


class ROCmBackend(Backend):
    """ROCm/HIP backend for decuda."""

    def compile(self, source: Path, output: Path) -> Path:
        command = [
            "hipcc",
            "-O3",
            str(source),
            "-o",
            str(output),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "HIP compilation failed\n"
                f"command: {' '.join(command)}\n"
                f"stderr:\n{result.stderr}"
            )

        return output

    def run(self, executable: Path) -> str:
        result = subprocess.run(
            [str(executable)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "HIP program failed\n"
                f"executable: {executable}\n"
                f"stderr:\n{result.stderr}"
            )

        return result.stdout