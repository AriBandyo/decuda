import subprocess
from pathlib import Path

from .base import Backend
from .base import Backend, RunResult


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

    def run(self, executable: Path, timeout: float = 60.0) -> RunResult:
        try:
            result = subprocess.run(
                [str(executable)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return RunResult(
                returncode=124,
                stdout=exc.stdout or "",
                stderr=f"execution timed out after {timeout}s",
            )

        return RunResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )