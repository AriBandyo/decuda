"""Evidence harness: runs baseline and candidate, records what happened.

Collects. Never concludes. There is no speedup calculation here and no
early exit on a result that "looks fine" -- judgment belongs to the referee.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from backend.base import Backend
from .evidence import (
    Evidence, ExecutionRecord, TimingSample, EnvironmentRecord,
)


class Harness:

    def __init__(self, backend: Backend, warmup: int = 10, iterations: int = 40):
        self.backend = backend
        self.warmup = warmup
        self.iterations = iterations

    def collect(self, kernel_name: str,
                baseline_source: Path, candidate_source: Path,
                build_dir: Path,
                device_name: str = "unknown",
                rocm_version: str = "unknown",
                peak_bandwidth_gb_s: float = 0.0) -> Evidence:

        build_dir.mkdir(parents=True, exist_ok=True)
        baseline_bin = build_dir / f"{kernel_name}_baseline"
        candidate_bin = build_dir / f"{kernel_name}_candidate"

        self.backend.compile(baseline_source, baseline_bin)
        self.backend.compile(candidate_source, candidate_bin)

        baseline_exec = self._execute_once(baseline_source, baseline_bin)
        candidate_exec = self._execute_once(candidate_source, candidate_bin)

        baseline_times, candidate_times = self._time_interleaved(
            baseline_bin, candidate_bin
        )

        return Evidence(
            kernel_name=kernel_name,
            environment=EnvironmentRecord(
                device_name=device_name,
                rocm_version=rocm_version,
                hostname=platform.node(),
                peak_bandwidth_gb_s=peak_bandwidth_gb_s,
            ),
            baseline_execution=baseline_exec,
            candidate_execution=candidate_exec,
            baseline_timing=TimingSample(
                tuple(baseline_times), self.warmup, len(baseline_times)),
            candidate_timing=TimingSample(
                tuple(candidate_times), self.warmup, len(candidate_times)),
            baseline_output=self._parse_output(baseline_exec.stdout),
            candidate_output=self._parse_output(candidate_exec.stdout),
        )

    def _execute_once(self, source: Path, binary: Path) -> ExecutionRecord:
        result = self.backend.run(binary)
        return ExecutionRecord(
            source_path=str(source),
            compile_command=("hipcc", "-O3", str(source), "-o", str(binary)),
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def _time_interleaved(self, baseline_bin: Path, candidate_bin: Path):
        """Alternate between the two binaries so thermal drift and clock ramp
        affect both equally. Running one in a block and then the other gives
        whichever went second systematically different hardware conditions --
        a real effect, and a common source of speedups that are really just
        'measured while the GPU was cold'.
        """
        for _ in range(self.warmup):
            self.backend.run(baseline_bin)
            self.backend.run(candidate_bin)

        baseline_times, candidate_times = [], []
        for _ in range(self.iterations):
            baseline_times.append(self._wall_time_ms(baseline_bin))
            candidate_times.append(self._wall_time_ms(candidate_bin))

        return baseline_times, candidate_times

    def _wall_time_ms(self, binary: Path) -> float:
        """Wall time around the whole process.

        Less precise than in-process HIP events -- process startup is included
        in every sample. But the process cannot exit before the GPU is done,
        so a missing device synchronize is structurally impossible rather than
        something to detect after the fact. Precision traded for integrity.
        """
        start = time.perf_counter()
        self.backend.run(binary)
        return (time.perf_counter() - start) * 1000.0

    @staticmethod
    def _parse_output(stdout: str) -> tuple[float, ...]:
        """Kernels emit one line of JSON. An 'output' array is optional --
        absent means correctness comparison is unavailable, not that it passed.
        """
        try:
            payload = json.loads(stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return ()
        values = payload.get("output")
        if not isinstance(values, list):
            return ()
        return tuple(float(v) for v in values)