"""Evidence schema for referee adjudication.

Evidence records what was OBSERVED. It never records a conclusion.
There is deliberately no `speedup`, `passed`, or `is_correct` field --
those are verdicts, and verdicts are the referee's job. A component that
both produces a result and labels it acceptable is grading its own homework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class ExecutionRecord:
    """What happened when one binary was run, once."""
    source_path: str
    compile_command: Sequence[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class TimingSample:
    """Per-iteration wall times for one binary. Raw samples only.

    Summary statistics are withheld on purpose: a mean cannot be
    interrogated, a distribution can.
    """
    times_ms: Sequence[float]
    warmup_iterations: int
    measured_iterations: int


@dataclass(frozen=True)
class EnvironmentRecord:
    """Where the measurement was taken. Timings are meaningless without it."""
    device_name: str
    rocm_version: str
    hostname: str
    peak_bandwidth_gb_s: float = 0.0 # 0 = unknown; bandwidth check is skipped
    # peak_bandwidth_gbs_s is the only refree check that can call a result impossible rather than merely being suspicicous and this is the deciding thing that helps this project 

@dataclass(frozen=True)
class Evidence:
    """Everything the referee is allowed to look at.

    Both sides are recorded symmetrically. The referee must not be able to
    tell which side is 'supposed' to win from the shape of the data.
    """
    kernel_name: str
    environment: EnvironmentRecord

    baseline_execution: ExecutionRecord
    candidate_execution: ExecutionRecord

    baseline_timing: TimingSample
    candidate_timing: TimingSample

    # Numerical output, flattened. Compared element-wise by the referee.
    baseline_output: Sequence[float] = field(default_factory=list)
    candidate_output: Sequence[float] = field(default_factory=list)

    baseline_row_sums: tuple[float, ...] = ()
    candidate_row_sums: tuple[float, ...] = ()

    # Adversarial inputs the candidate was additionally run against,
    # keyed by case name -> flattened output.
    adversarial_outputs: dict[str, Sequence[float]] = field(default_factory=dict)
    