from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Sequence

from .evidence import Evidence


@dataclass(frozen = True) # so the class behaves as a simple data container and the data is immutable 
class CheckResult:
    name: str
    passed : bool
    detail : str



@dataclass(frozen=True)
class Verdict:
    accepted: bool
    checks : list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def feedback(self) -> str:

        if self.accepted:
            return "ACCEPTED: all checks passed."
        lines = ["REJECTED"]
        lines += [f" [{c.name}] {c.detail}" for c in self.failures]
        return "\n".join(lines)


class CorrectnessReferee:
    """Decides whether a candidate computes the same thing as the baseline."""

    def __init__(self, tolerance: float = 1e-5, row_sum_tolerance: float = 1e-4):
        self.tolerance = tolerance
        self.row_sum_tolerance = row_sum_tolerance

    def adjudicate(self, evidence: Evidence) -> Verdict:
        checks: list[CheckResult] = []

        checks.append(self._check_execution(evidence))
        checks.append(self._check_shape(evidence))

        # Only meaningful once shape is known good.
        if all(c.passed for c in checks):
            checks.append(self._check_finite(evidence))

        if all(c.passed for c in checks):
            checks.append(self._check_magnitude(evidence))

        if all(c.passed for c in checks):
            checks.append(self._check_row_sums(evidence))

        return Verdict(accepted=all(c.passed for c in checks), checks=checks)

    def _check_execution(self, ev: Evidence) -> CheckResult:
        rc = ev.candidate_execution.returncode
        if rc != 0:
            return CheckResult(
                "execution", False,
                f"candidate exited {rc}; stderr: {ev.candidate_execution.stderr.strip()[:200]}",
            )
        return CheckResult("execution", True, "candidate exited 0")

    def _check_shape(self, ev: Evidence) -> CheckResult:
        nb, nc = len(ev.baseline_output), len(ev.candidate_output)
        if nb == 0:
            return CheckResult("shape", False, "baseline produced no output")
        if nb != nc:
            return CheckResult(
                "shape", False,
                f"candidate produced {nc} elements, baseline produced {nb} "
                f"-- a kernel that writes less output is not faster, it is incomplete",
            )
        return CheckResult("shape", True, f"{nc} elements, matches baseline")

    def _check_finite(self, ev: Evidence) -> CheckResult:
        """Must run BEFORE magnitude. NaN compares false to everything, so a
        NaN-producing candidate would otherwise pass a tolerance check with
        zero recorded error."""
        for i, v in enumerate(ev.candidate_output):
            if not math.isfinite(v):
                return CheckResult(
                    "finite", False,
                    f"candidate output[{i}] is {v} -- non-finite values silently "
                    f"defeat tolerance comparison",
                )
        return CheckResult("finite", True, "all candidate values finite")

    def _check_magnitude(self, ev: Evidence) -> CheckResult:
        worst, worst_i = 0.0, -1
        for i, (b, c) in enumerate(zip(ev.baseline_output, ev.candidate_output)):
            d = abs(b - c)
            if d > worst:
                worst, worst_i = d, i
        if worst > self.tolerance:
            return CheckResult(
                "magnitude", False,
                f"max abs error {worst:.3e} at index {worst_i} "
                f"exceeds tolerance {self.tolerance:.3e}",
            )
        return CheckResult("magnitude", True, f"max abs error {worst:.3e}")

    def _check_row_sums(self, ev: Evidence) -> CheckResult:
        """Property check. Softmax rows sum to 1 by definition.

        Reads sums the kernel computed over the FULL matrix, not sums derived
        from the sparse output sample -- a sample cannot be sliced into rows,
        and deriving them from one would check a different claim than the one
        being made.
        """
        sums = ev.candidate_row_sums
        if not sums:
            return CheckResult(
                "row_sums", False,
                "candidate emitted no row_sums -- absent evidence is not "
                "passing evidence",
            )

        for r, s in enumerate(sums):
            if abs(s - 1.0) > self.row_sum_tolerance:
                return CheckResult(
                    "row_sums", False,
                    f"row {r} sums to {s:.9f}, not 1.0 -- output is not a "
                    f"probability distribution",
                )
        return CheckResult("row_sums", True, f"all {len(sums)} rows sum to 1.0")

