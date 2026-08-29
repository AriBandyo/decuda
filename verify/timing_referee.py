"""Timing referee: audits whether a performance measurement is trustworthy.

Deliberately NOT a speed measurement. There is no ground truth for runtime,
only samples from a noisy distribution, so the only answerable question is
whether the evidence supports the claim -- not whether the number is 'right'.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .evidence import Evidence
from .referee import CheckResult, Verdict


@dataclass(frozen=True)
class TimingPolicy:
    min_measured_iterations: int = 20
    min_warmup_iterations: int = 5

    # A kernel moving megabytes cannot complete in microseconds.
    min_plausible_ms: float = 1e-3

    # Real GPU timings scatter. Identical samples mean something is wrong.
    min_relative_stddev: float = 1e-4

    # Noise larger than this makes small speedups indistinguishable.
    max_relative_stddev: float = 0.15

    # Claimed improvement must exceed combined noise by this factor.
    min_separation_factor: float = 2.0


class TimingReferee:

    def __init__(self, policy: TimingPolicy | None = None):
        self.policy = policy or TimingPolicy()

    def adjudicate(self, ev: Evidence, bytes_moved: int | None = None) -> Verdict:
        checks: list[CheckResult] = []

        checks.append(self._check_sample_count(ev))
        checks.append(self._check_plausible_duration(ev))
        checks.append(self._check_not_degenerate(ev))
        checks.append(self._check_noise_level(ev))
        checks.append(self._check_warmup(ev))

        if bytes_moved and ev.environment.peak_bandwidth_gb_s > 0:
            checks.append(self._check_bandwidth_ceiling(ev, bytes_moved))

        # Only meaningful if the measurements themselves are sound.
        if all(c.passed for c in checks):
            checks.append(self._check_separation(ev))

        return Verdict(accepted=all(c.passed for c in checks),
                       checks=tuple(checks))

    # -- individual checks -------------------------------------------------

    def _check_sample_count(self, ev: Evidence) -> CheckResult:
        p = self.policy
        for label, t in (("baseline", ev.baseline_timing),
                         ("candidate", ev.candidate_timing)):
            if len(t.times_ms) < p.min_measured_iterations:
                return CheckResult(
                    "sample_count", False,
                    f"{label} has {len(t.times_ms)} samples, "
                    f"need at least {p.min_measured_iterations}",
                )
        return CheckResult("sample_count", True,
                           f"{len(ev.candidate_timing.times_ms)} samples per side")

    def _check_plausible_duration(self, ev: Evidence) -> CheckResult:
        """Near-zero times usually mean the kernel never ran, or the timing
        bracket omitted a device synchronize and measured async dispatch."""
        p = self.policy
        for label, t in (("baseline", ev.baseline_timing),
                         ("candidate", ev.candidate_timing)):
            fastest = min(t.times_ms)
            if fastest < p.min_plausible_ms:
                return CheckResult(
                    "plausible_duration", False,
                    f"{label} fastest iteration {fastest:.6f} ms is below "
                    f"{p.min_plausible_ms} ms -- kernel likely did not run, or "
                    f"timing did not include a device synchronize",
                )
        return CheckResult("plausible_duration", True, "durations are plausible")

    def _check_not_degenerate(self, ev: Evidence) -> CheckResult:
        """Real GPU timings scatter. Identical samples mean a cached value,
        a timer coarser than the measurement, or fabricated numbers."""
        p = self.policy
        for label, t in (("baseline", ev.baseline_timing),
                         ("candidate", ev.candidate_timing)):
            mean = statistics.fmean(t.times_ms)
            sd = statistics.pstdev(t.times_ms)
            if mean > 0 and (sd / mean) < p.min_relative_stddev:
                return CheckResult(
                    "not_degenerate", False,
                    f"{label} relative stddev {sd / mean:.2e} is implausibly low "
                    f"-- samples are near-identical, which real hardware does not produce",
                )
        return CheckResult("not_degenerate", True, "sample variation is plausible")

    def _check_noise_level(self, ev: Evidence) -> CheckResult:
        p = self.policy
        for label, t in (("baseline", ev.baseline_timing),
                         ("candidate", ev.candidate_timing)):
            mean = statistics.fmean(t.times_ms)
            sd = statistics.pstdev(t.times_ms)
            if mean > 0 and (sd / mean) > p.max_relative_stddev:
                return CheckResult(
                    "noise_level", False,
                    f"{label} relative stddev {sd / mean:.2%} exceeds "
                    f"{p.max_relative_stddev:.0%} -- measurement is too noisy to "
                    f"support any speedup claim",
                )
        return CheckResult("noise_level", True, "noise within limits")

    def _check_warmup(self, ev: Evidence) -> CheckResult:
        """A dramatically slower first sample means warmup was skipped and
        cold-cache cost is polluting the mean."""
        p = self.policy
        for label, t in (("baseline", ev.baseline_timing),
                         ("candidate", ev.candidate_timing)):
            if t.warmup_iterations < p.min_warmup_iterations:
                return CheckResult(
                    "warmup", False,
                    f"{label} ran {t.warmup_iterations} warmup iterations, "
                    f"need at least {p.min_warmup_iterations}",
                )
            rest = t.times_ms[1:]
            if rest:
                median_rest = statistics.median(rest)
                if median_rest > 0 and t.times_ms[0] > 2.0 * median_rest:
                    return CheckResult(
                        "warmup", False,
                        f"{label} first sample {t.times_ms[0]:.4f} ms is more than "
                        f"2x the median of the rest ({median_rest:.4f} ms) -- "
                        f"warmup was insufficient",
                    )
        return CheckResult("warmup", True, "warmup adequate")

    def _check_bandwidth_ceiling(self, ev: Evidence, bytes_moved: int) -> CheckResult:
        """The strongest check available. A memory-bound kernel cannot exceed
        the device's peak bandwidth. Physics is a harder referee than statistics.
        """
        peak = ev.environment.peak_bandwidth_gb_s
        fastest_ms = min(ev.candidate_timing.times_ms)
        achieved = (bytes_moved / 1e9) / (fastest_ms / 1e3)
        if achieved > peak:
            return CheckResult(
                "bandwidth_ceiling", False,
                f"candidate implies {achieved:.1f} GB/s on a device with peak "
                f"{peak:.1f} GB/s -- the measurement is impossible, not fast",
            )
        return CheckResult(
            "bandwidth_ceiling", True,
            f"{achieved:.1f} GB/s, {achieved / peak:.0%} of peak",
        )

    def _check_separation(self, ev: Evidence) -> CheckResult:
        """Comparing means is what everyone does and it is not enough. If the
        distributions overlap, the speedup is not distinguishable from noise.
        """
        p = self.policy
        b, c = ev.baseline_timing.times_ms, ev.candidate_timing.times_ms
        mb, mc = statistics.fmean(b), statistics.fmean(c)
        sb, sc = statistics.pstdev(b), statistics.pstdev(c)

        difference = mb - mc
        combined_noise = math.sqrt(sb ** 2 + sc ** 2)

        if difference <= 0:
            return CheckResult(
                "separation", False,
                f"candidate mean {mc:.4f} ms is not faster than baseline "
                f"{mb:.4f} ms",
            )

        if difference < p.min_separation_factor * combined_noise:
            return CheckResult(
                "separation", False,
                f"improvement of {difference:.4f} ms is within "
                f"{p.min_separation_factor}x combined noise "
                f"({combined_noise:.4f} ms) -- distributions overlap, so the "
                f"speedup is not established",
            )

        return CheckResult(
            "separation", True,
            f"{mb / mc:.2f}x speedup, {difference / combined_noise:.1f} sigma separation",
        )