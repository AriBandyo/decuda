from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from verify.harness import Harness
from verify.referee import CorrectnessReferee, Verdict
from verify.timing_referee import TimingReferee
from .tuner import TunerAgent, Attempt


@dataclass
class Iteration:
    number: int
    candidate_source: str
    accepted: bool
    correctness_verdict: Verdict
    timing_verdict: Verdict | None   # None if correctness rejected first
    baseline_mean_ms: float | None = None
    candidate_mean_ms: float | None = None


@dataclass
class LoopResult:
    iterations: list[Iteration] = field(default_factory=list)
    accepted_source: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.accepted_source is not None


class TuningLoop:

    def __init__(self, agent: TunerAgent, harness: Harness,
                 correctness: CorrectnessReferee | None = None,
                 timing: TimingReferee | None = None,
                 max_iterations: int = 5):
        self.agent = agent
        self.harness = harness
        self.correctness = correctness or CorrectnessReferee()
        self.timing = timing or TimingReferee()
        self.max_iterations = max_iterations

    def run(self, kernel_name: str, baseline_source_path: Path,
            work_dir: Path, bytes_moved: int | None = None,
            **env_kwargs) -> LoopResult:

        baseline_source = baseline_source_path.read_text()
        work_dir.mkdir(parents=True, exist_ok=True)

        result = LoopResult()
        history: list[Attempt] = []

        for n in range(1, self.max_iterations + 1):
            candidate_source = self.agent.propose(baseline_source, history)

            candidate_path = work_dir / f"{kernel_name}_candidate_{n}.hip.cpp"
            candidate_path.write_text(candidate_source)

            try:
                evidence = self.harness.collect(
                    kernel_name=kernel_name,
                    baseline_source=baseline_source_path,
                    candidate_source=candidate_path,
                    build_dir=work_dir / "build",
                    **env_kwargs,
                )
            except RuntimeError as exc:
                history.append(Attempt(
                    source=candidate_source,
                    rejection=f"Candidate did not build. {exc}",
                ))
                continue

            correctness_verdict = self.correctness.adjudicate(evidence)

            timing_verdict = None
            if correctness_verdict.accepted:
                timing_verdict = self.timing.adjudicate(
                    evidence, bytes_moved=bytes_moved)

            accepted = (correctness_verdict.accepted
                        and timing_verdict is not None
                        and timing_verdict.accepted)

            result.iterations.append(Iteration(
                number=n,
                candidate_source=candidate_source,
                accepted=accepted,
                correctness_verdict=correctness_verdict,
                timing_verdict=timing_verdict,
                baseline_mean_ms=_mean_or_none(evidence.baseline_timing.times_ms),
                candidate_mean_ms=_mean_or_none(evidence.candidate_timing.times_ms),
            ))

            if accepted:
                result.accepted_source = candidate_source
                return result

            history.append(Attempt(
                source=candidate_source,
                rejection=_combined_feedback(correctness_verdict, timing_verdict),
            ))

        return result


def _mean_or_none(times):
    return sum(times) / len(times) if times else None


def _combined_feedback(correctness: Verdict, timing: Verdict | None) -> str:
    """What the agent sees. Failure descriptions only -- never thresholds,
    never what a passing value would be."""
    parts = [correctness.feedback()]
    if timing is not None:
        parts.append(timing.feedback())
    return "\n".join(parts)