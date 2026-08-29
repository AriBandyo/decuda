import json
from pathlib import Path

from backend.base import Backend, RunResult
from agent.loop import TuningLoop
from agent.tuner import Attempt
from verify.harness import Harness


GOOD_ROWS = [1.0, 1.0]
BAD_ROWS = [0.5, 0.5]


def _payload(row_sums, output=(0.25, 0.25, 0.25, 0.25)):
    return json.dumps({
        "kernel": "softmax", "correct": True,
        "row_sums": list(row_sums), "output": list(output),
    })


class FakeBackend(Backend):
    def __init__(self, candidate_payload, baseline_payload):
        self.candidate_payload = candidate_payload
        self.baseline_payload = baseline_payload

    def compile(self, source: Path, output: Path) -> Path:
        return output

    def run(self, executable: Path) -> RunResult:
        payload = (self.candidate_payload if "candidate" in executable.name
                   else self.baseline_payload)
        return RunResult(returncode=0, stdout=payload, stderr="")


class FakeAgent:
    """Records what it was told, so tests can assert on the feedback boundary."""

    def __init__(self, sources=None):
        self.sources = sources or ["// candidate"]
        self.calls = []

    def propose(self, baseline_source, history=None):
        self.calls.append(list(history or []))
        idx = min(len(self.calls) - 1, len(self.sources) - 1)
        return self.sources[idx]


def _run(tmp_path, candidate_rows, max_iterations=3):
    backend = FakeBackend(_payload(candidate_rows), _payload(GOOD_ROWS))
    agent = FakeAgent()
    baseline = tmp_path / "baseline.hip.cpp"
    baseline.write_text("// baseline")

    loop = TuningLoop(agent, Harness(backend, warmup=1, iterations=2),
                      max_iterations=max_iterations)
    result = loop.run("softmax", baseline, tmp_path / "work")
    return result, agent


def test_exhausts_iterations_when_always_rejected(tmp_path):
    result, agent = _run(tmp_path, BAD_ROWS, max_iterations=3)
    assert not result.succeeded
    assert len(result.iterations) == 3
    assert len(agent.calls) == 3


def test_rejection_history_accumulates(tmp_path):
    """Iteration 3 must see what failed in 1 and 2, or it will re-propose
    the same broken idea."""
    _, agent = _run(tmp_path, BAD_ROWS, max_iterations=3)
    assert len(agent.calls[0]) == 0
    assert len(agent.calls[1]) == 1
    assert len(agent.calls[2]) == 2


def test_incorrect_candidate_is_never_timed(tmp_path):
    """Timing a wrong kernel wastes GPU seconds and can produce an
    impressive-looking number that lands in the logs."""
    result, _ = _run(tmp_path, BAD_ROWS, max_iterations=1)
    it = result.iterations[0]
    assert not it.correctness_verdict.accepted
    assert it.timing_verdict is None


def test_feedback_leaks_no_thresholds(tmp_path):
    """The boundary that keeps the agent honest. If it learns the tolerance,
    it can tune to sit just inside it."""
    _, agent = _run(tmp_path, BAD_ROWS, max_iterations=2)
    feedback = agent.calls[1][0].rejection

    for leak in ("tolerance", "min_separation", "row_sum_tolerance",
                 "max_relative_stddev", "policy"):
        assert leak not in feedback, f"feedback leaked '{leak}': {feedback}"


def test_records_an_iteration_per_attempt(tmp_path):
    result, _ = _run(tmp_path, BAD_ROWS, max_iterations=2)
    assert [it.number for it in result.iterations] == [1, 2]
    assert all(it.candidate_source for it in result.iterations)