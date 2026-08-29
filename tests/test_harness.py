"""Harness tests. The backend is mocked -- what is under test is whether
evidence is COLLECTED correctly, not whether HIP works.
"""
import json
from pathlib import Path

from backend.base import Backend, RunResult
from verify.harness import Harness


class FakeBackend(Backend):
    """Records every call so the tests can assert on ordering."""

    def __init__(self, stdout_by_binary=None, returncode_by_binary=None):
        self.compile_calls = []
        self.run_calls = []
        self.stdout_by_binary = stdout_by_binary or {}
        self.returncode_by_binary = returncode_by_binary or {}

    def compile(self, source: Path, output: Path) -> Path:
        self.compile_calls.append((source, output))
        return output

    def run(self, executable: Path) -> RunResult:
        name = executable.name
        self.run_calls.append(name)
        return RunResult(
            returncode=self.returncode_by_binary.get(name, 0),
            stdout=self.stdout_by_binary.get(name, '{"kernel":"k","correct":true}'),
            stderr="",
        )


def _harness(backend, warmup=2, iterations=3):
    return Harness(backend, warmup=warmup, iterations=iterations)


def _collect(backend, tmp_path, **kwargs):
    return _harness(backend, **kwargs).collect(
        kernel_name="k",
        baseline_source=Path("base.hip.cpp"),
        candidate_source=Path("cand.hip.cpp"),
        build_dir=tmp_path / "build",
    )


def test_compiles_both_sources(tmp_path):
    backend = FakeBackend()
    _collect(backend, tmp_path)
    assert len(backend.compile_calls) == 2


def test_timing_runs_are_interleaved(tmp_path):
    """Not baseline-then-candidate in blocks. Whichever ran second would get
    a hotter GPU and systematically different clocks."""
    backend = FakeBackend()
    _collect(backend, tmp_path, warmup=2, iterations=3)

    # Drop the two initial correctness runs, then warmup, then measured.
    timed = backend.run_calls[2:]
    measured = timed[2 * 2:]  # skip warmup pairs

    assert measured == [
        "k_baseline", "k_candidate",
        "k_baseline", "k_candidate",
        "k_baseline", "k_candidate",
    ], backend.run_calls


def test_warmup_iterations_are_not_recorded(tmp_path):
    """Warmup must be discarded, not averaged in."""
    backend = FakeBackend()
    ev = _collect(backend, tmp_path, warmup=5, iterations=3)

    assert len(ev.baseline_timing.times_ms) == 3
    assert len(ev.candidate_timing.times_ms) == 3
    assert ev.baseline_timing.warmup_iterations == 5


def test_crashed_candidate_still_produces_evidence(tmp_path):
    """A segfaulting candidate is normal agent output. The evidence must
    survive so the referee can reject it with a reason."""
    backend = FakeBackend(returncode_by_binary={"k_candidate": 139})
    ev = _collect(backend, tmp_path)

    assert ev.candidate_execution.returncode == 139
    assert ev.baseline_execution.returncode == 0


def test_missing_output_array_yields_empty_not_fabricated(tmp_path):
    """Absent evidence is not passing evidence."""
    backend = FakeBackend()
    ev = _collect(backend, tmp_path)
    assert ev.candidate_output == ()


def test_output_array_is_parsed_when_present(tmp_path):
    payload = json.dumps({"kernel": "k", "correct": True,
                          "output": [0.25, 0.25, 0.25, 0.25]})
    backend = FakeBackend(stdout_by_binary={
        "k_baseline": payload, "k_candidate": payload})
    ev = _collect(backend, tmp_path)

    assert ev.candidate_output == (0.25, 0.25, 0.25, 0.25)


def test_evidence_contains_no_conclusions(tmp_path):
    """No speedup, no verdict, no passed flag. If a conclusion lived here,
    something other than the referee would have made the judgment."""
    backend = FakeBackend()
    ev = _collect(backend, tmp_path)

    forbidden = {"speedup", "passed", "accepted", "is_faster", "verdict"}
    assert forbidden.isdisjoint(vars(ev).keys())