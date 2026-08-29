import math

import pytest

from verify.evidence import (
    Evidence, ExecutionRecord, TimingSample, EnvironmentRecord,
)
from verify.referee import CorrectnessReferee


COLS = 4


def _softmax_row(vals):
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps)
    return [e / s for e in exps]


def _make_evidence(candidate_output, candidate_returncode=0, baseline_output=None):
    if baseline_output is None:
        baseline_output = tuple(
            _softmax_row([1.0, 2.0, 3.0, 4.0]) + _softmax_row([0.5, 0.5, 0.5, 0.5])
        )

    env = EnvironmentRecord("test-device", "rocm-test", "testhost")
    ok = ExecutionRecord("baseline.hip.cpp", ("hipcc", "-O3"), 0, "", "")
    cand = ExecutionRecord("candidate.hip.cpp", ("hipcc", "-O3"),
                           candidate_returncode, "", "segfault" if candidate_returncode else "")
    timing = TimingSample((1.0, 1.0, 1.0), 5, 3)

    return Evidence(
        kernel_name="softmax",
        environment=env,
        baseline_execution=ok,
        candidate_execution=cand,
        baseline_timing=timing,
        candidate_timing=timing,
        baseline_output=baseline_output,
        candidate_output=tuple(candidate_output),
    )


def _failed_check_names(verdict):
    return {c.name for c in verdict.failures}


def test_accepts_an_honest_candidate():
    """A referee that rejects everything catches every fake and is useless."""
    ev = _make_evidence(_make_evidence(()).baseline_output)
    verdict = CorrectnessReferee().adjudicate(ev, cols=COLS)
    assert verdict.accepted, verdict.feedback()


def test_rejects_truncated_output():
    """Fast because it computed fewer elements."""
    full = _make_evidence(()).baseline_output
    ev = _make_evidence(full[:COLS])
    verdict = CorrectnessReferee().adjudicate(ev, cols=COLS)
    assert not verdict.accepted
    assert "shape" in _failed_check_names(verdict)


def test_rejects_nan_output():
    """NaN compares false to everything. If the magnitude check ran first,
    this would pass with zero recorded error."""
    full = list(_make_evidence(()).baseline_output)
    full[3] = float("nan")
    ev = _make_evidence(full)
    verdict = CorrectnessReferee().adjudicate(ev, cols=COLS)
    assert not verdict.accepted
    assert "finite" in _failed_check_names(verdict)
    assert "magnitude" not in _failed_check_names(verdict)


def test_rejects_grossly_unnormalized_output():
    """Skipped normalization entirely. Ordinary element-wise comparison
    catches this one -- included to show where the cheap check does suffice."""
    unnormalized = []
    for row in ([1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]):
        m = max(row)
        unnormalized += [math.exp(v - m) for v in row]

    verdict = CorrectnessReferee().adjudicate(_make_evidence(unnormalized), cols=COLS)
    assert not verdict.accepted
    assert "magnitude" in _failed_check_names(verdict)
    
def test_rejects_output_that_is_elementwise_close_but_not_a_distribution():
    """The fake that element-wise comparison cannot catch.

    Every element is within the magnitude tolerance of baseline, so a
    self-grading agent comparing outputs would accept it. But the rows no
    longer sum to 1, so the output is not a probability distribution. Only
    checking a property the algorithm must satisfy catches this.
    """
    baseline = list(_make_evidence(()).baseline_output)

    # Scale each element by a factor small enough that no single element
    # moves more than the magnitude tolerance, but the row sum does.
    scale = 1.0 + 3e-5
    candidate = [v * scale for v in baseline]

    ref = CorrectnessReferee(tolerance=1e-4, row_sum_tolerance=1e-6)
    verdict = ref.adjudicate(_make_evidence(candidate), cols=COLS)

    # Magnitude must NOT be what catches it -- that is the whole point.
    assert "magnitude" not in _failed_check_names(verdict)
    assert "row_sums" in _failed_check_names(verdict)
    assert not verdict.accepted

def test_rejects_candidate_that_crashed():
    """Nonzero exit is not a performance result."""
    full = _make_evidence(()).baseline_output
    ev = _make_evidence(full, candidate_returncode=139)
    verdict = CorrectnessReferee().adjudicate(ev, cols=COLS)
    assert not verdict.accepted
    assert "execution" in _failed_check_names(verdict)


def test_feedback_names_the_failing_check():
    """Rejections must be actionable -- at step 10 this text is what the
    agent gets back."""
    ev = _make_evidence(_make_evidence(()).baseline_output[:COLS])
    verdict = CorrectnessReferee().adjudicate(ev, cols=COLS)
    assert "REJECTED" in verdict.feedback()
    assert "shape" in verdict.feedback()