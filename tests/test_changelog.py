from agent.loop import Iteration, LoopResult
from verify.changelog import render
from verify.referee import CheckResult, Verdict


def _verdict(accepted, checks):
    return Verdict(accepted=accepted, checks=tuple(checks))


def _passing():
    return _verdict(True, [
        CheckResult("shape", True, "8 elements"),
        CheckResult("row_sums", True, "all rows sum to 1.0"),
    ])


def _failing(name="row_sums", detail="row 47 sums to 0.9994, not 1.0"):
    return _verdict(False, [
        CheckResult("shape", True, "8 elements"),
        CheckResult(name, False, detail),
    ])


def _iteration(n, accepted, correctness, timing=None):
    return Iteration(
        number=n, candidate_source="// candidate\n// extra line",
        accepted=accepted,
        correctness_verdict=correctness, timing_verdict=timing,
        baseline_mean_ms=100.0, candidate_mean_ms=70.0,
    )


BASELINE = "// baseline"


def test_renders_a_row_per_stage():
    result = LoopResult(iterations=[
        _iteration(1, False, _failing()),
        _iteration(2, True, _passing(), _passing()),
    ], accepted_source="// candidate")

    table = render(result, BASELINE)
    lines = [l for l in table.splitlines() if l.startswith("|")]

    # header + separator + baseline + 2 iterations + final
    assert len(lines) == 6


def test_rejected_row_names_the_failing_check():
    """The changelog is the evidence trail. A row saying only 'rejected'
    records nothing."""
    result = LoopResult(iterations=[_iteration(1, False, _failing())])
    table = render(result, BASELINE)

    assert "row_sums" in table
    assert "0.9994" in table


def test_no_acceptance_is_rendered_as_a_result_not_a_gap():
    """If the referee rejects everything, that is the system working."""
    result = LoopResult(iterations=[
        _iteration(1, False, _failing()),
        _iteration(2, False, _failing("magnitude", "max abs error 3.2e-02")),
    ])
    table = render(result, BASELINE)

    assert "No candidate accepted" in table
    assert "the system working" in table


def test_baseline_row_states_the_fairness_claim():
    result = LoopResult(iterations=[])
    table = render(result, BASELINE)

    assert "hipcc -O3" in table
    assert "No hand tuning" in table


def test_diff_summary_is_structural_not_narrated():
    """Line counts, not the agent's account of what it did."""
    result = LoopResult(iterations=[_iteration(1, False, _failing())])
    table = render(result, BASELINE)

    assert "line(s) added" in table


def test_timing_numbers_appear_when_present():
    result = LoopResult(iterations=[
        _iteration(1, True, _passing(), _passing()),
    ], accepted_source="// candidate")
    table = render(result, BASELINE)

    assert "100.0000 ms" in table
    assert "70.0000 ms" in table