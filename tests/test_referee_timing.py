"""Adversarial timing tests: fabricated measurements representing specific
ways a speedup can be fake. Runs anywhere -- the referee is a pure function.
"""
import random

from verify.evidence import (
    Evidence, ExecutionRecord, TimingSample, EnvironmentRecord,
)
from verify.timing_referee import TimingReferee


def _samples(mean_ms, rel_sd, n=40, seed=0):
    """Plausible-looking timings around a mean, with controlled scatter."""
    rng = random.Random(seed)
    return tuple(rng.gauss(mean_ms, mean_ms * rel_sd) for _ in range(n))


def _make_evidence(baseline_times, candidate_times,
                   warmup=10, peak_bw=0.0):
    env = EnvironmentRecord("test-device", "rocm-test", "testhost", peak_bw)
    rec = ExecutionRecord("k.hip.cpp", ("hipcc", "-O3"), 0, "", "")

    return Evidence(
        kernel_name="softmax",
        environment=env,
        baseline_execution=rec,
        candidate_execution=rec,
        baseline_timing=TimingSample(baseline_times, warmup, len(baseline_times)),
        candidate_timing=TimingSample(candidate_times, warmup, len(candidate_times)),
    )


def _failed(verdict):
    return {c.name for c in verdict.failures}


def test_accepts_a_genuine_speedup():
    """A referee that rejects everything catches every fake and is useless."""
    ev = _make_evidence(_samples(100.0, 0.01, seed=1),
                        _samples(70.0, 0.01, seed=2))
    v = TimingReferee().adjudicate(ev)
    assert v.accepted, v.feedback()


def test_rejects_too_few_samples():
    """Three measurements are an anecdote, not a distribution."""
    ev = _make_evidence((100.0, 99.0, 101.0), (70.0, 71.0, 69.0))
    v = TimingReferee().adjudicate(ev)
    assert not v.accepted
    assert "sample_count" in _failed(v)


def test_rejects_missing_device_synchronize():
    """The most common fake in GPU work, and usually honest error: the timing
    bracket stopped before hipDeviceSynchronize, so it measured async dispatch
    rather than the kernel."""
    ev = _make_evidence(_samples(100.0, 0.01, seed=1),
                        _samples(0.0002, 0.01, seed=2))
    v = TimingReferee().adjudicate(ev)
    assert not v.accepted
    assert "plausible_duration" in _failed(v)


def test_rejects_identical_samples():
    """Real hardware does not produce forty identical measurements."""
    ev = _make_evidence(tuple([100.0] * 40), tuple([70.0] * 40))
    v = TimingReferee().adjudicate(ev)
    assert not v.accepted
    assert "not_degenerate" in _failed(v)


def test_rejects_measurement_too_noisy_to_conclude_from():
    ev = _make_evidence(_samples(100.0, 0.40, seed=1),
                        _samples(70.0, 0.40, seed=2))
    v = TimingReferee().adjudicate(ev)
    assert not v.accepted
    assert "noise_level" in _failed(v)


def test_rejects_speedup_inside_the_noise():
    """The one that catches honest self-deception rather than fraud.
    Comparing means alone would call this a 5% win."""
    ev = _make_evidence(_samples(100.0, 0.08, seed=1),
                        _samples(95.0, 0.08, seed=2))
    v = TimingReferee().adjudicate(ev)
    assert not v.accepted
    assert "separation" in _failed(v)


def test_rejects_candidate_that_is_not_faster():
    ev = _make_evidence(_samples(100.0, 0.01, seed=1),
                        _samples(110.0, 0.01, seed=2))
    v = TimingReferee().adjudicate(ev)
    assert not v.accepted
    assert "separation" in _failed(v)


def test_rejects_insufficient_warmup():
    """A dramatically slower first sample means cold-cache cost is polluting
    the mean."""
    base = list(_samples(100.0, 0.01, seed=1))
    cand = list(_samples(70.0, 0.01, seed=2))
    cand[0] = 400.0
    ev = _make_evidence(tuple(base), tuple(cand), warmup=10)
    v = TimingReferee().adjudicate(ev)
    assert not v.accepted
    assert "warmup" in _failed(v)


def test_rejects_impossible_bandwidth():
    """The physics check, and the case where only physics catches it."""
    ev = _make_evidence(_samples(100.0, 0.01, seed=1),
                        _samples(0.005, 0.01, seed=2),
                        peak_bw=5300.0)
    v = TimingReferee().adjudicate(ev, bytes_moved=256 * 1024 * 1024)

    # The coarse floor does NOT catch this -- that is the point.
    assert "plausible_duration" not in _failed(v)
    assert "bandwidth_ceiling" in _failed(v)
    assert not v.accepted


def test_bandwidth_check_skipped_when_device_unknown():
    """A check that cannot run is skipped, not failed -- otherwise nothing
    would ever pass on a machine without ROCm."""
    ev = _make_evidence(_samples(100.0, 0.01, seed=1),
                        _samples(70.0, 0.01, seed=2), peak_bw=0.0)
    v = TimingReferee().adjudicate(ev, bytes_moved=256 * 1024 * 1024)
    assert "bandwidth_ceiling" not in {c.name for c in v.checks}
    assert v.accepted