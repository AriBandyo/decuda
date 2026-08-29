from __future__ import annotations

import difflib

from agent.loop import LoopResult, Iteration


def render(result: LoopResult, baseline_source: str,
           kernel_name: str = "softmax") -> str:
    rows = [_baseline_row(kernel_name)]
    for it in result.iterations:
        rows.append(_iteration_row(it, baseline_source))
    rows.append(_final_row(result))

    header = (
        "| Stage | What was tried | Evidence | Decision / learning |\n"
        "|---|---|---|---|\n"
    )
    return header + "\n".join(rows) + "\n"


def _baseline_row(kernel_name: str) -> str:
    return (
        "| Baseline "
        f"| Mechanical HIPIFY translation of {kernel_name}.cu, built with "
        "`hipcc -O3`. No hand tuning. "
        "| Translation diff is API renames only; kernel body byte-identical "
        "to the CUDA source (asserted by test). "
        "| Fair starting point. Any later speedup is measured against a "
        "kernel nobody hobbled. |"
    )


def _iteration_row(it: Iteration, baseline_source: str) -> str:
    changed = _summarize_diff(baseline_source, it.candidate_source)
    evidence = _summarize_evidence(it)
    decision = "ACCEPTED" if it.accepted else _first_failure(it)

    return (f"| Iteration {it.number} | {changed} | {evidence} | {decision} |")


def _final_row(result: LoopResult) -> str:
    if result.succeeded:
        n = len(result.iterations)
        return (
            f"| Final | Accepted candidate from iteration {n}. "
            f"| Passed every correctness and timing check. "
            f"| {n} proposal(s), {n - 1} rejected before one survived "
            f"independent verification. |"
        )
    return (
        "| Final | No candidate accepted. "
        f"| {len(result.iterations)} proposals, all rejected. "
        "| A negative result. The referee rejected every candidate, which is "
        "the system working, not failing. |"
    )


def _summarize_diff(baseline: str, candidate: str) -> str:
    """Structural summary, not the agent's account of itself."""
    b_lines = baseline.splitlines()
    c_lines = candidate.splitlines()
    added = removed = 0
    for line in difflib.unified_diff(b_lines, c_lines, lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return f"{added} line(s) added, {removed} removed vs baseline"


def _summarize_evidence(it: Iteration) -> str:
    parts = []
    if it.baseline_mean_ms and it.candidate_mean_ms:
        parts.append(
            f"baseline {it.baseline_mean_ms:.4f} ms, "
            f"candidate {it.candidate_mean_ms:.4f} ms"
        )
    checks = list(it.correctness_verdict.checks)
    if it.timing_verdict is not None:
        checks += list(it.timing_verdict.checks)
    passed = sum(1 for c in checks if c.passed)
    parts.append(f"{passed}/{len(checks)} checks passed")
    return "; ".join(parts)


def _first_failure(it: Iteration) -> str:
    for verdict in (it.correctness_verdict, it.timing_verdict):
        if verdict is None:
            continue
        for check in verdict.failures:
            return f"REJECTED [{check.name}] — {check.detail}"
    return "REJECTED"