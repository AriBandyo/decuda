import csv, datetime
from pathlib import Path

SCHEMA_VERSION = "1"
HARNESS_VERSION = "2026.08.30"


FIELDS = [

    "schema_version", "harness_version", "timestamp",
    "kernel", "device", "rocm_version", "arch", "peak_bandwidth_gb_s",
    "warmup", "samples_per_side", "kernel_launches_per_sample",
    "iteration", "accepted",
    "baseline_ms", "candidate_ms", "speedup",
    "correctness_passed", "timing_passed",
    "check_name", "check_passed", "check_detail",
]

def write_run_csv (result, out_dir, * , kernel, device, rocm_version,arch,
                   peak_bandwidth_gb_s, warmup,samples_per_side, kernel_launches_per_sample):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exixst_ok= True)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f" {stamp}_{kernel}_iterations.csv"


    with path.open("w", newline = "")as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for it in result.iterations:
            base = {
                "schema_version": SCHEMA_VERSION,
                "harness_version": HARNESS_VERSION,
                "timestamp": stamp,
                "kernel": kernel,
                "device": device,
                "rocm_version": rocm_version,
                "arch": arch,
                "peak_bandwidth_gb_s": peak_bandwidth_gb_s,
                "warmup": warmup,
                "samples_per_side": samples_per_side,
                "kernel_launches_per_sample": kernel_launches_per_sample,
                "iteration": it.number,
                "accepted": it.accepted,
                "baseline_ms": it.baseline_mean_ms,
                "candidate_ms": it.candidate_mean_ms,
                "speedup": (
                    it.baseline_mean_ms / it.candidate_mean_ms
                    if it.baseline_mean_ms and it.candidate_mean_ms else None
                ),
                "correctness_passed": it.correctness_verdict.accepted,
                "timing_passed": (
                    it.timing_verdict.accepted if it.timing_verdict else None
                ),

            }
            checks = list(it.correctness_verdict.checks)
            if it.timing_verdict:
                checks += list(it.timing_verdict.checks)

            for c in checks:
                w.writerow({**base, "check_name": c.name,
                            "check_passed": c.passed, "check_detail": c.detail})


    return path