import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


from pathlib import Path
from agent.tuner import TunerAgent
from agent.loop import TuningLoop
from verify.harness import Harness
from backend.rocm import ROCmBackend


from verify.record import write_run_csv
from verify.changelog import render

# softmax reads the input once and writes the output once, at minimum.
# 1024 * 1024 * 4 bytes, times two.
BYTES_MOVED = 2 * 1024 * 1024 * 4

def main():
    loop = TuningLoop(
        agent=TunerAgent(),
        harness=Harness(ROCmBackend(), warmup=20, iterations=200),
        max_iterations=5,
    )

    result = loop.run(
        kernel_name="softmax",
        baseline_source_path=Path("kernels/softmax/softmax_baseline.hip.cpp"),
        work_dir=Path("/tmp/softmax_run"),
        bytes_moved=BYTES_MOVED,
        device_name="AMD Instinct MI300X",
        rocm_version="10.0.0",
        peak_bandwidth_gb_s=5300.0,
    )

    print("=" * 60)
    print(f"attempts: {len(result.iterations)}")
    for it in result.iterations:
        print(f"--- iteration {it.number}: accepted={it.accepted}")
        print(f"    baseline_ms={it.baseline_mean_ms} candidate_ms={it.candidate_mean_ms}")
        print(f"    correctness: {it.correctness_verdict}")
        print(f"    timing: {it.timing_verdict}")

    csv_path = write_run_csv(
        result,
        out_dir=Path("runs"),
        kernel="softmax",
        device="AMD Instinct MI300X",
        rocm_version="10.0.0",
        arch="gfx942",
        peak_bandwidth_gb_s=5300.0,
        warmup=20,
        samples_per_side=200,
        kernel_launches_per_sample=20,
    )
    print(f"record written: {csv_path}")

    baseline_text = Path("kernels/softmax/softmax_baseline.hip.cpp").read_text()
    Path("runs").mkdir(exist_ok=True)
    Path("runs/CHANGELOG.md").write_text(
        render(result, baseline_text, kernel_name="softmax")
    )

    traj_dir = Path("runs/trajectories")
    traj_dir.mkdir(parents=True, exist_ok=True)
    for it in result.iterations:
        (traj_dir / f"candidate_{it.number}.hip.cpp").write_text(it.candidate_source)

if __name__ == "__main__":
    main()
