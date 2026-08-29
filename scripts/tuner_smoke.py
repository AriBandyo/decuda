import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent.tuner import TunerAgent

BASELINE = pathlib.Path("kernels/softmax/softmax_baseline.hip.cpp")

def main():
    src = BASELINE.read_text()
    agent = TunerAgent()
    out = agent.propose(src)

    
    print(out)


    checks = {
        "non-empty": bool(out.strip()),
        "has kernel": "__global__" in out,
        "has hip include": "hip_runtime" in out,
        "prints json": '"' in out and "printf" in out,
        "no fence leakage": "```" not in out,
        "not identical to baseline": out.strip() != src.strip(),
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")

    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/tuner_smoke_candidate.hip.cpp").write_text(out)

if __name__ == "__main__":
    main()