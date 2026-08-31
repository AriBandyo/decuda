| Stage | What was tried | Evidence | Decision / learning |
|---|---|---|---|
| Baseline | Mechanical HIPIFY translation of softmax.cu, built with `hipcc -O3`. No hand tuning. | Translation diff is API renames only; kernel body byte-identical to the CUDA source (asserted by test). | Fair starting point. Any later speedup is measured against a kernel nobody hobbled. |
| Iteration 1 | 102 line(s) added, 74 removed vs baseline | baseline 0.0064 ms, candidate 0.0065 ms; 11/12 checks passed | REJECTED [separation] — candidate mean 0.0065 ms is not faster than baseline 0.0064 ms |
| Iteration 2 | 80 line(s) added, 52 removed vs baseline | baseline 0.0064 ms, candidate 0.0058 ms; 12/12 checks passed | ACCEPTED |
| Final | Accepted candidate from iteration 2. | Passed every correctness and timing check. | 2 proposal(s), 1 rejected before one survived independent verification. |
