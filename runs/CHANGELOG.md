| Stage | What was tried | Evidence | Decision / learning |
|---|---|---|---|
| Baseline | Mechanical HIPIFY translation of softmax.cu, built with `hipcc -O3`. No hand tuning. | Translation diff is API renames only; kernel body byte-identical to the CUDA source (asserted by test). | Fair starting point. Any later speedup is measured against a kernel nobody hobbled. |
| Iteration 1 | 113 line(s) added, 76 removed vs baseline | baseline 0.0064 ms, candidate 177.8851 ms; 0/2 checks passed | REJECTED [execution] — candidate exited 2; stderr:  |
| Iteration 2 | 137 line(s) added, 72 removed vs baseline | baseline 0.0064 ms, candidate 178.1415 ms; 0/2 checks passed | REJECTED [execution] — candidate exited 2; stderr:  |
| Iteration 3 | 196 line(s) added, 173 removed vs baseline | baseline 0.0064 ms, candidate 0.0057 ms; 12/12 checks passed | ACCEPTED |
| Final | Accepted candidate from iteration 3. | Passed every correctness and timing check. | 3 proposal(s), 2 rejected before one survived independent verification. |
