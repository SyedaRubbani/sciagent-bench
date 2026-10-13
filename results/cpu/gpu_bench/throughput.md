# Kernel throughput on `cpu`

## MIMO Monte-Carlo capacity (8 x 6 complex channel)

| draws | NumPy (s) | torch fp64 (s) | torch fp32 (s) | best draws/s |
|---|---|---|---|---|
| 65,536 | 0.272 | 0.338 | 0.137 | 477,560 |
| 1,048,576 | 4.866 | 5.517 | 2.890 | 362,803 |
| 8,388,608 | – | 47.496 | 22.873 | 366,749 |

## Heat plate: sparse direct (SciPy) vs matrix-free PCG (torch, fp64, rtol 1e-10)

| grid | unknowns | SciPy direct (s) | torch PCG (s) | PCG iterations |
|---|---|---|---|---|
| 64 | 4,096 | 0.012 | 0.031 | 250 |
| 128 | 16,384 | 0.063 | 0.126 | 500 |
| 256 | 65,536 | 0.448 | 0.560 | 1000 |
| 512 | 262,144 | 2.901 | 6.673 | 2000 |
| 1024 | 1,048,576 | – | 70.791 | 4000 |
