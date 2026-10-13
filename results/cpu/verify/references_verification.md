# Reference verification

Device: `cpu`

## reactor (exact)

Optimum 0.6522580256 at {"T": 320.50441, "tau": 6.81895, "cat": 0.59153}. Scaled gradient norm 1.34e-06; Hessian eigenvalues -31.8, -0.177, -0.121 (strict local maximum).

## heat (best known, grid64 task definition)

Reference -0.50916528. 5 independent global searches found -0.50916528, -0.50916528, -0.50916528, -0.50916528, -0.50916528 (spread 3.66e-10; reference minus best search -8.84e-11).

| grid n | objective at reference design | CG iterations | rel. residual | seconds |
|---|---|---|---|---|
| 32 | -0.48519703 | 125 | 3.4e-11 | 0.01 |
| 64 | -0.50916528 | 250 | 3.6e-11 | 0.03 |
| 128 | -0.52176086 | 500 | 5.1e-11 | 0.15 |
| 256 | -0.52139968 | 1000 | 6.0e-11 | 0.66 |
| 512 | -0.52134856 | 2000 | 8.1e-11 | 8.19 |

Observed order 2.82; Richardson continuum estimate -0.52135699; discretisation error of the grid64 definition +1.22e-02.

## mimo (40k-draw estimator as task definition)

High-precision value at the reference design: 14.447904 ± 5.9e-04 (SE, 2,000,000 draws, 11.6 s). The 40k estimator differs by -8.98e-04.

Runner-up design n_tx=7 at 23.25 dBm. Gap 0.0480 ± 0.0005 (batch-means SE over 20 independent batches; z = 97.3).

