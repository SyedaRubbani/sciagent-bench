# Data card: `reactor` surrogate dataset

*Simulator* reactor v1.0 - Isothermal batch reactor with consecutive first-order reactions A->B->C. Maximise yield of the intermediate B minus heating, catalyst and lost-throughput (time) costs. Higher temperature speeds both reactions but favours the unwanted B->C step (higher activation energy).

*Rows* 10209  *SHA-256* `c8c8bbb54f5e88c6...`

## Curation

| step | count |
|---|---|
| generated | 12288 |
| from_runs | 215 |
| dropped_failed | 0 |
| dropped_nonfinite | 0 |
| dropped_duplicate | 2 |
| dropped_unstable | 2292 |

## Splits (by configuration hash - no design leaks between splits)

| split | rows |
|---|---|
| train | 8135 |
| val | 1070 |
| test | 1004 |

## Rows per fidelity

| fidelity | rows |
|---|---|
| euler_coarse | 2235 |
| rk4_medium | 3863 |
| bdf_fine | 4111 |

## Features

| name | range | unit |
|---|---|---|
| T | [300, 420] | K |
| tau | [0.05, 10] | min |
| cat | [0, 1] | fraction |
| fidelity | one-hot over levels | |

Target: `net_yield_B` (range -0.2318 to 0.6992).
