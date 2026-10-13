# Data card: `reactor` surrogate dataset

*Simulator* reactor v1.0 - Isothermal batch reactor with consecutive first-order reactions A->B->C. Maximise yield of the intermediate B minus heating, catalyst and lost-throughput (time) costs. Higher temperature speeds both reactions but favours the unwanted B->C step (higher activation energy).

*Rows* 10035  *SHA-256* `9bdc0d5641b51847...`

## Curation

| step | count |
|---|---|
| generated | 12288 |
| from_runs | 0 |
| dropped_failed | 0 |
| dropped_nonfinite | 0 |
| dropped_duplicate | 0 |
| dropped_unstable | 2253 |

## Splits (by configuration hash - no design leaks between splits)

| split | rows |
|---|---|
| train | 7995 |
| val | 1060 |
| test | 980 |

## Rows per fidelity

| fidelity | rows |
|---|---|
| euler_coarse | 2128 |
| rk4_medium | 3811 |
| bdf_fine | 4096 |

## Features

| name | range | unit |
|---|---|---|
| T | [300, 420] | K |
| tau | [0.05, 10] | min |
| cat | [0, 1] | fraction |
| fidelity | one-hot over levels | |

Target: `net_yield_B` (range -0.2318 to 0.6992).
