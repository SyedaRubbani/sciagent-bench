# Data card: `mimo` surrogate dataset

*Simulator* mimo v1.0 - Uplink MIMO link with 8 receive antennas. Choose transmit power (dBm) and the number of active transmit RF chains (1-8) to maximise energy efficiency in Mbit/J. Capacity grows with power and streams, but every active chain costs circuit power. Cheap fidelities are noisy Monte-Carlo estimates (unbiased); the metric 'stderr' reports their uncertainty.

*Rows* 12288  *SHA-256* `f93cd5bfa36133ab...`

## Curation

| step | count |
|---|---|
| generated | 12288 |
| from_runs | 0 |
| dropped_failed | 0 |
| dropped_nonfinite | 0 |
| dropped_duplicate | 0 |
| dropped_unstable | 0 |

## Splits (by configuration hash - no design leaks between splits)

| split | rows |
|---|---|
| train | 9769 |
| val | 1274 |
| test | 1245 |

## Rows per fidelity

| fidelity | rows |
|---|---|
| mc64 | 4096 |
| mc512 | 4096 |
| mc4096 | 4096 |

## Features

| name | range | unit |
|---|---|---|
| p_dbm | [0, 30] | dBm |
| n_tx | [1, 8] |  |
| fidelity | one-hot over levels | |

Target: `energy_efficiency_Mbit_per_J` (range 0.7709 to 14.67).
