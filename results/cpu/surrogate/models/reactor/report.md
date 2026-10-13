# Surrogate report: `reactor`

Held-out test set: 980 rows (413 at top fidelity), split by design so no test configuration was seen in training. Latency is single-threaded CPU, median.

Parameters: teacher 34,049, student 1,313 (26x smaller).

| model | MAE (all) | R² (all) | MAE (top fid.) | R² (top fid.) | size | latency b=1 (µs) | latency b=4096 (ms) |
|---|---|---|---|---|---|---|---|
| teacher (torch fp32) | 0.005514 | 0.9985 | 0.004521 | 0.9986 | 136.0 KiB | 56.7 | 4.44 |
| student distilled (torch fp32) | 0.00563 | 0.9986 | 0.004442 | 0.9988 | 7.8 KiB | 40.5 | 0.46 |
| student from scratch (torch fp32) | 0.009756 | 0.9966 | 0.008241 | 0.9973 | 8.2 KiB | 42.4 | 0.57 |
| student distilled (torch int8 dynamic) | 0.02395 | 0.9875 | 0.02094 | 0.9913 | 6.0 KiB | 132.6 | 0.92 |
| student distilled (ONNX Runtime fp32) | 0.00563 | 0.9986 | 0.004442 | 0.9988 | 5.9 KiB | 11.6 | 0.33 |
| student distilled (ONNX Runtime int8) | 0.01702 | 0.9931 | 0.0146 | 0.9953 | 4.8 KiB | 15.0 | 0.48 |

One top-fidelity simulator call: 41663 µs. The fastest surrogate variant amortises to 0.08 µs per prediction in batch, i.e. ~513,320x cheaper per query.

Trained on cpu in 80 s. Environment: torch 2.14.1+cu130, onnxruntime 1.29.0.
