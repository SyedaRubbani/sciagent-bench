# Surrogate report: `reactor`

Held-out test set: 1004 rows (415 at top fidelity), split by design so no test configuration was seen in training. Latency is single-threaded CPU, median.

Parameters: teacher 34,049, student 1,313 (26x smaller).

| model | MAE (all) | R² (all) | MAE (top fid.) | R² (top fid.) | size | latency b=1 (µs) | latency b=4096 (ms) |
|---|---|---|---|---|---|---|---|
| teacher (torch fp32) | 0.006153 | 0.9980 | 0.005145 | 0.9981 | 136.0 KiB | 53.0 | 11.39 |
| student distilled (torch fp32) | 0.006707 | 0.9979 | 0.005694 | 0.9979 | 7.8 KiB | 38.8 | 0.47 |
| student from scratch (torch fp32) | 0.01179 | 0.9947 | 0.01034 | 0.9948 | 8.2 KiB | 39.1 | 0.48 |
| student distilled (torch int8 dynamic) | 0.02073 | 0.9914 | 0.01948 | 0.9926 | 6.0 KiB | 143.7 | 0.97 |
| student distilled (ONNX Runtime fp32) | 0.006707 | 0.9979 | 0.005694 | 0.9979 | 5.9 KiB | 18.8 | 0.49 |
| student distilled (ONNX Runtime int8) | 0.01662 | 0.9927 | 0.01194 | 0.9962 | 4.8 KiB | 17.1 | 0.78 |

One top-fidelity simulator call: 48358 µs. The fastest surrogate variant amortises to 0.12 µs per prediction in batch, i.e. ~420,407x cheaper per query.

Environment: torch 2.14.1+cu130, onnxruntime 1.29.0.
