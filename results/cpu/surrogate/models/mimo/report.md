# Surrogate report: `mimo`

Held-out test set: 1245 rows (403 at top fidelity), split by design so no test configuration was seen in training. Latency is single-threaded CPU, median.

Parameters: teacher 33,921, student 1,281 (26x smaller).

| model | MAE (all) | R² (all) | MAE (top fid.) | R² (top fid.) | size | latency b=1 (µs) | latency b=4096 (ms) |
|---|---|---|---|---|---|---|---|
| teacher (torch fp32) | 0.03671 | 0.9998 | 0.01792 | 1.0000 | 135.5 KiB | 53.9 | 6.71 |
| student distilled (torch fp32) | 0.03684 | 0.9998 | 0.01861 | 1.0000 | 7.6 KiB | 62.4 | 0.67 |
| student from scratch (torch fp32) | 0.03289 | 0.9998 | 0.01238 | 1.0000 | 8.1 KiB | 57.3 | 0.49 |
| student distilled (torch int8 dynamic) | 0.1692 | 0.9974 | 0.1619 | 0.9978 | 6.0 KiB | 174.6 | 0.92 |
| student distilled (ONNX Runtime fp32) | 0.03684 | 0.9998 | 0.01861 | 1.0000 | 5.8 KiB | 11.4 | 0.36 |
| student distilled (ONNX Runtime int8) | 0.1282 | 0.9985 | 0.1114 | 0.9990 | 4.7 KiB | 26.5 | 0.70 |

One top-fidelity simulator call: 16020 µs. The fastest surrogate variant amortises to 0.09 µs per prediction in batch, i.e. ~181,013x cheaper per query.

Trained on cpu in 98 s. Environment: torch 2.14.1+cu130, onnxruntime 1.29.0.
