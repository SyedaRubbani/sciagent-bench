# LLM planner results

LLM planner results are produced by the GPU pipeline (`notebooks/kaggle_gpu_experiments.ipynb`, steps
`bench_llm` and `bench_distill`) and merged into `docs/TECHNICAL_REPORT.md` with
`sciagent report results/cpu results/gpu`. They are not run yet.

To benchmark API models instead of open models:

```bash
export ANTHROPIC_API_KEY=...            # and/or OPENAI_API_KEY
sciagent bench --planners bench/planners.example.yaml --seeds 10 --budget 300 --parallel 4 --keep-runs --out results/api/bench_llm
sciagent analyze results/api/bench_llm --out results/api/bench_llm/analysis
```

Record for each model the provider and model id, the date, the prices used for `price_*_per_mtok`, and keep the
run directories: their cassettes make every result replayable.
