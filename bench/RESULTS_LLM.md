# LLM planner results

Not yet run. LLM planners need API keys or a GPU-served open model. When measured, the results go here,
in the same format as the CPU tables in the README.

```bash
export ANTHROPIC_API_KEY=...            # and/or OPENAI_API_KEY; start vLLM for open models
sciagent bench --planners bench/planners.example.yaml --seeds 10 --budget 300 --keep-runs --out bench/results_llm_b300
```

Record for each model: provider and model id, the date run, the prices used for `price_*_per_mtok`, and the
`results_digest` of one run, so that it can be replayed from its cassette.
