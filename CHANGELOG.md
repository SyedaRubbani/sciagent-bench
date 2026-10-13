# Changelog

## 0.2.0

Research-grade release, organised in four phases.

**Phase 1: scientific benchmark foundation**
- Solver verification tests: Euler first order, RK4 fourth order, SISO Rayleigh capacity against its closed form,
  Monte-Carlo unbiasedness, heat-solver positivity; torch kernels checked against NumPy/SciPy.
- `sciagent verify-references`: reactor stationarity and Hessian check; heat optimiser robustness over independent
  global searches and a grid-convergence study with Richardson extrapolation; MIMO high-precision value with standard
  error and a batch-means test against the runner-up design. GPU-capable.
- Reserve/commit `BudgetLedger`; property-based tests (Hypothesis) show no proposal can exceed the budget or bounds.
  The LLM USD cap is enforced before each call with a worst-case estimate.
- Study specs carry `schema_version`; `sciagent schema` prints the JSON Schema.

**Phase 2: adaptive agent and inspectability**
- Redirect: switch the planner mid-run (CLI, API, viewer, MCP).
- Crash-safe `sciagent resume`: fast-forwards recorded runs and re-applies steering at its original batch; identical
  results for all four planner types after a simulated preemption.
- Bearer-token auth for the HTTP API (`SCIAGENT_API_TOKEN`); the viewer asks for the token.
- A fake OpenAI-compatible server for testing the LLM planner over real HTTP, including retries on 503.

**Phase 3: evaluation and model optimisation**
- `sciagent analyze`: bootstrap CIs, paired Wilcoxon tests with Holm correction, win rates, success rates,
  anytime AUC.
- `sciagent bench --parallel` (identical results, lets vLLM batch requests).
- GPU kernels: batched Monte-Carlo capacity and matrix-free PCG heat solver in PyTorch; `sciagent gpu-bench`.
- GPU surrogate training; transformers-based OpenAI-compatible server with dynamic batching (vLLM fallback);
  server launcher that installs vLLM in its own virtualenv.
- `sciagent experiments`: resumable pipeline with `kaggle-t4`, `quick`, `cpu-smoke` and `cpu-full` profiles;
  Kaggle and Colab notebooks generated from one definition.

**Phase 4: researcher-ready release**
- `sciagent report` builds the technical report from result files.
- CI: coverage, a CPU smoke run of the experiment pipeline, notebook freshness check, Docker smoke test.

## 0.1.0

Initial release: three simulators, four planners, guard, tracing, record/replay, steering, FastAPI viewer, MCP
server, Slurm executor, benchmark harness, surrogate distillation and quantisation, LoRA recipe.
