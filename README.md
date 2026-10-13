# SciAgent-Bench

**Budget-aware agentic planning of simulation campaigns, with every decision traceable and every run
reproducible.**

A researcher describes a study in a short YAML file: what can be varied, what counts as better, which
simulator fidelities exist and what each costs, and the budget. A planning agent then decides which
simulations to run next, and at which fidelity. It adapts after every batch, stays inside hard budget and
bounds limits, and can be steered by the researcher while it runs. The same harness benchmarks LLM
planners (closed APIs or open models on your own GPU) against classical baselines on scientific tasks with
known optima.

```
sciagent run studies/reactor.yaml                 # run a campaign (successive halving by default)
sciagent run studies/reactor.yaml --planner llm --record
sciagent replay runs/<dir>                        # re-execute and verify bit-identical results
sciagent serve                                    # API + live viewer; steer, pause or stop studies
sciagent bench --seeds 10 --budget 300            # planners x tasks x seeds -> regret table + plot
sciagent surrogate make-data|train|report         # curate data, distil and quantise a surrogate
sciagent mcp                                      # expose simulators and studies to any MCP client
```

## What is in the box

| Area | What it does | Where |
|---|---|---|
| Planning and orchestration | Plan → guard → execute → observe loop under compute, LLM-cost, wall-clock and batch limits; adapts as results arrive | `orchestrator.py`, `guard.py` |
| Planners | Random search, budget-aware multi-fidelity successive halving, GP Bayesian optimisation (batched EI), LLM agent with tool "advisors" (GP suggestions, learned surrogate), repair loop, LLM-budget fallback to BO | `planners/` |
| Models | Anthropic Messages API and any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama); record/replay cassettes | `llm/providers.py` |
| Scientific tasks | Chemical reactor (closed-form optimum; biased cheap solvers), 2-D heat conduction PDE (grid fidelity), MIMO energy efficiency (noisy Monte-Carlo fidelity) | `simulators/` |
| Inspectability | Parent-linked JSONL trace of every LLM call, plan, guard intervention, run and steering action; manifest with environment and digests; CLI timeline and web viewer | `tracing.py`, `static/viewer.html` |
| Reproducibility | Declared costs, derived seeds, content-addressed cache, results digest, `replay` that fails on any difference | `cache.py`, `cli.py replay` |
| Steering | Notes to the planner, narrowed bounds, budget changes, pause/resume/stop, via CLI, HTTP or MCP | `steering.py`, `service.py`, `mcp_server.py` |
| Compute | In-process, local process pool, or Slurm job arrays via submitit | `executors.py` |
| Evaluation | Regret of the *reported* incumbent against best-known optima; tables, curves, per-model token and USD cost | `bench.py` |
| Data and model optimisation | Data curation with a data card and leakage-free splits, PyTorch teacher, distilled student, int8 quantisation (PyTorch and ONNX Runtime), accuracy/size/latency report | `surrogate/` |
| LLM distillation and serving (GPU) | Teacher traces → curated SFT set → LoRA fine-tune of a 0.5B model → vLLM serving (fp16 and 4-bit) → latency, throughput and planning-quality benchmark | `recipes/distill_planner_lora/` |
| Reuse | Simulator template and guide, study templates, Docker image, CI, tutorial with student projects | `templates/`, `docs/`, `.github/` |

## Results measured so far (CPU)

**Planner benchmark.** Simple regret of the reported incumbent (lower is better), mean ± sd over 10 seeds.
Reproduce with `sciagent bench --seeds 10 --budget 300` (about 1 minute on a laptop).

Budget 300 compute units:

| planner | reactor | heat | mimo |
|---|---|---|---|
| random | 0.0266 ± 0.0271 | 0.0760 ± 0.0402 | 0.4905 ± 0.6095 |
| sh | 0.0104 ± 0.0066 | 0.0493 ± 0.0185 | 0.1018 ± 0.1014 |
| bo | 0.0269 ± 0.0264 | 0.0322 ± 0.0114 | **0.0057 ± 0.0085** |
| scripted-mf | **0.0039 ± 0.0044** | **0.0128 ± 0.0098** | 0.0327 ± 0.0235 |

Budget 120 compute units:

| planner | reactor | heat | mimo |
|---|---|---|---|
| random | 0.1319 ± 0.1238 | 0.1250 ± 0.0514 | 1.4354 ± 1.7204 |
| sh | 0.0157 ± 0.0130 | 0.0649 ± 0.0364 | 0.2505 ± 0.2714 |
| bo | 0.1319 ± 0.1238 | 0.1250 ± 0.0514 | 1.5363 ± 1.7875 |
| scripted-mf | **0.0116 ± 0.0175** | **0.0398 ± 0.0186** | **0.0759 ± 0.0802** |

Regret is in each task's objective units (net yield fraction, normalised temperature, Mbit/J). Median regret curves: `bench/results_b300/regret_curves.png`.

`scripted-mf` is the full LLM-agent code path (prompting, JSON parsing, guard, tracing) driven by a
deterministic mock provider that follows a hand-written explore-refine-confirm heuristic. It is a
pipeline check and a baseline, not a language model. Results for real LLMs are produced with
`sciagent bench --planners bench/planners.example.yaml` and recorded in `bench/RESULTS_LLM.md` once run.

What the table already shows:

- **No planner wins everywhere.** At 300 units BO is strongest on the noisy-but-unbiased MIMO task, while
  the explore-refine-confirm policy wins on reactor and heat. Successive halving beats BO on the reactor
  but not on heat.
- **At 120 units, single-fidelity BO is no better than random search.** It never gets past its initial
  design at 16 units per run, so spending on cheap fidelities matters most when budgets are tight.
- **Cheap fidelities can mislead.** On the reactor, explicit Euler *over*-estimates the yield near the
  optimum and diverges for hot, long runs. That is why the incumbent is defined at the highest fidelity
  evaluated.

**Surrogate distillation and quantisation** (`surrogate_results/report.md`, reactor task, 10,209 curated
rows, held-out designs):

| model | MAE (top fidelity) | R² (top fidelity) | size | latency, batch 1 | latency per prediction, batch 4096 |
|---|---|---|---|---|---|
| teacher (torch fp32) | 0.0051 | 0.9981 | 136.0 KiB | 53 µs | 2.78 µs |
| student distilled (torch fp32) | 0.0057 | 0.9979 | 7.8 KiB | 39 µs | 0.12 µs |
| student from scratch (torch fp32) | 0.0103 | 0.9948 | 8.2 KiB | 39 µs | 0.12 µs |
| student distilled (torch int8 dynamic) | 0.0195 | 0.9926 | 6.0 KiB | 144 µs | 0.24 µs |
| student distilled (ONNX Runtime fp32) | 0.0057 | 0.9979 | 5.9 KiB | 19 µs | 0.12 µs |
| student distilled (ONNX Runtime int8) | 0.0119 | 0.9962 | 4.8 KiB | 17 µs | 0.19 µs |

For comparison, one top-fidelity simulator call takes 48 ms. Latencies are single-threaded CPU medians.

Distillation gives a 26x smaller model that keeps the teacher's accuracy, and it beats the same small
network trained from scratch with the same number of gradient steps. Int8 quantisation does not pay off at
this size: error rises 2-3x and there is no speed-up, because a 1.3k-parameter network is dominated by
overheads rather than by weight memory. The
4-bit path is applied where it pays off, to the 0.5B-parameter LLM planner in the GPU recipe.

## Quick start

```bash
git clone https://github.com/SyedaRubbani/sciagent-bench && cd sciagent-bench
pip install -e ".[dev]"                     # core + tests; add [surrogate,mcp,slurm] as needed
pytest -q                                   # 41 tests, ~25 s
sciagent sims                               # describe the built-in simulators
sciagent run studies/reactor.yaml --out runs/demo
sciagent trace runs/demo | head -30
sciagent serve --runs-dir runs              # open http://127.0.0.1:8000
```

Use a real model:

```bash
export ANTHROPIC_API_KEY=...                # or run an open model:  vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001
sciagent run studies/heat_llm_claude.yaml   # records a cassette; replay it later without the API
sciagent replay runs/<dir>
```

Bring your own simulator: see [`docs/ADDING_A_SIMULATOR.md`](docs/ADDING_A_SIMULATOR.md). A study can
reference `simulator: "my_lab.my_sim:MySimulator"` without changes to this package.

## A study spec

```yaml
name: reactor-yield
simulator: reactor
description: >
  Maximise net yield of B. Catalyst is expensive; we would rather not run hotter than ~380 K.
budget: {compute_units: 300, llm_usd: 1.0, max_batches: 40}
batch_size: 4
bounds: {T: [300, 380]}               # enforced by the guard, can be narrowed live
planner:
  kind: llm
  advisors: [bo, surrogate]           # tool outputs the agent sees each batch
  options: {fallback: bo, surrogate_path: surrogate_models}
  llm: {provider: openai, base_url: "http://localhost:8001/v1", model: Qwen/Qwen2.5-7B-Instruct, mode: record}
executor: {kind: slurm, slurm: {slurm_partition: normal.4h, timeout_min: 10}}
```

## Design

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). In brief: planners only *propose*. A deterministic
guard enforces bounds, fidelities, batch size, budget and de-duplication, and logs every intervention. The
reported answer is the best result at the highest fidelity evaluated. Reproducibility is checked by
`sciagent replay`, which compares results digests, and the test suite runs that check for LLM and non-LLM
planners.

## Status and limitations

- Tested here: everything on CPU (41 tests, lint, the CLI, record/replay, the API and viewer, the MCP
  tools, the Slurm executor via submitit's local backend, the surrogate pipeline). The Dockerfile and the CI
  workflow are written but not yet run.
- Not yet run: the GPU recipe (`recipes/distill_planner_lora`) and benchmarks with real LLMs. They need a GPU
  or API keys, and their numbers will be added when measured.
- References for `heat` and `mimo` are best-known optima from global search, not proofs. `reactor` is exact.
- The three tasks are deliberately cheap so that the benchmark runs in CI. They are stand-ins for real
  lab codes, which plug in through the simulator template.

## Teaching

[`docs/TUTORIAL.md`](docs/TUTORIAL.md) has a two-session lab (budgets and fidelity bias; agents, evaluation
and reproducibility) and five project ideas sized from 2 to 12 weeks.

## License and citation

MIT. See `CITATION.cff`.
