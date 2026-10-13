# SciAgent-Bench

**A reproducible benchmark and orchestration system for AI agents that plan scientific simulation campaigns
under a budget.**

A researcher describes a study in a short YAML file: what can be varied, what counts as better, which simulator
fidelities exist and what each costs, and the budget. A planning agent decides which simulations to run next, and at
which fidelity. It adapts after every batch, cannot overspend, and can be paused, redirected or resumed after a
crash. The same harness benchmarks LLM planners (closed APIs or open models on your GPU) against classical
baselines on scientific tasks whose optima are verified, and reports the results with proper statistics.

[Technical report](docs/TECHNICAL_REPORT.md) · [Architecture](docs/ARCHITECTURE.md) ·
[GPU experiments on Kaggle/Colab](docs/GPU_EXPERIMENTS.md) · [Add your simulator](docs/ADDING_A_SIMULATOR.md) ·
[Tutorial](docs/TUTORIAL.md) · [Changelog](CHANGELOG.md)

```bash
sciagent run studies/reactor.yaml                     # a budgeted campaign (successive halving by default)
sciagent run studies/heat_llm_claude.yaml             # an LLM planner, every call recorded
sciagent replay runs/<dir>                            # re-execute and verify bit-identical results
sciagent resume runs/<dir>                            # continue an interrupted study in place
sciagent steer runs/<dir> --bound T=300,330 --planner bo   # narrow the space and redirect, live
sciagent serve                                        # API + live viewer (token auth optional)
sciagent bench --seeds 20 --parallel 4                # planners x tasks x seeds
sciagent analyze bench_results                        # bootstrap CIs, paired tests, win rates, AUC
sciagent verify-references --device cuda              # reference optima with uncertainty
sciagent experiments --profile kaggle-t4              # the full resumable GPU pipeline
sciagent report results/cpu results/gpu               # technical report, generated from result files
```

## The four phases

| Phase | What it delivers | Where |
|---|---|---|
| **1. Scientific benchmark foundation** | Three simulators with biased or noisy cheap fidelities; YAML study specs with a JSON Schema; a reserve/commit budget ledger; four baseline planners; references verified with uncertainty; convergence-order, closed-form and property-based tests | `simulators/`, `spec.py`, `budget.py`, `planners/classic.py`, `verify.py`, `tests/` |
| **2. Adaptive agent and inspectability** | LLM planner with tool advisors and a repair loop; replanning after every batch; pause, resume and redirect; crash-safe resume; parent-linked traces; deterministic replay of LLM campaigns; FastAPI service and viewer with token auth. The guard and ledger enforce budgets without trusting the LLM | `planners/llm_planner.py`, `orchestrator.py`, `guard.py`, `tracing.py`, `service.py` |
| **3. Evaluation and model optimisation** | Multi-seed comparisons with statistics; curated surrogate datasets; teacher-student distillation and int8 quantisation; GPU kernels; LLM-planner distillation with LoRA; 4-bit serving; vLLM with a transformers fallback; Kaggle and Colab notebooks | `bench.py`, `analysis.py`, `surrogate/`, `gpu/`, `serving/`, `experiments.py`, `notebooks/` |
| **4. Researcher-ready release** | Slurm executor; MCP server; Docker; CI with coverage and a pipeline smoke run; simulator template; tutorial; generated technical report | `executors.py`, `mcp_server.py`, `Dockerfile`, `.github/`, `templates/`, `docs/` |

## Tasks

| Task | Domain | Parameters | Cheap fidelities are | Reference |
|---|---|---|---|---|
| `reactor` | chemical engineering: consecutive reactions A → B → C | temperature, residence time, catalyst | **biased** (explicit Euler over-estimates the yield near the optimum and diverges for hot, long runs) | exact closed form |
| `heat` | physics: steady 2-D conduction, place a cooling pin | pin x, y, radius | **biased** (coarse grids smear sources and pin) | best known; independent global searches agree to 4e-10 |
| `mimo` | information theory: uplink energy efficiency, 8 receive antennas | transmit power, active RF chains (integer) | **noisy but unbiased** (few Monte-Carlo draws) | best known; high-precision value with standard error |

## Results

RESULTS_SECTION

## Quick start

```bash
git clone https://github.com/SyedaRubbani/sciagent-bench && cd sciagent-bench
pip install -e ".[dev,surrogate,mcp]"
pytest -q
sciagent run studies/reactor.yaml --out runs/demo && sciagent trace runs/demo | head -30
sciagent serve --runs-dir runs              # http://127.0.0.1:8000
```

Using a model:

```bash
export ANTHROPIC_API_KEY=...                # or an open model: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001
sciagent run studies/heat_llm_claude.yaml   # records a cassette; `sciagent replay` re-runs it without the API
```

Reproducing everything:

```bash
sciagent experiments --profile cpu-full --results results/cpu     # CPU parts, about 40 minutes
# GPU parts: notebooks/kaggle_gpu_experiments.ipynb on Kaggle (2 x T4), see docs/GPU_EXPERIMENTS.md
sciagent report results/cpu results/gpu --out docs/TECHNICAL_REPORT.md
```

## A study spec

```yaml
schema_version: 1
name: reactor-yield
simulator: reactor                    # or "my_lab.my_sim:MySimulator"
description: >
  Maximise net yield of B. Catalyst is expensive; we would rather not run hotter than ~380 K.
budget: {compute_units: 300, llm_usd: 1.0, max_batches: 40}
batch_size: 4
bounds: {T: [300, 380]}               # enforced by the guard; can be narrowed live
planner:
  kind: llm
  advisors: [bo, surrogate]           # tool outputs the agent sees each batch
  options: {fallback: bo, surrogate_path: results/cpu/surrogate/models}
  llm: {provider: openai, base_url: "http://localhost:8001/v1", model: Qwen/Qwen2.5-7B-Instruct, mode: record}
executor: {kind: slurm, slurm: {slurm_partition: normal.4h, timeout_min: 10}}
```

`sciagent schema > study.schema.json` gives editors validation and completion.

## Guarantees and how they are tested

| Guarantee | Test |
|---|---|
| No proposal can exceed the compute budget or the (live-narrowed) bounds | Hypothesis property tests on random proposals; a "greedy LLM" that requests 1,000 top-fidelity runs |
| The LLM USD cap is never exceeded | worst-case pre-call check; capped-run test |
| Campaigns are reproducible | same seed gives the same results digest; `replay` of LLM campaigns from cassettes |
| Interrupted studies resume exactly | simulated preemption for each planner type, compared with an uninterrupted run |
| Parallel benchmarking does not change results | digests compared between `--parallel 1` and `4` |
| Solvers are what they claim | convergence orders (Euler 1, RK4 4), SISO Rayleigh closed form, PCG vs sparse direct |
| The LLM path works over real HTTP | fake OpenAI-compatible server with injected 503s |
| Batched serving does not change outputs | greedy batched == single generation in the transformers server |

## Status and limitations

- Measured here (CPU): all tests, reference verification, kernel timings on CPU, the classical-planner benchmark with
  statistics, the surrogate pipeline, and the full experiment pipeline in `cpu-smoke` mode (fake teacher, tiny
  student). Results are in `results/cpu` and the [technical report](docs/TECHNICAL_REPORT.md).
- Not yet measured: GPU timings, open-model LLM planner results, the distilled planner and 4-bit serving. The Kaggle
  notebook produces them; the report lists them as "not run" until then.
- The tasks are cheap stand-ins chosen so that the benchmark and its statistics run in CI. Real lab codes plug in
  through the simulator template, and conclusions about planners may not transfer to them.
- Regret is in each task's own units; compare within a task, not across tasks.

## License and citation

MIT. See `CITATION.cff`.
