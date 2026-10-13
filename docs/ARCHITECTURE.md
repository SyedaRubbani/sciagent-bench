# Architecture

```mermaid
flowchart LR
    R[Researcher<br/>study.yaml, steering] --> O
    subgraph O[Orchestrator loop]
      direction TB
      P[Planner<br/>random / SH / BO / LLM] --> G[Guard<br/>bounds, fidelity, budget, duplicates]
      G --> X[Executor<br/>inline / process pool / Slurm]
      X --> C[(Content-addressed<br/>result cache)]
      X --> S[State + incumbent]
      S --> P
    end
    A[Advisors<br/>GP-EI, distilled surrogate] --> P
    L[LLM providers<br/>Anthropic, OpenAI-compatible: vLLM, Ollama] <--> K[(Cassette<br/>record / replay)]
    K <--> P
    O --> T[(trace.jsonl, runs.jsonl,<br/>manifest, summary)]
    T --> V[FastAPI + viewer]
    T --> M[MCP server]
    V -- steer --> R
```

## Design decisions

**Planners propose, the guard disposes.** Every planner, including an LLM, only returns a *proposal*. A
deterministic guard enforces parameter bounds (including bounds the researcher narrowed mid-run), valid
fidelities, batch size, the remaining compute budget, and de-duplication of deterministic runs. Every
intervention is a `guard` event in the trace, so you can see how often an LLM tried to overspend.

**Declared costs, measured times.** Budgets use each fidelity's declared cost so that a campaign is
reproducible across machines. Wall-clock is still measured and recorded on every run.

**The incumbent is the best result at the highest fidelity evaluated.** This keeps planners honest: a
cheap, biased fidelity cannot produce the reported answer unless nothing better was checked (see the reactor
task, where explicit Euler over-estimates the yield near the optimum).

**Reproducibility is tested, not claimed.** Each run writes its spec, a manifest (package versions, git
SHA, spec digest), and a `results_digest` over what was run and what came out. Seeds are derived from
`(study seed, run index)`. LLM calls are recorded to a cassette keyed by a hash of the exact request;
`sciagent replay <run>` re-executes a campaign from the cassette and fails if the digest differs. The test
suite does this for LLM and non-LLM planners.

**Inspectable by people and by agents.** The same trace feeds the CLI timeline (`sciagent trace`), the web
viewer (`sciagent serve`) and MCP tools (`study_trace`), so a researcher or another agent can audit and
steer a run.

**Graceful degradation.** Malformed LLM output gets a bounded repair loop; repeated failure ends the
campaign with a recorded reason instead of a crash; an exhausted LLM budget can fall back to BO.

## Event kinds in `trace.jsonl`

| kind | emitted when | key fields |
|---|---|---|
| `study.start` / `study.end` | campaign boundaries | budget, spec digest / summary |
| `batch.start` / `batch.end` | each planning round | spent, incumbent |
| `llm.call` | each model call (incl. repairs) | model, tokens, attempt, replayed, response |
| `plan` | a proposal | planner, rationale, proposed runs, usage |
| `guard` | the guard changed a proposal | per-run action and reason |
| `run` | a simulation finished | params, fidelity, seed, objective, metrics, cost, cached |
| `execute` | a batch was dispatched | executor, jobs, cache hits, seconds |
| `steer` / `pause` | researcher intervention | what was applied |
| `error` | the orchestrator itself failed | exception |
