# Tutorial and student projects

A two-session lab for a course on AI for science, or a starting point for a semester project.

## Session 1: budgets, fidelity and bias (90 min)

1. `sciagent sims` and read the three simulator cards. For each task, predict which cheap fidelity is
   *biased* and which is only *noisy*. Check your prediction with
   `pytest tests/test_simulators.py -k "biased or noise" -v`.
2. Run the same reactor study with every planner:
   `for p in random sh bo llm; do sciagent run studies/reactor.yaml --planner $p --out runs/$p; done`
   Compare `runs_per_fidelity` and regret. Why does successive halving spend most runs on `euler_coarse`?
3. Open `sciagent serve` and look at the objective-per-run chart for `sh`. Find a promoted configuration
   whose objective *dropped* when it moved to a higher fidelity. What does that tell you about rung 0?
4. Steer a live run: start a 600-unit study, then `sciagent steer runs/<dir> --bound T=300,330 --note "heater limit"`.
   Find the `steer` and `guard` events in the trace.

## Session 2: agents, evaluation and reproducibility (90 min)

1. Run `sciagent bench --seeds 5` and read `bench_results/results.md`. Is the ranking the same on every
   task? Rerun with `--budget 120`. What changes?
2. Point the LLM planner at a model you can access (`studies/heat_llm_claude.yaml` or a local Ollama
   model with `provider: openai, base_url: http://localhost:11434/v1`). Record the run, then
   `sciagent replay` it. Why is the replay identical even though LLM sampling is not?
3. Read the guard events for the LLM run. How often did the model propose something infeasible?

## Project ideas (2-12 weeks)

- **Your own simulator.** Wrap a code from your group (see `docs/ADDING_A_SIMULATOR.md`) and benchmark
  planners on it. The deliverable is a new task in the benchmark plus a short report.
- **Multi-fidelity BO.** Add a planner with a GP over (parameters, fidelity) and a cost-aware acquisition.
  Does it beat SH and single-fidelity BO on the biased reactor task?
- **Is the LLM using the advisors?** Ablate `advisors: [bo]` against no advisors and measure how often the
  model's proposals coincide with the GP's suggestions.
- **Distillation.** Follow `recipes/distill_planner_lora` and measure how much planning quality a 0.5B model
  retains, and at what latency and cost.
- **Surrogate-in-the-loop.** Use the distilled surrogate as a free fidelity level 0 and study when it helps
  and when its errors mislead the planner.
