# Adding your lab's simulator

SciAgent treats a simulator as a black box with **parameters**, ordered **fidelity levels** with declared
costs, and a deterministic `run(params, fidelity, seed)`. Anything you can call from Python fits: a function,
a compiled solver behind `subprocess`, a container, or a job submitted to a cluster.

1. Copy [`templates/simulator_template.py`](../templates/simulator_template.py) into your own package.
2. Fill in `params` (with units), `fidelities` (cheapest first, cost in a unit your lab budgets in, such as
   core-minutes or GPU-hours) and `_evaluate`.
3. Write the `description` and each fidelity's `notes` for a reader who has never seen your code. The LLM
   planner reads them verbatim, and so do people reading traces later. Say what each fidelity gets wrong.
4. Reference it from a study without touching SciAgent:

```yaml
name: my-first-campaign
simulator: "my_lab.my_sim:MySimulator"
budget: {compute_units: 200}
planner: {kind: sh}
```

5. Run `sciagent run my_study.yaml`, then `sciagent trace runs/<dir>` and `sciagent serve` to inspect it.

## Choosing fidelity costs

Costs are *declared*, not measured, so that budgets and results are reproducible across machines. Measured
wall-clock is still recorded on every run. A good starting point is the median runtime of each level on
your cluster, in core-minutes.

## Checklist before a real campaign

- [ ] `version` bumped whenever the code's results change (the cache key includes it)
- [ ] failures raise exceptions (they are traced, charged and survived; silent NaNs are rejected)
- [ ] `stochastic = True` if the seed matters, so repeated runs are not de-duplicated
- [ ] the cheapest fidelity was checked against the most expensive on a handful of points
- [ ] a small `sciagent run --planner random --budget 20` dry run finished cleanly
