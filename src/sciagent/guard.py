"""Deterministic guard between any planner (LLM or not) and the compute.

Planners propose; the guard disposes. It enforces parameter bounds, valid fidelities, batch size and the
remaining budget, and reports every intervention so it shows up in the trace.
"""

from __future__ import annotations

from .cache import run_key
from .state import Plan, RunRequest, StudyState


def check_plan(plan: Plan, state: StudyState, reserve: bool = False) -> tuple[list[RunRequest], list[dict]]:
    """Validate a proposal. With ``reserve=True`` each accepted run holds a ledger reservation
    (``RunRequest.token``) that the orchestrator commits after execution or releases on failure."""
    sim = state.sim
    accepted: list[RunRequest] = []
    issues: list[dict] = []
    remaining = state.remaining_compute
    seen_keys = {run_key(sim.name, sim.version, r.params, r.fidelity, None) for r in state.history if r.ok} \
        if not sim.stochastic else set()
    fid_names = {f.name for f in sim.fidelities}
    for j, req in enumerate(plan.runs):
        if len(accepted) >= state.spec.batch_size:
            issues.append({"run": j, "action": "dropped", "reason": "exceeds batch_size"})
            continue
        if req.fidelity not in fid_names:
            issues.append({"run": j, "action": "dropped", "reason": f"unknown fidelity {req.fidelity!r}"})
            continue
        try:
            params = sim.normalize(req.params)
        except ValueError as e:
            issues.append({"run": j, "action": "dropped", "reason": str(e)})
            continue
        clipped = {}
        for p in sim.params:
            lo, hi = state.bounds[p.name]
            v = params[p.name]
            v2 = min(max(v, lo), hi)
            if p.kind == "int":
                v2 = int(round(v2))
            if v2 != v:
                clipped[p.name] = (v, v2)
            params[p.name] = v2
        if clipped:
            issues.append({"run": j, "action": "clipped", "reason": "outside effective bounds", "detail": clipped})
        cost = sim.fidelity(req.fidelity).cost
        key = run_key(sim.name, sim.version, params, req.fidelity, None)
        if key in seen_keys:
            issues.append({"run": j, "action": "dropped", "reason": "duplicate of an existing deterministic run"})
            continue
        if cost > remaining + 1e-9:
            issues.append({"run": j, "action": "dropped", "reason": f"cost {cost} exceeds remaining budget {remaining:.4g}"})
            continue
        remaining -= cost
        seen_keys.add(key)
        acc = RunRequest(params, req.fidelity, req.why)
        if reserve:
            acc.token = state.ledger.reserve(cost, ref=f"batch{state.batch}:run{j}")
        accepted.append(acc)
    return accepted, issues
