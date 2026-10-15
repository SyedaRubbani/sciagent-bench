"""LLM planning agent.

Each batch the agent receives a compact, deterministic description of the study -- the researcher's goal,
the simulator card, effective bounds, the fidelity/cost table, budget status, results so far, researcher
steering notes and (optionally) suggestions from tool "advisors" such as a GP or a learned surrogate --
and must answer with a JSON plan. Malformed answers trigger a bounded repair loop; anything that would
break bounds or budget is later corrected by the deterministic guard, and every intervention is traced.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from ..llm.providers import Provider, make_provider
from ..spec import StudySpec
from ..state import Plan, RunRequest, StudyState
from .base import Planner
from .classic import BayesOptPlanner, gp_predict, suggest_ei

SYSTEM_PROMPT = """You are a research-computing planning agent working with a scientist.
You decide which simulations to run next, and at which fidelity, to find the best design within a fixed budget.

Rules:
- Propose at most `batch_size` runs. Their total cost must not exceed the remaining compute budget.
- Use only the listed parameters, inside the listed bounds, and only the listed fidelity names.
- Cheap fidelities are for exploration; read each fidelity's notes (some are biased, some only noisy).
- The result reported to the scientist is the best run at the HIGHEST fidelity evaluated, so reserve budget
  to confirm promising candidates at the top fidelity.
- Do not repeat a deterministic run that already exists (same parameters and fidelity); it adds nothing.
  The history lists every run so far: propose NEW parameter values, informed by which regions scored best.
- If the simulator is stochastic, repeating one point only averages noise. Explore different parameter values
  instead of re-running the same design.
- `previous_plan_feedback` says which of your last proposals the safety guard dropped or clipped, and why.
  Fix those problems in the next plan.
- Follow the scientist's steering notes when present.
- Spend the budget. Set "stop": true only when the remaining budget cannot improve the result, and never
  before most of it is used. Each batch must contain at least one run while budget remains.

Reply with ONE JSON object and nothing else:
{"reasoning_summary": "<= 60 words on your strategy", "stop": false,
 "runs": [{"params": {"<name>": <number>, ...}, "fidelity": "<fidelity name>", "why": "<short reason>"}]}"""


class _Run(BaseModel):
    params: dict[str, float | int]
    fidelity: str
    why: str = ""


class _PlanOut(BaseModel):
    reasoning_summary: str = ""
    stop: bool = False
    runs: list[_Run] = Field(default_factory=list)


def _r(x: Any, sig: int = 6):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return float(f"{x:.{sig}g}") if np.isfinite(x) else None
    return x


def extract_json(text: str) -> dict:
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    blob = fence.group(1) if fence else None
    if blob is None:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object found")
        blob = text[start: end + 1]
    return json.loads(blob)


class LLMPlanner(Planner):
    name = "llm"

    def __init__(self, spec: StudySpec, provider: Provider | None = None):
        super().__init__(spec.planner.options, spec.seed)
        self.spec = spec
        self.llm = spec.planner.llm
        self.provider = provider or make_provider(self.llm, **({"fail_first": self.options["mock_fail_first"]}
                                                                if "mock_fail_first" in self.options else {}))
        self.advisors = list(spec.planner.advisors)
        self.fallback = BayesOptPlanner({}, spec.seed) if self.options.get("fallback") == "bo" else None
        self._surrogate = None
        self._last_runs: list[RunRequest] = []
        self._last_feedback: dict | None = None

    def feedback(self, issues: list[dict], n_accepted: int) -> None:
        """Remember what the guard did to the last proposal so the next prompt can say so."""
        items = []
        for it in issues[:8]:
            j = it.get("run")
            run = self._last_runs[j] if isinstance(j, int) and 0 <= j < len(self._last_runs) else None
            items.append({"proposed": None if run is None else {"params": {k: _r(v) for k, v in run.params.items()},
                                                               "fidelity": run.fidelity},
                          "action": it.get("action"), "reason": it.get("reason")})
        self._last_feedback = {"accepted": n_accepted, "proposed": len(self._last_runs), "problems": items}

    def _too_early_to_stop(self, state: StudyState) -> bool:
        frac = float(self.options.get("min_stop_fraction", 0.0))
        if frac <= 0 or state.compute_budget <= 0:
            return False
        return (state.spent_compute / state.compute_budget < frac
                and state.remaining_compute >= state.cheapest_cost() - 1e-9)

    # ------------------------------------------------------------------ prompt
    def state_json(self, state: StudyState) -> dict:
        sim = state.sim
        ok = state.history
        max_hist = int(self.options.get("max_history", 60))
        rows = [
            {"i": r.index, "params": {k: _r(v) for k, v in r.params.items()}, "fidelity": r.fidelity,
             "fidelity_level": r.fidelity_level, "objective": _r(r.objective) if r.ok else None,
             "metrics": {k: _r(v, 4) for k, v in list(r.metrics.items())[:4]}, "ok": r.ok}
            for r in ok
        ]
        if len(rows) > max_hist:
            by_f: dict[int, list] = {}
            for row in rows:
                by_f.setdefault(row["fidelity_level"], []).append(row)
            keep = set()
            for lst in by_f.values():
                for row in sorted([x for x in lst if x["objective"] is not None], key=lambda x: -x["objective"])[:15]:
                    keep.add(row["i"])
            keep |= {row["i"] for row in rows[-2 * self.spec.batch_size:]}
            rows = [row for row in rows if row["i"] in keep]
        inc = state.incumbent()
        return {
            "study": self.spec.name,
            "goal": self.spec.description,
            "simulator": {"name": sim.name, "description": sim.description,
                          "objective": f"maximize {sim.objective_name}", "deterministic": not sim.stochastic},
            "params": [{"name": p.name, "low": _r(p.low), "high": _r(p.high), "kind": p.kind, "unit": p.unit,
                        "description": p.description} for p in state.effective_params()],
            "fidelities": [{"name": f.name, "level": f.level, "cost": f.cost, "notes": f.notes} for f in sim.fidelities],
            "batch_size": self.spec.batch_size,
            "batch": state.batch,
            "seed": self.spec.seed,
            "budget": {"compute_units": _r(state.compute_budget), "spent": _r(state.spent_compute),
                       "remaining_compute_units": _r(state.remaining_compute),
                       "affordable_fidelities": [f.name for f in sim.fidelities
                                                 if f.cost <= state.remaining_compute + 1e-9]},
            "incumbent": None if inc is None else {"i": inc.index, "params": {k: _r(v) for k, v in inc.params.items()},
                                                   "fidelity": inc.fidelity, "objective": _r(inc.objective)},
            "history_shown": len(rows), "history_total": len(state.history),
            "history": rows,
            "steering_notes": state.notes,
            "previous_plan_feedback": self._last_feedback,
            "advisors": self.advice(state),
        }

    def advice(self, state: StudyState) -> dict:
        out: dict[str, Any] = {}
        if "bo" in self.advisors:
            out["gp_expected_improvement"] = self._bo_advice(state)
        if "surrogate" in self.advisors:
            out["learned_surrogate"] = self._surrogate_advice(state)
        return out

    def _ei_points(self, state: StudyState, k: int) -> list[dict]:
        """Up to ``k`` GP expected-improvement points, or [] when there is too little data for a GP."""
        ok = state.ok_runs()
        if not ok:
            return []
        d = len(state.sim.params)
        lvl = max(r.fidelity_level for r in ok)
        data = [r for r in ok if r.fidelity_level == lvl]
        if len(data) < d + 2:
            lvl = min(r.fidelity_level for r in ok)
            data = [r for r in ok if r.fidelity_level == lvl]
            if len(data) < d + 2:
                return []
        X = np.array([self.to_unit(state, r.params) for r in data])
        y = np.array([r.objective for r in data])
        rng = np.random.default_rng(state.batch + 29)
        return [self.from_unit(state, u) for u, _ in suggest_ei(X, y, k, rng, d, n_cand=1500)]

    def _guard_repair(self, plan: Plan, state: StudyState) -> Plan:
        """If the guard would reject every run of ``plan``, keep the model's fidelity choice when it is affordable and
        replace the rejected parameter values with GP-EI points. Added runs are labelled and counted in the trace."""
        from ..guard import check_plan

        if plan.stop or state.remaining_compute < state.cheapest_cost() - 1e-9:
            return plan
        accepted, issues = check_plan(plan, state)
        if accepted:
            return plan
        afford = sorted([f for f in state.sim.fidelities if f.cost <= state.remaining_compute + 1e-9],
                        key=lambda f: f.level)
        if not afford:
            return plan
        names = {f.name for f in afford}
        fid = next((r.fidelity for r in plan.runs if r.fidelity in names), afford[-1].name)
        k = self.spec.batch_size
        pts = self._ei_points(state, k)
        while len(pts) < k:
            pts.append(self.sample_point(state))
        reasons = sorted({str(i.get("reason", ""))[:40] for i in issues})[:2]
        why = "guard-repair (model's proposals were rejected: " + "; ".join(reasons) + ")"
        plan.runs = list(plan.runs) + [RunRequest(p, fid, why) for p in pts]
        plan.usage["repaired"] = len(pts)
        plan.rationale = (plan.rationale + f" [guard-repair: {len(pts)} runs added at {fid}]").strip()
        return plan

    def _bo_advice(self, state: StudyState):
        ok = state.ok_runs()
        if not ok:
            return "no data yet"
        lvl = max(r.fidelity_level for r in ok)
        data = [r for r in ok if r.fidelity_level == lvl]
        d = len(state.sim.params)
        if len(data) < d + 2:
            lvl = min(r.fidelity_level for r in ok)
            data = [r for r in ok if r.fidelity_level == lvl]
            if len(data) < d + 2:
                return "too little data for a GP"
        X = np.array([self.to_unit(state, r.params) for r in data])
        y = np.array([r.objective for r in data])
        rng = np.random.default_rng(state.batch + 17)
        picks = suggest_ei(X, y, 3, rng, d, n_cand=1500)
        pts = np.array([u for u, _ in picks])
        m, s = gp_predict(X, y, pts)
        fname = state.sim.fidelities[lvl].name
        return [{"params": {k: _r(v, 5) for k, v in self.from_unit(state, u).items()}, "gp_mean": _r(mm, 4),
                 "gp_std": _r(ss, 3), "fitted_on": fname} for u, mm, ss in zip(pts, m, s)]

    def _surrogate_advice(self, state: StudyState):
        from ..surrogate.advisor import SurrogateAdvisor

        if self._surrogate is None:
            path = self.options.get("surrogate_path")
            if not path:
                return "surrogate advisor enabled but options.surrogate_path not set"
            self._surrogate = SurrogateAdvisor(path, state.sim)
        return self._surrogate.advise(state, seed=state.batch)

    def build_messages(self, state: StudyState) -> list[dict]:
        sj = json.dumps(self.state_json(state), sort_keys=True, separators=(",", ":"))
        user = ("Plan the next batch. The full study state follows.\n\nSTATE_JSON\n```json\n" + sj + "\n```\n"
                "Answer with the JSON plan only.")
        return [{"role": "user", "content": user}]

    # ------------------------------------------------------------------ propose
    def propose(self, state: StudyState) -> Plan:
        if self.spec.budget.llm_usd is not None and state.spent_usd >= self.spec.budget.llm_usd:
            if self.fallback:
                plan = self.fallback.propose(state)
                plan.rationale = "LLM budget exhausted -> BO fallback: " + plan.rationale
                return plan
            return Plan([], "LLM budget exhausted", stop=True)
        messages = self.build_messages(state)
        from ..budget import llm_call_affordable

        ok, worst = llm_call_affordable(state.spent_usd, self.spec.budget.llm_usd,
                                        len(SYSTEM_PROMPT) + sum(len(m["content"]) for m in messages),
                                        self.llm.max_tokens, self.llm.price_in_per_mtok, self.llm.price_out_per_mtok)
        if not ok:
            if self.fallback:
                plan = self.fallback.propose(state)
                plan.rationale = f"LLM budget would be exceeded (worst case ${worst:.4f}) -> BO fallback: " + plan.rationale
                return plan
            return Plan([], f"LLM budget would be exceeded by the next call (worst case ${worst:.4f})", stop=True)
        usage = {"tokens_in": 0, "tokens_out": 0, "usd": 0.0, "latency_s": 0.0, "repairs": 0, "calls": [],
                 "model": self.provider.model}
        errors: list[str] = []
        for attempt in range(self.llm.max_repairs + 1):
            resp = self.provider.complete(SYSTEM_PROMPT, messages, self.llm.temperature, self.llm.max_tokens)
            usd = (resp.input_tokens * self.llm.price_in_per_mtok + resp.output_tokens * self.llm.price_out_per_mtok) / 1e6
            usage["tokens_in"] += resp.input_tokens
            usage["tokens_out"] += resp.output_tokens
            usage["usd"] += usd
            usage["latency_s"] += resp.latency_s
            usage["calls"].append({"attempt": attempt, "tokens_in": resp.input_tokens, "tokens_out": resp.output_tokens,
                                   "replayed": resp.cached, "response": resp.text[:4000]})
            try:
                out = _PlanOut.model_validate(extract_json(resp.text))
                errors = self._semantic_errors(out, state)
                if errors:
                    raise ValueError("; ".join(errors))
                stop = out.stop
                if stop and self._too_early_to_stop(state):
                    if not out.runs:
                        raise ValueError("you may not stop yet: most of the budget is unspent. "
                                         "Propose new runs (new parameter values) in `runs`")
                    stop = False  # runs were proposed: execute them and keep going
                runs = [RunRequest(dict(r.params), r.fidelity, r.why) for r in out.runs]
                plan = Plan(runs, out.reasoning_summary, stop=stop, usage=usage)
                if self.options.get("guard_repair"):
                    plan = self._guard_repair(plan, state)
                self._last_runs = plan.runs
                return plan
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                usage["repairs"] = attempt + 1
                errors = [str(e)[:500]]
                messages = messages[:1] + [
                    {"role": "assistant", "content": resp.text[:4000]},
                    {"role": "user", "content": f"Your reply could not be used: {errors[0]}\n"
                                                "Reply again with ONE valid JSON object following the schema."},
                ]
        usage["failed"] = True
        self._last_runs = []
        self._last_feedback = {"accepted": 0, "proposed": 0,
                               "problems": [{"proposed": None, "action": "rejected",
                                             "reason": f"your last reply was unusable: {errors[0] if errors else 'invalid'}"}]}
        return Plan([], f"LLM failed to produce a valid plan after {self.llm.max_repairs} repairs: {errors}",
                    stop=False, usage=usage)

    @staticmethod
    def _semantic_errors(out: _PlanOut, state: StudyState) -> list[str]:
        names = {p.name for p in state.sim.params}
        fids = {f.name for f in state.sim.fidelities}
        errs = []
        for j, r in enumerate(out.runs):
            if r.fidelity not in fids:
                errs.append(f"run {j}: unknown fidelity {r.fidelity!r} (use one of {sorted(fids)})")
            if set(r.params) != names:
                errs.append(f"run {j}: params must be exactly {sorted(names)}, got {sorted(r.params)}")
        return errs
