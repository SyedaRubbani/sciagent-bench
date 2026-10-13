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
- Follow the scientist's steering notes when present.
- Set "stop": true only when further runs are unlikely to improve the result.

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
                       "remaining_compute_units": _r(state.remaining_compute)},
            "incumbent": None if inc is None else {"i": inc.index, "params": {k: _r(v) for k, v in inc.params.items()},
                                                   "fidelity": inc.fidelity, "objective": _r(inc.objective)},
            "history_shown": len(rows), "history_total": len(state.history),
            "history": rows,
            "steering_notes": state.notes,
            "advisors": self.advice(state),
        }

    def advice(self, state: StudyState) -> dict:
        out: dict[str, Any] = {}
        if "bo" in self.advisors:
            out["gp_expected_improvement"] = self._bo_advice(state)
        if "surrogate" in self.advisors:
            out["learned_surrogate"] = self._surrogate_advice(state)
        return out

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
                runs = [RunRequest(dict(r.params), r.fidelity, r.why) for r in out.runs]
                return Plan(runs, out.reasoning_summary, stop=out.stop, usage=usage)
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                usage["repairs"] = attempt + 1
                errors = [str(e)[:500]]
                messages = messages[:1] + [
                    {"role": "assistant", "content": resp.text[:4000]},
                    {"role": "user", "content": f"Your reply could not be used: {errors[0]}\n"
                                                "Reply again with ONE valid JSON object following the schema."},
                ]
        usage["failed"] = True
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
