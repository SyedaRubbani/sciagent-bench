"""The campaign loop: plan -> guard -> execute (with cache) -> observe -> steer -> repeat."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import yaml

from .cache import RunCache, run_key
from .executors import Executor, make_executor
from .guard import check_plan
from .planners import Planner, make_planner
from .simulators import get_simulator, references
from .spec import StudySpec
from .state import RunRecord, StudyState
from .tracing import Tracer, environment_info, results_digest


def derive_seed(study_seed: int, index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{study_seed}:{index}".encode()).digest()[:4], "little") & 0x7FFFFFFF


class Orchestrator:
    def __init__(self, spec: StudySpec, run_dir: str | Path | None = None, *, planner: Planner | None = None,
                 provider=None, executor: Executor | None = None, score: bool = True, verbose: bool = False):
        self.spec = spec
        self.run_dir = Path(run_dir) if run_dir else None
        self.sim = get_simulator(spec.simulator)
        for name in spec.bounds:
            self.sim.param(name)  # fail early on typos
        if spec.planner.kind == "llm" and spec.planner.llm.mode == "record" and not spec.planner.llm.cassette \
                and self.run_dir:
            spec.planner.llm.cassette = str(self.run_dir / "llm_cassette.jsonl")
        self.state = StudyState(spec, self.sim)
        self.planner = planner or make_planner(spec, provider=provider)
        self.executor = executor or make_executor(spec.executor, self.run_dir)
        cache_dir = spec.cache_dir or (self.run_dir / "cache" if self.run_dir else None)
        self.cache = RunCache(cache_dir)
        self.tracer = Tracer(self.run_dir)
        self.score = score
        self.verbose = verbose
        self._steer_version = 0
        self.status = "created"
        self.incumbent_curve: list[dict] = []

    # ------------------------------------------------------------------ steering
    def _control_path(self) -> Path | None:
        return self.run_dir / "control.json" if self.run_dir else None

    def _read_control(self) -> dict:
        p = self._control_path()
        if p and p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _apply_steering(self, root: str) -> bool:
        """Apply new researcher instructions; block while paused. Returns False if asked to stop."""
        while True:
            ctl = self._read_control()
            if ctl.get("version", 0) > self._steer_version:
                self._steer_version = ctl["version"]
                applied = {}
                if ctl.get("note"):
                    self.state.notes.append(str(ctl["note"]))
                    applied["note"] = ctl["note"]
                for name, (lo, hi) in (ctl.get("bounds") or {}).items():
                    p = self.sim.param(name)
                    lo, hi = max(float(lo), p.low), min(float(hi), p.high)
                    if lo <= hi:
                        self.state.bounds[name] = (lo, hi)
                        applied.setdefault("bounds", {})[name] = (lo, hi)
                if ctl.get("compute_units") is not None:
                    self.state.compute_budget = max(float(ctl["compute_units"]), self.state.spent_compute)
                    applied["compute_units"] = self.state.compute_budget
                self.tracer.event("steer", {"version": self._steer_version, "applied": applied}, parent=root)
            if ctl.get("stop"):
                self.state.stop_reason = "stopped by researcher"
                return False
            if not ctl.get("pause"):
                return True
            if self.status != "paused":
                self.status = "paused"
                self._write_status()
                self.tracer.event("pause", {}, parent=root)
            time.sleep(0.3)

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        spec, st, sim = self.spec, self.state, self.sim
        if self.run_dir:
            (self.run_dir / "spec.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
            manifest = {"study": spec.name, "spec_digest": spec.digest(), "simulator": sim.card(),
                        "environment": environment_info(), "started": time.time()}
            (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        root = self.tracer.event("study.start", {"name": spec.name, "simulator": sim.name, "planner": spec.planner.kind,
                                                 "budget": spec.budget.model_dump(), "spec_digest": spec.digest()})
        self.status = "running"
        empty_plans = 0
        t_start = time.perf_counter()
        try:
            while True:
                if not self._apply_steering(root):
                    break
                self.status = "running"
                if st.batch >= spec.budget.max_batches:
                    st.stop_reason = "max_batches reached"
                    break
                if st.remaining_compute < st.cheapest_cost() - 1e-9:
                    st.stop_reason = "compute budget exhausted"
                    break
                if spec.budget.wall_clock_s and time.perf_counter() - t_start > spec.budget.wall_clock_s:
                    st.stop_reason = "wall-clock budget exhausted"
                    break

                with self.tracer.span("batch", {"batch": st.batch}, parent=root) as (bid, bout):
                    t0 = time.perf_counter()
                    plan = self.planner.propose(st)
                    plan_s = time.perf_counter() - t0
                    u = plan.usage
                    if u:
                        st.tokens_in += u.get("tokens_in", 0)
                        st.tokens_out += u.get("tokens_out", 0)
                        st.spent_usd += u.get("usd", 0.0)
                        for call in u.get("calls", []):
                            self.tracer.event("llm.call", {"model": u.get("model"), **call}, parent=bid)
                    self.tracer.event("plan", {"planner": self.planner.name, "rationale": plan.rationale,
                                               "stop": plan.stop, "n_proposed": len(plan.runs), "plan_s": plan_s,
                                               "runs": [r.__dict__ for r in plan.runs],
                                               "usage": {k: v for k, v in u.items() if k != "calls"}}, parent=bid)
                    accepted, issues = check_plan(plan, st)
                    if issues:
                        self.tracer.event("guard", {"issues": issues, "accepted": len(accepted)}, parent=bid)
                    if not accepted:
                        empty_plans += 1
                        bout["accepted"] = 0
                        if plan.stop:
                            st.stop_reason = f"planner stopped: {plan.rationale}"
                            break
                        if empty_plans >= 3:
                            st.stop_reason = "planner produced no runnable work 3 times"
                            break
                        st.batch += 1
                        continue
                    empty_plans = 0
                    records = self._execute(accepted, bid)
                    self.planner.observe(records)
                    st.batch += 1
                    inc = st.incumbent()
                    self.incumbent_curve.append({"batch": st.batch, "spent": st.spent_compute,
                                                 "incumbent": None if inc is None else inc.index})
                    bout.update({"accepted": len(accepted), "spent": st.spent_compute,
                                 "incumbent_objective": None if inc is None else inc.objective})
                    if self.verbose:
                        print(f"[batch {st.batch:3d}] spent {st.spent_compute:7.1f}/{st.compute_budget:g}  "
                              f"incumbent {inc.objective if inc else float('nan'):.5g} @ {inc.fidelity if inc else '-'}"
                              f"  | {plan.rationale[:70]}")
                    self._write_status()
                    if plan.stop:
                        st.stop_reason = f"planner stopped: {plan.rationale}"
                        break
        except Exception as e:
            self.status = "failed"
            self.tracer.event("error", {"error": f"{type(e).__name__}: {e}"}, parent=root)
            self._write_status()
            raise
        finally:
            self.executor.close()
        self.status = "finished"
        summary = self.summary()
        self.tracer.event("study.end", {k: v for k, v in summary.items() if k != "curve"}, parent=root)
        self.tracer.close()
        if self.run_dir:
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        self._write_status()
        return summary

    def _execute(self, accepted, parent) -> list[RunRecord]:
        st, sim = self.state, self.sim
        jobs, slots, records = [], [], []
        for req in accepted:
            idx = len(st.history) + len(records) + len(jobs)
            seed = derive_seed(self.spec.seed, idx)
            key = run_key(sim.name, sim.version, req.params, req.fidelity, seed if sim.stochastic else None)
            hit = self.cache.get(key)
            if hit is not None:
                records.append((idx, req, seed, hit, True))
            else:
                jobs.append((req.params, req.fidelity, seed))
                slots.append((idx, req, seed, key))
        t0 = time.perf_counter()
        results = self.executor.map(sim.name, jobs)
        exec_s = time.perf_counter() - t0
        for (idx, req, seed, key), res in zip(slots, results):
            if res["ok"]:
                self.cache.put(key, res)
            records.append((idx, req, seed, res, False))
        records.sort(key=lambda t: t[0])
        out = []
        for idx, req, seed, res, cached in records:
            fid = sim.fidelity(req.fidelity)
            rec = RunRecord(index=idx, batch=st.batch, params=req.params, fidelity=fid.name, fidelity_level=fid.level,
                            seed=seed, objective=res["objective"] if res["ok"] else None, metrics=res["metrics"],
                            cost=0.0 if cached else fid.cost, wall_s=res["wall_s"], ok=res["ok"], error=res["error"],
                            cached=cached, why=req.why)
            st.history.append(rec)
            st.spent_compute += rec.cost
            out.append(rec)
            self.tracer.event("run", rec.to_dict(), parent=parent)
            if self.run_dir:
                with open(self.run_dir / "runs.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec.to_dict()) + "\n")
        self.tracer.event("execute", {"n_jobs": len(jobs), "n_cached": len(records) - len(jobs), "exec_s": exec_s,
                                      "executor": type(self.executor).__name__}, parent=parent)
        return out

    # ------------------------------------------------------------------ reporting
    def _write_status(self):
        if not self.run_dir:
            return
        st = self.state
        inc = st.incumbent()
        status = {"status": self.status, "batch": st.batch, "spent_compute": st.spent_compute,
                  "compute_budget": st.compute_budget, "spent_usd": st.spent_usd, "n_runs": len(st.history),
                  "incumbent": None if inc is None else inc.to_dict(), "stop_reason": st.stop_reason,
                  "bounds": st.bounds, "notes": st.notes}
        tmp = self.run_dir / ".status.tmp"
        tmp.write_text(json.dumps(status, default=str))
        tmp.replace(self.run_dir / "status.json")

    def summary(self) -> dict:
        st, sim = self.state, self.sim
        inc = st.incumbent()
        per_f = {}
        for r in st.history:
            per_f[r.fidelity] = per_f.get(r.fidelity, 0) + 1
        s = {
            "study": self.spec.name, "simulator": sim.name, "planner": self.spec.planner.kind,
            "model": getattr(getattr(self.planner, "provider", None), "model", None),
            "seed": self.spec.seed, "n_runs": len(st.history), "n_failed": sum(not r.ok for r in st.history),
            "n_cached": sum(r.cached for r in st.history), "runs_per_fidelity": per_f, "batches": st.batch,
            "spent_compute": st.spent_compute, "compute_budget": st.compute_budget,
            "llm_tokens_in": st.tokens_in, "llm_tokens_out": st.tokens_out, "llm_usd": st.spent_usd,
            "stop_reason": st.stop_reason or "completed",
            "incumbent": None if inc is None else {"params": inc.params, "fidelity": inc.fidelity,
                                                   "objective": inc.objective, "index": inc.index},
            "results_digest": results_digest([r.to_dict() for r in st.history]),
        }
        if self.score and inc is not None:
            ref = references().get(sim.name, {}).get("optimum")
            true = sim.true_objective(inc.params)
            s["incumbent_true_objective"] = true
            s["reference_optimum"] = ref
            s["regret"] = None if ref is None else max(ref - true, 0.0)
            curve, cache = [], {}
            for pt in self.incumbent_curve:
                i = pt["incumbent"]
                if i is None:
                    continue
                if i not in cache:
                    cache[i] = sim.true_objective(st.history[i].params)
                curve.append({"spent": pt["spent"], "true": cache[i],
                              "regret": None if ref is None else max(ref - cache[i], 0.0)})
            s["curve"] = curve
        return s
