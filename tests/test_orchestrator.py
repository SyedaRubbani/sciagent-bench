import json

import pytest

from sciagent.guard import check_plan
from sciagent.orchestrator import Orchestrator
from sciagent.simulators import get_simulator
from sciagent.state import Plan, RunRequest, StudyState


@pytest.mark.parametrize("kind", ["random", "sh", "bo", "llm"])
@pytest.mark.parametrize("sim", ["reactor", "mimo"])
def test_planners_respect_budget(make_spec, kind, sim, tmp_path):
    spec = make_spec(sim, kind, budget=100)
    s = Orchestrator(spec, tmp_path / "run").run()
    assert s["spent_compute"] <= 100 + 1e-9
    assert s["n_runs"] > 0 and s["incumbent"] is not None
    assert s["regret"] is not None and s["regret"] >= 0
    for f in ("trace.jsonl", "runs.jsonl", "summary.json", "manifest.json", "spec.yaml", "status.json"):
        assert (tmp_path / "run" / f).exists()


def test_guard_clips_drops_and_budgets(make_spec):
    spec = make_spec("reactor", "random", budget=20, bounds={"T": (300, 340)})
    st = StudyState(spec, get_simulator("reactor"))
    plan = Plan([
        RunRequest({"T": 400, "tau": 1, "cat": 0.1}, "bdf_fine"),           # clipped to T=340
        RunRequest({"T": 320, "tau": 1, "cat": 0.1}, "nonexistent"),          # dropped
        RunRequest({"T": 320, "tau": 1}, "euler_coarse"),                      # dropped (missing param)
        RunRequest({"T": 320, "tau": 2, "cat": 0.1}, "bdf_fine"),             # dropped: 16+16 > 20
        RunRequest({"T": 320, "tau": 2, "cat": 0.1}, "rk4_medium"),           # ok
    ])
    acc, issues = check_plan(plan, st)
    assert [r.fidelity for r in acc] == ["bdf_fine", "rk4_medium"]
    assert acc[0].params["T"] == 340
    actions = sorted(i["action"] for i in issues)
    assert actions.count("dropped") == 3 and "clipped" in actions


def test_deterministic_duplicates_are_not_rerun(make_spec):
    spec = make_spec("reactor", "random", budget=100)
    o = Orchestrator(spec)
    o._execute([RunRequest({"T": 320, "tau": 2, "cat": 0.1}, "rk4_medium")], None)
    acc, issues = check_plan(Plan([RunRequest({"T": 320, "tau": 2, "cat": 0.1}, "rk4_medium")]), o.state)
    assert not acc and "duplicate" in issues[0]["reason"]


def test_runs_are_reproducible(make_spec):
    a = Orchestrator(make_spec("heat", "bo", budget=80)).run()
    b = Orchestrator(make_spec("heat", "bo", budget=80)).run()
    assert a["results_digest"] == b["results_digest"]
    c = Orchestrator(make_spec("heat", "bo", budget=80, seed=7)).run()
    assert c["results_digest"] != a["results_digest"]


def test_trace_has_parented_spans(make_spec, tmp_path):
    Orchestrator(make_spec("reactor", "sh", budget=40), tmp_path).run()
    evs = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    ids = {e["span_id"] for e in evs}
    assert all(e["parent_id"] in ids for e in evs if e["parent_id"])
    kinds = {e["kind"] for e in evs}
    assert {"study.start", "batch.start", "plan", "run", "execute", "batch.end", "study.end"} <= kinds


def test_steering_note_bounds_budget_and_stop(make_spec, tmp_path):
    from sciagent.steering import steer

    spec = make_spec("reactor", "random", budget=400)
    steer(tmp_path, note="stay cool", bounds={"T": (300, 330)}, compute_units=64)
    s = Orchestrator(spec, tmp_path).run()
    runs = [json.loads(x) for x in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert all(r["params"]["T"] <= 330 for r in runs)
    assert s["spent_compute"] <= 64
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["notes"] == ["stay cool"]

    d2 = tmp_path / "second"
    d2.mkdir()
    steer(d2, stop=True)
    s2 = Orchestrator(make_spec("reactor", "random", budget=400), d2).run()
    assert s2["n_runs"] == 0 and s2["stop_reason"] == "stopped by researcher"


def test_failed_simulation_is_recorded_not_fatal(make_spec, monkeypatch):
    from sciagent.simulators.reactor import ReactorSimulator

    def boom(self, p, f, rng):
        raise RuntimeError("solver diverged")
    monkeypatch.setattr(ReactorSimulator, "_evaluate", boom)
    o = Orchestrator(make_spec("reactor", "random", budget=48), score=False)
    s = o.run()
    assert s["n_failed"] == s["n_runs"] == 3 and s["incumbent"] is None
    assert "solver diverged" in o.state.history[0].error


def test_process_pool_matches_inline(make_spec):
    a = Orchestrator(make_spec("mimo", "random", budget=64)).run()
    b = Orchestrator(make_spec("mimo", "random", budget=64, executor={"kind": "local", "workers": 2})).run()
    assert a["results_digest"] == b["results_digest"]


def test_slurm_executor_via_submitit_local(make_spec, tmp_path):
    pytest.importorskip("submitit")
    spec = make_spec("reactor", "random", budget=32, executor={"kind": "slurm", "slurm": {"cluster": "debug"}})
    s = Orchestrator(spec, tmp_path).run()
    ref = Orchestrator(make_spec("reactor", "random", budget=32)).run()
    assert s["results_digest"] == ref["results_digest"]
