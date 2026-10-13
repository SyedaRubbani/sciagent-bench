import json

import pytest

from sciagent.executors import InlineExecutor
from sciagent.orchestrator import Orchestrator
from sciagent.steering import steer


class CrashAfter(InlineExecutor):
    """Simulates a preempted job: dies on the n-th batch."""

    def __init__(self, n):
        self.n, self.calls = n, 0

    def map(self, sim_name, jobs):
        self.calls += 1
        if self.calls == self.n:
            raise KeyboardInterrupt("preempted")
        return super().map(sim_name, jobs)


@pytest.mark.parametrize("kind,sim", [("sh", "reactor"), ("bo", "heat"), ("random", "mimo"), ("llm", "reactor")])
def test_resume_after_crash_is_identical(make_spec, tmp_path, kind, sim):
    ref = Orchestrator(make_spec(sim, kind, budget=120)).run()
    d = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        Orchestrator(make_spec(sim, kind, budget=120), d, executor=CrashAfter(2 if kind in ("bo", "random") else 5)).run()
    assert not (d / "summary.json").exists()
    o = Orchestrator.resume(d)
    assert o._resumed_from > 0
    s = o.run()
    assert s["results_digest"] == ref["results_digest"]
    assert s["spent_compute"] == ref["spent_compute"]
    kinds = [json.loads(x)["kind"] for x in (d / "trace.jsonl").read_text().splitlines()]
    assert "resume" in kinds and "resume.diverged" not in kinds


def test_resume_reapplies_steering_at_original_batch(make_spec, tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    steer(d, bounds={"T": (300, 330)}, note="cool")
    ref = Orchestrator(make_spec("reactor", "random", budget=96), d).run()
    (d / "summary.json").unlink()
    s = Orchestrator.resume(d).run()  # resume a *finished* run: pure fast-forward, nothing new
    assert s["results_digest"] == ref["results_digest"]
    runs = [json.loads(x) for x in (d / "runs.jsonl").read_text().splitlines()]
    assert all(r["params"]["T"] <= 330 for r in runs)


def test_redirect_switches_planner_mid_run(make_spec, tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    spec = make_spec("reactor", "random", budget=160)
    o = Orchestrator(spec, d)
    calls = {"n": 0}
    orig = o.planner.propose

    def propose_then_redirect(state):
        calls["n"] += 1
        if calls["n"] == 2:
            steer(d, planner={"kind": "sh"}, note="switch to multi-fidelity")
        return orig(state)
    o.planner.propose = propose_then_redirect
    s = o.run()
    evs = [json.loads(x) for x in (d / "trace.jsonl").read_text().splitlines()]
    steer_ev = [e for e in evs if e["kind"] == "steer"]
    assert steer_ev and steer_ev[0]["data"]["applied"]["planner"]["kind"] == "sh"
    planners = [e["data"]["planner"] for e in evs if e["kind"] == "plan"]
    assert planners[0] == "random" and planners[-1] == "sh"
    assert s["spent_compute"] <= 160
