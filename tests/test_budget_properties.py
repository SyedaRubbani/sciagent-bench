"""Property-based tests: no proposal, however adversarial, can break the budget or the bounds."""

import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sciagent.budget import BudgetExceeded, BudgetLedger, llm_call_affordable
from sciagent.guard import check_plan
from sciagent.llm.providers import LLMResponse, Provider
from sciagent.orchestrator import Orchestrator
from sciagent.simulators import get_simulator
from sciagent.spec import StudySpec
from sciagent.state import Plan, RunRequest, StudyState

SIM = get_simulator("reactor")
FIDS = [f.name for f in SIM.fidelities] + ["bogus"]
num = st.one_of(st.floats(allow_nan=False, allow_infinity=False, width=32), st.integers(-10**6, 10**6))
run = st.builds(lambda T, tau, cat, f, drop: RunRequest(
    {k: v for k, v in {"T": T, "tau": tau, "cat": cat}.items() if k != drop}, f),
    num, num, num, st.sampled_from(FIDS), st.sampled_from([None, None, None, "cat"]))


@settings(max_examples=300, deadline=None)
@given(st.lists(run, max_size=40), st.floats(1, 200), st.integers(1, 12))
def test_guard_never_exceeds_budget_or_bounds(runs, budget, batch):
    spec = StudySpec(name="p", simulator="reactor", budget={"compute_units": budget}, batch_size=batch,
                     bounds={"T": (310, 350)})
    state = StudyState(spec, SIM)
    acc, _ = check_plan(Plan(runs), state, reserve=True)
    assert len(acc) <= batch
    assert sum(SIM.fidelity(a.fidelity).cost for a in acc) <= budget + 1e-9
    assert state.ledger.reserved <= budget + 1e-9
    for a in acc:
        lo, hi = state.bounds["T"]
        assert lo <= a.params["T"] <= hi
        for p in SIM.params:
            assert p.low <= a.params[p.name] <= p.high


@settings(max_examples=200, deadline=None)
@given(st.floats(0, 100), st.lists(st.tuples(st.floats(0, 30), st.floats(0, 1), st.booleans()), max_size=30))
def test_ledger_invariant(total, ops):
    led = BudgetLedger(total)
    for cost, frac, commit in ops:
        try:
            t = led.reserve(cost)
        except BudgetExceeded:
            assert cost > led.remaining
            continue
        if commit:
            led.commit(t, cost * frac)
        else:
            led.release(t)
        led.check()
    assert led.committed <= total + 1e-6 and math.isclose(led.reserved, 0.0, abs_tol=1e-9)


class Greedy(Provider):
    """An 'LLM' that always asks for 1000 top-fidelity runs."""

    model = "greedy"

    def complete(self, system, messages, temperature, max_tokens):
        import json

        runs = [{"params": {"T": 320 + i % 30, "tau": 5 + (i % 7) / 10, "cat": 0.5}, "fidelity": "bdf_fine"}
                for i in range(1000)]
        return LLMResponse(json.dumps({"runs": runs}), 100, 100, self.model)


@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.floats(10, 150))
def test_greedy_llm_cannot_overspend(budget):
    spec = StudySpec(name="g", simulator="reactor", budget={"compute_units": budget}, planner={"kind": "llm"})
    s = Orchestrator(spec, provider=Greedy(), score=False).run()
    assert s["spent_compute"] <= budget + 1e-9


def test_llm_usd_cap_is_hard():
    ok, worst = llm_call_affordable(0.0, 0.01, 40_000, 1500, 3.0, 15.0)
    assert not ok and worst > 0.01
    assert llm_call_affordable(0.0, None, 10, 10, 1, 1)[0]
    spec = StudySpec(name="u", simulator="reactor", budget={"compute_units": 5000, "llm_usd": 0.1, "max_batches": 500},
                     planner={"kind": "llm", "llm": {"provider": "mock", "price_in_per_mtok": 3.0,
                                                     "price_out_per_mtok": 15.0}})
    s = Orchestrator(spec, score=False).run()
    assert 0 < s["llm_usd"] <= 0.1 and "LLM budget would be exceeded" in s["stop_reason"]
