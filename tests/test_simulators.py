import numpy as np
import pytest

from sciagent.simulators import get_simulator, list_simulators, references
from sciagent.simulators.reactor import analytic_yield, rate_constants


def test_registry():
    assert set(list_simulators()) >= {"reactor", "heat", "mimo"}
    with pytest.raises(KeyError):
        get_simulator("nope")


@pytest.mark.parametrize("name", ["reactor", "heat", "mimo"])
def test_card_and_run(name):
    sim = get_simulator(name)
    card = sim.card()
    assert card["params"] and card["fidelities"]
    costs = [f.cost for f in sim.fidelities]
    assert costs == sorted(costs), "fidelities must be ordered by cost"
    rng = np.random.default_rng(0)
    p = {q.name: q.sample(rng) for q in sim.params}
    for f in sim.fidelities:
        r = sim.run(p, f.name, seed=1)
        assert r.ok and np.isfinite(r.objective) and r.cost == f.cost


def test_reactor_bdf_matches_closed_form():
    sim = get_simulator("reactor")
    for p in [dict(T=320, tau=6.8, cat=0.6), dict(T=360, tau=1.0, cat=0.1), dict(T=305, tau=9.0, cat=1.0)]:
        r = sim.run(p, "bdf_fine", 0)
        assert abs(r.objective - sim.true_objective(p)) < 1e-6
        k1, k2 = rate_constants(p["T"], p["cat"])
        assert abs(r.metrics["yield_B"] - analytic_yield(k1, k2, p["tau"])) < 1e-6


def test_reactor_cheap_fidelity_is_biased():
    sim = get_simulator("reactor")
    p = references()["reactor"]["argmax"]
    err = {f.name: abs(sim.run(p, f.name, 0).objective - sim.true_objective(p)) for f in sim.fidelities}
    assert err["euler_coarse"] > err["rk4_medium"] > err["bdf_fine"]


def test_heat_grid_convergence():
    sim = get_simulator("heat")
    p = dict(pin_x=0.5, pin_y=0.5, pin_r=0.04)
    vals = [sim.run(p, f.name, 0).objective for f in sim.fidelities]
    assert abs(vals[1] - vals[2]) < abs(vals[0] - vals[2])


def test_mimo_noise_shrinks_with_draws():
    sim = get_simulator("mimo")
    p = dict(p_dbm=22.0, n_tx=6)
    spread = {f.name: np.std([sim.run(p, f.name, s).objective for s in range(12)]) for f in sim.fidelities}
    assert spread["mc64"] > spread["mc4096"]


def test_normalize_rejects_unknown_and_missing():
    sim = get_simulator("reactor")
    with pytest.raises(ValueError):
        sim.normalize({"T": 300, "tau": 1})
    with pytest.raises(ValueError):
        sim.normalize({"T": 300, "tau": 1, "cat": 0, "x": 1})


def test_references_beat_random_sampling():
    for name in ("reactor", "heat"):
        sim = get_simulator(name)
        rng = np.random.default_rng(1)
        best = max(sim.true_objective({q.name: q.sample(rng) for q in sim.params}) for _ in range(150))
        assert references()[name]["optimum"] >= best - 1e-9
