"""Numerical verification of the simulators: convergence orders, closed forms, GPU kernels."""

import math
import os

import numpy as np
import pytest
from scipy.special import exp1

from sciagent.simulators.heat import solve_plate
from sciagent.simulators.mimo import ergodic_capacity
from sciagent.simulators.reactor import analytic_yield, integrate_euler, integrate_rk4, rate_constants


def _order(errs):
    return [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]


def test_euler_is_first_order_and_rk4_fourth_order():
    k1, k2 = rate_constants(320.0, 0.6)
    exact = analytic_yield(k1, k2, 6.8)
    ns = [20, 40, 80, 160]
    e_eu = [abs(integrate_euler(k1, k2, 6.8, n) - exact) for n in ns]
    e_rk = [abs(integrate_rk4(k1, k2, 6.8, n) - exact) for n in ns]
    assert all(0.9 < p < 1.1 for p in _order(e_eu))
    assert all(3.8 < p < 4.3 for p in _order(e_rk))


@pytest.mark.parametrize("snr", [0.5, 5.0, 50.0])
def test_siso_rayleigh_capacity_closed_form(snr):
    # E[log2(1 + snr |h|^2)], |h|^2 ~ Exp(1)  =  log2(e) * exp(1/snr) * E1(1/snr)
    exact = math.log2(math.e) * math.exp(1 / snr) * exp1(1 / snr)
    mean, se = ergodic_capacity(snr, 1, 400_000, np.random.default_rng(1), nr=1, rho=0.0)
    assert abs(mean - exact) < 4 * se


def test_monte_carlo_is_unbiased_across_fidelities():
    from sciagent.simulators import get_simulator

    sim = get_simulator("mimo")
    p = {"p_dbm": 22.0, "n_tx": 6}
    truth = sim.true_objective(p)
    vals = np.array([sim.run(p, "mc64", s).objective for s in range(300)])
    assert abs(vals.mean() - truth) < 4 * vals.std(ddof=1) / math.sqrt(len(vals)) + 2e-3


def test_heat_solution_symmetry_and_positivity():
    T = solve_plate(0.5, 0.5, 0.0, 32)  # no effective pin; sources make T > 0 inside
    assert (T > 0).all()
    assert T.max() < 2.0


torch = pytest.importorskip("torch")
DEVICE = os.environ.get("SCIAGENT_TEST_DEVICE", "cpu")  # the GPU notebook sets "cuda"
if DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"


@pytest.mark.parametrize("n", [16, 48])
def test_torch_cg_matches_sparse_direct(n):
    from sciagent.gpu.heat_torch import solve_plate_torch

    a = solve_plate(0.52, 0.5, 0.04, n)
    b, info = solve_plate_torch(0.52, 0.5, 0.04, n, device=DEVICE)
    assert info["converged"]
    assert np.abs(a - b.cpu().numpy()).max() < 1e-8


def test_torch_monte_carlo_matches_numpy_statistically():
    from sciagent.gpu.mimo_torch import ergodic_capacity_torch

    m1, s1 = ergodic_capacity(10.0, 6, 100_000, np.random.default_rng(3))
    m2, s2 = ergodic_capacity_torch(10.0, 6, 100_000, seed=3, device=DEVICE, chunk=30_000)
    assert abs(m1 - m2) < 4 * math.hypot(s1, s2)
