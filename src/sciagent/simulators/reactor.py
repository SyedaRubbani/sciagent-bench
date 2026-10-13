"""Consecutive reaction A -> B -> C in an isothermal batch reactor (chemical engineering).

Goal: maximise the yield of the intermediate B, minus heating, catalyst and residence-time costs.

    dA/dt = -k1 A,  dB/dt = k1 A - k2 B,  A(0)=1, B(0)=0
    k_i(T) = A_i exp(-Ea_i / (R T)),  k1 is boosted by catalyst loading.

Because Ea2 > Ea1, lower temperature favours selectivity to B but slows the reaction, and the
residence time is capped -- so the optimum is an interior trade-off between T, tau and catalyst.
The closed-form solution gives an exact ground truth for scoring planners.

Fidelities differ in *integration error* (systematic bias), as with real solvers:
  euler_coarse  explicit Euler, 6 steps           (cheap, biased)
  rk4_medium    classical RK4, 40 steps           (small bias)
  bdf_fine      scipy BDF, rtol 1e-9              (reference quality)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .base import Fidelity, Param, Simulator

R = 8.314
A1, EA1 = 1.45e7, 50_000.0
A2, EA2 = 8.4e9, 70_000.0
HEAT_COST = 0.06  # objective penalty across the full temperature range
CAT_COST = 0.10   # objective penalty at full catalyst loading
TIME_COST = 0.08  # objective penalty at the maximum residence time (lost throughput)


def rate_constants(T: float, cat: float) -> tuple[float, float]:
    k1 = A1 * np.exp(-EA1 / (R * T)) * (1.0 + 3.0 * np.sqrt(max(cat, 0.0)))
    k2 = A2 * np.exp(-EA2 / (R * T))
    return float(k1), float(k2)


def analytic_yield(k1: float, k2: float, t: float) -> float:
    if abs(k2 - k1) < 1e-12:
        return k1 * t * np.exp(-k1 * t)
    return k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))


def _rhs(k1: float, k2: float):
    def f(_t, y):
        a, b = y
        return np.array([-k1 * a, k1 * a - k2 * b])
    return f


def integrate_euler(k1: float, k2: float, tau: float, n: int) -> float:
    """Explicit Euler for B(tau); first-order accurate."""
    a, b, dt = 1.0, 0.0, tau / n
    for _ in range(n):
        a, b = a - dt * k1 * a, b + dt * (k1 * a - k2 * b)
    return float(b)


def integrate_rk4(k1: float, k2: float, tau: float, n: int) -> float:
    """Classical Runge-Kutta 4 for B(tau); fourth-order accurate."""
    f = _rhs(k1, k2)
    y, dt = np.array([1.0, 0.0]), tau / n
    for _ in range(n):
        s1 = f(0, y)
        s2 = f(0, y + 0.5 * dt * s1)
        s3 = f(0, y + 0.5 * dt * s2)
        s4 = f(0, y + dt * s3)
        y = y + dt / 6.0 * (s1 + 2 * s2 + 2 * s3 + s4)
    return float(y[1])


class ReactorSimulator(Simulator):
    name = "reactor"
    version = "1.0"
    description = (
        "Isothermal batch reactor with consecutive first-order reactions A->B->C. Maximise yield of the "
        "intermediate B minus heating, catalyst and lost-throughput (time) costs. Higher temperature speeds both reactions but "
        "favours the unwanted B->C step (higher activation energy)."
    )
    objective_name = "net_yield_B"
    params = (
        Param("T", 300.0, 420.0, unit="K", description="reactor temperature"),
        Param("tau", 0.05, 10.0, unit="min", description="residence (batch) time"),
        Param("cat", 0.0, 1.0, unit="fraction", description="catalyst loading (boosts A->B only, diminishing returns)"),
    )
    fidelities = (
        Fidelity("euler_coarse", 0, 1.0, "explicit Euler with 6 steps; biased, and numerically unstable when k*tau/6 > 2 (hot, long runs)"),
        Fidelity("rk4_medium", 1, 4.0, "RK4 with 40 steps; small bias"),
        Fidelity("bdf_fine", 2, 16.0, "stiff BDF solver, rtol 1e-9; reference accuracy"),
    )

    def _penalty(self, p) -> float:
        return HEAT_COST * (p["T"] - 300.0) / 120.0 + CAT_COST * p["cat"] + TIME_COST * p["tau"] / 10.0

    def _evaluate(self, p, fidelity, rng):
        k1, k2 = rate_constants(p["T"], p["cat"])
        tau = p["tau"]
        if fidelity.name == "euler_coarse":
            b = integrate_euler(k1, k2, tau, 6)
        elif fidelity.name == "rk4_medium":
            b = integrate_rk4(k1, k2, tau, 40)
        else:
            sol = solve_ivp(_rhs(k1, k2), (0.0, tau), np.array([1.0, 0.0]), method="BDF", rtol=1e-9, atol=1e-12)
            b = float(sol.y[1, -1])
        unstable = not (-0.01 <= b <= 1.01)
        b_rep = float(np.clip(b, -1.0, 2.0))  # an unstable explicit solve is reported, not hidden
        obj = b_rep - self._penalty(p)
        return obj, {"yield_B": b_rep, "numerically_unstable": float(unstable), "k1": k1, "k2": k2, "conversion_A": 1.0 - float(np.exp(-k1 * tau))}

    def true_objective(self, params) -> float:
        p = self.normalize(params)
        k1, k2 = rate_constants(p["T"], p["cat"])
        return float(analytic_yield(k1, k2, p["tau"]) - self._penalty(p))
