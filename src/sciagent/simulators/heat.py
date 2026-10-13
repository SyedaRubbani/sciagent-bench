"""Steady-state heat conduction on a square plate with two heat sources (physics / thermal design).

Goal: place and size a single cooling pin (held at T=0) to minimise the peak plate temperature,
with a cost on pin area. Solved as

    -lap(T) + beta * chi(x) * T = q(x)   on [0,1]^2,   T = 0 on the boundary,

with a smoothed (diffuse-interface) indicator chi for the pin so that the objective varies smoothly
with pin position and radius. Fidelity = grid resolution; coarse grids smear the sources and the pin,
which biases the peak temperature low and shifts the apparent optimum.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .base import Fidelity, Param, Simulator

SOURCES = ((0.30, 0.35, 1.0, 0.07), (0.70, 0.62, 0.8, 0.09))  # (x, y, strength, width)
BETA = 2.0e4
AREA_COST = 6.0   # penalty per unit pin area, in units of normalised temperature
Q_SCALE = 60.0    # scales the source so the uncooled peak is O(1)


@lru_cache(maxsize=8)
def _laplacian(n: int):
    h = 1.0 / (n + 1)
    main = 4.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    t = sp.diags([off, main, off], [-1, 0, 1], shape=(n, n))
    eye = sp.identity(n)
    lap = (sp.kron(eye, t) + sp.kron(sp.diags([off, off], [-1, 1], shape=(n, n)), eye)) / h**2
    xs = np.linspace(h, 1 - h, n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    return lap.tocsr(), X, Y, h


def _source(X, Y):
    q = np.zeros_like(X)
    for (sx, sy, a, w) in SOURCES:
        q += a * np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (2 * w * w))
    return Q_SCALE * q


def solve_plate(px: float, py: float, r: float, n: int) -> np.ndarray:
    lap, X, Y, h = _laplacian(n)
    eps = max(h, 1.0 / 128) * 0.75
    dist = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
    chi = 0.5 * (1.0 - np.tanh((dist - r) / eps))
    A = lap + sp.diags(BETA * chi.ravel())
    T = spsolve(A.tocsc(), _source(X, Y).ravel())
    return T.reshape(n, n)


class HeatSimulator(Simulator):
    name = "heat"
    version = "1.0"
    description = (
        "Steady 2-D heat conduction on a unit plate with two Gaussian heat sources (a strong one near "
        "(0.30,0.35), a weaker, wider one near (0.70,0.62)) and cold edges. Choose the position and radius of "
        "one cooling pin to minimise the peak temperature; larger pins cost more. Objective = -(peak "
        "temperature) - area cost, so higher is better."
    )
    objective_name = "neg_peak_temp_penalized"
    params = (
        Param("pin_x", 0.05, 0.95, description="pin centre x"),
        Param("pin_y", 0.05, 0.95, description="pin centre y"),
        Param("pin_r", 0.01, 0.15, description="pin radius"),
    )
    fidelities = (
        Fidelity("grid16", 0, 1.0, "16x16 finite differences; smears sources and pin, peak biased low"),
        Fidelity("grid32", 1, 4.0, "32x32 finite differences"),
        Fidelity("grid64", 2, 16.0, "64x64 finite differences; reference accuracy"),
    )

    _N = {"grid16": 16, "grid32": 32, "grid64": 64}

    def _evaluate(self, p, fidelity, rng):
        T = solve_plate(p["pin_x"], p["pin_y"], p["pin_r"], self._N[fidelity.name])
        peak = float(T.max())
        area = float(np.pi * p["pin_r"] ** 2)
        return -peak - AREA_COST * area, {"peak_temp": peak, "pin_area": area, "mean_temp": float(T.mean())}
