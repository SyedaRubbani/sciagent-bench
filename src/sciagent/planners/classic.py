"""Non-LLM baselines: random search, multi-fidelity successive halving, Bayesian optimisation."""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from ..state import Plan, RunRecord, RunRequest, StudyState
from .base import Planner


class RandomPlanner(Planner):
    """Uniform random search at one fidelity (default: the most accurate)."""

    name = "random"

    def propose(self, state: StudyState) -> Plan:
        fid = state.sim.fidelity(self.options.get("fidelity", state.sim.top_fidelity.name))
        n = min(state.spec.batch_size, int(state.remaining_compute // fid.cost))
        if n <= 0:
            return Plan([], "budget exhausted", stop=True)
        return Plan([RunRequest(self.sample_point(state), fid.name, "random sample") for _ in range(n)],
                    f"{n} uniform random samples at {fid.name}")


class SuccessiveHalvingPlanner(Planner):
    """Budget-aware successive halving across fidelity levels (one Hyperband-style bracket at a time).

    Start many configurations at the cheapest fidelity, promote the top 1/eta to the next fidelity,
    and so on; when a bracket finishes and budget remains, start a new, smaller bracket.
    """

    name = "sh"

    def __init__(self, options=None, seed=0):
        super().__init__(options, seed)
        self.eta = float(self.options.get("eta", 3))
        self.rung = 0
        self.queue: list[RunRequest] = []
        self.rung_results: list[RunRecord] = []
        self.rung_size = 0
        self.awaiting = 0

    def _bracket_size(self, state: StudyState) -> int:
        costs = [f.cost for f in state.sim.fidelities]
        per_config = sum(c / self.eta**i for i, c in enumerate(costs))
        n = int(state.remaining_compute // per_config)
        return int(min(max(n, 0), self.options.get("max_configs", 81)))

    def _start_bracket(self, state: StudyState) -> bool:
        n = self._bracket_size(state)
        if n < 1:
            return False
        f0 = state.sim.fidelities[0].name
        self.rung, self.rung_size, self.rung_results = 0, n, []
        self.queue = [RunRequest(self.sample_point(state), f0, "new bracket: explore at cheapest fidelity")
                      for _ in range(n)]
        return True

    def _promote(self, state: StudyState) -> bool:
        if self.rung + 1 >= len(state.sim.fidelities):
            return False
        ok = sorted((r for r in self.rung_results if r.ok and r.objective is not None),
                    key=lambda r: r.objective, reverse=True)
        k = max(1, math.ceil(len(ok) / self.eta)) if ok else 0
        if k == 0:
            return False
        self.rung += 1
        fid = state.sim.fidelities[self.rung]
        k = min(k, int(state.remaining_compute // fid.cost))
        if k == 0:
            return False
        self.queue = [RunRequest(dict(r.params), fid.name, f"promoted: top-{k} of rung {self.rung - 1}") for r in ok[:k]]
        self.rung_size, self.rung_results = k, []
        return True

    def propose(self, state: StudyState) -> Plan:
        if not self.queue and self.awaiting == 0:
            if self.rung_size and len(self.rung_results) >= self.rung_size and self._promote(state):
                pass
            elif not self._start_bracket(state):
                return Plan([], "no budget for another bracket", stop=True)
        batch, self.queue = self.queue[: state.spec.batch_size], self.queue[state.spec.batch_size:]
        self.awaiting = len(batch)
        fid = batch[0].fidelity if batch else "-"
        return Plan(batch, f"successive halving rung {self.rung} ({fid}), {len(self.queue)} queued")

    def observe(self, records):
        self.rung_results.extend(records)
        self.awaiting = 0


class BayesOptPlanner(Planner):
    """GP + expected improvement at a single fidelity, batched with the 'kriging believer' heuristic."""

    name = "bo"

    def propose(self, state: StudyState) -> Plan:
        fid = state.sim.fidelity(self.options.get("fidelity", state.sim.top_fidelity.name))
        n = min(state.spec.batch_size, int(state.remaining_compute // fid.cost))
        if n <= 0:
            return Plan([], "budget exhausted", stop=True)
        data = [r for r in state.ok_runs() if r.fidelity == fid.name]
        d = len(state.sim.params)
        n_init = int(self.options.get("n_init", max(2 * d, state.spec.batch_size)))
        if len(data) < n_init:
            pts = [self.sample_point(state) for _ in range(n)]
            return Plan([RunRequest(p, fid.name, "initial design") for p in pts], f"BO initial design at {fid.name}")
        X = np.array([self.to_unit(state, r.params) for r in data])
        y = np.array([r.objective for r in data])
        picks = suggest_ei(X, y, n, self.rng, d)
        runs = [RunRequest(self.from_unit(state, u), fid.name, f"EI={ei:.3g}") for u, ei in picks]
        return Plan(runs, f"BO: {n} EI picks at {fid.name} from {len(data)} observations")


def fit_gp(X: np.ndarray, y: np.ndarray):
    mu, sd = float(y.mean()), float(y.std() or 1.0)
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.full(X.shape[1], 0.3), nu=2.5,
                                          length_scale_bounds=(1e-2, 10)) + WhiteKernel(1e-4, (1e-8, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=2, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, (y - mu) / sd)
    return gp, mu, sd


def _ei(m, s, best, xi=0.01):
    s = np.maximum(s, 1e-9)
    z = (m - best - xi) / s
    return (m - best - xi) * norm.cdf(z) + s * norm.pdf(z)


def suggest_ei(X: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator, d: int,
               n_cand: int = 3000) -> list[tuple[np.ndarray, float]]:
    """Return n (unit-cube point, EI) pairs using kriging-believer fantasies."""
    Xf, yf = X.copy(), y.copy()
    picks = []
    for _ in range(n):
        gp, mu, sd = fit_gp(Xf, yf)
        best = (yf.max() - mu) / sd
        top = Xf[np.argsort(yf)[-3:]]
        local = np.clip(top[rng.integers(0, len(top), n_cand // 2)] + rng.normal(0, 0.05, (n_cand // 2, d)), 0, 1)
        cand = np.vstack([rng.uniform(0, 1, (n_cand - n_cand // 2, d)), local])
        m, s = gp.predict(cand, return_std=True)
        ei = _ei(m, s, best)
        i = int(np.argmax(ei))
        picks.append((cand[i], float(ei[i] * sd)))
        Xf = np.vstack([Xf, cand[i]])
        yf = np.append(yf, m[i] * sd + mu)  # believe the posterior mean
    return picks


def gp_predict(X, y, Xq):
    gp, mu, sd = fit_gp(X, y)
    m, s = gp.predict(Xq, return_std=True)
    return m * sd + mu, s * sd
