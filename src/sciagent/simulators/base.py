"""Simulator interface.

A simulator is a black-box scientific code with:
  * a typed parameter space (what the researcher can change),
  * an ordered set of fidelity levels, each with a declared cost in compute units,
  * a deterministic ``run(params, fidelity, seed)``.

Declared costs (not wall-clock) are used for budgeting so that campaigns are reproducible
across machines; measured wall-clock is still recorded in every result.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Param:
    name: str
    low: float
    high: float
    kind: str = "float"  # "float" | "int"
    unit: str = ""
    description: str = ""

    def clip(self, value: float) -> float | int:
        v = min(max(float(value), self.low), self.high)
        return int(round(v)) if self.kind == "int" else v

    def sample(self, rng: np.random.Generator) -> float | int:
        if self.kind == "int":
            return int(rng.integers(int(self.low), int(self.high) + 1))
        return float(rng.uniform(self.low, self.high))

    def to_unit(self, value: float) -> float:
        return (float(value) - self.low) / (self.high - self.low)

    def from_unit(self, u: float) -> float | int:
        return self.clip(self.low + float(u) * (self.high - self.low))


@dataclass(frozen=True)
class Fidelity:
    name: str
    level: int  # 0 = cheapest; higher = more accurate
    cost: float  # declared compute units per run
    notes: str = ""  # what a researcher should know (bias, noise, ...)


@dataclass
class SimResult:
    objective: float
    metrics: dict[str, float] = field(default_factory=dict)
    cost: float = 0.0
    wall_s: float = 0.0
    ok: bool = True
    error: str | None = None


class Simulator(ABC):
    name: str = "base"
    version: str = "0"
    description: str = ""
    objective_name: str = "objective"
    objective_sense: str = "maximize"
    stochastic: bool = False  # True if the seed changes the result (e.g. Monte-Carlo)
    params: tuple[Param, ...] = ()
    fidelities: tuple[Fidelity, ...] = ()

    # --- to implement -------------------------------------------------------------
    @abstractmethod
    def _evaluate(self, params: dict[str, Any], fidelity: Fidelity, rng: np.random.Generator) -> tuple[float, dict]:
        """Return (objective, metrics). Higher objective is better."""

    def true_objective(self, params: dict[str, Any]) -> float:
        """Noise-free, highest-accuracy objective used only for *scoring* (never by planners)."""
        top = self.fidelities[-1]
        obj, _ = self._evaluate(params, top, np.random.default_rng(12345))
        return obj

    # --- shared ---------------------------------------------------------------------
    def fidelity(self, name: str) -> Fidelity:
        for f in self.fidelities:
            if f.name == name:
                return f
        raise KeyError(f"unknown fidelity {name!r} for {self.name}; options: {[f.name for f in self.fidelities]}")

    @property
    def top_fidelity(self) -> Fidelity:
        return self.fidelities[-1]

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def normalize(self, params: dict[str, Any]) -> dict[str, float | int]:
        missing = [p.name for p in self.params if p.name not in params]
        if missing:
            raise ValueError(f"missing parameters {missing}")
        extra = set(params) - {p.name for p in self.params}
        if extra:
            raise ValueError(f"unknown parameters {sorted(extra)}")
        return {p.name: p.clip(params[p.name]) for p in self.params}

    def run(self, params: dict[str, Any], fidelity: str, seed: int) -> SimResult:
        fid = self.fidelity(fidelity)
        t0 = time.perf_counter()
        try:
            p = self.normalize(params)
            obj, metrics = self._evaluate(p, fid, np.random.default_rng(seed))
            if not math.isfinite(obj):
                raise FloatingPointError(f"non-finite objective {obj}")
            return SimResult(objective=float(obj), metrics=metrics, cost=fid.cost, wall_s=time.perf_counter() - t0)
        except Exception as e:  # a failed run still consumes budget, as on a real cluster
            return SimResult(objective=float("nan"), cost=fid.cost, wall_s=time.perf_counter() - t0, ok=False,
                             error=f"{type(e).__name__}: {e}")

    def card(self) -> dict:
        """Machine- and LLM-readable description of the simulator."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "objective": {"name": self.objective_name, "sense": self.objective_sense},
            "params": [p.__dict__ for p in self.params],
            "fidelities": [f.__dict__ for f in self.fidelities],
        }
