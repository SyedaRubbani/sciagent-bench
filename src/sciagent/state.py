"""Shared data structures passed between the orchestrator, planners and the API."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .budget import BudgetLedger
from .simulators.base import Param, Simulator
from .spec import StudySpec


@dataclass
class RunRequest:
    params: dict[str, Any]
    fidelity: str
    why: str = ""
    token: int | None = None  # budget reservation held between guard and execution


@dataclass
class Plan:
    runs: list[RunRequest]
    rationale: str = ""
    stop: bool = False
    usage: dict[str, Any] = field(default_factory=dict)  # tokens, usd, latency, repairs


@dataclass
class RunRecord:
    index: int
    batch: int
    params: dict[str, Any]
    fidelity: str
    fidelity_level: int
    seed: int
    objective: float | None
    metrics: dict[str, float]
    cost: float
    wall_s: float
    ok: bool
    error: str | None
    cached: bool
    why: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["objective"] is not None and not math.isfinite(d["objective"]):
            d["objective"] = None
        return d


class StudyState:
    def __init__(self, spec: StudySpec, sim: Simulator):
        self.spec = spec
        self.sim = sim
        self.history: list[RunRecord] = []
        self.batch = 0
        self.ledger = BudgetLedger(spec.budget.compute_units)
        self.spent_usd = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.notes: list[str] = []
        self.bounds: dict[str, tuple[float, float]] = {}
        for p in sim.params:
            lo, hi = spec.bounds.get(p.name, (p.low, p.high))
            self.bounds[p.name] = (max(lo, p.low), min(hi, p.high))
        self.started = time.time()
        self.stop_reason: str | None = None

    # --- views ---------------------------------------------------------------
    @property
    def remaining_compute(self) -> float:
        return self.ledger.remaining

    @property
    def spent_compute(self) -> float:
        return self.ledger.committed

    @property
    def compute_budget(self) -> float:
        return self.ledger.total

    def effective_params(self) -> list[Param]:
        return [Param(p.name, *self.bounds[p.name], kind=p.kind, unit=p.unit, description=p.description)
                for p in self.sim.params]

    def ok_runs(self) -> list[RunRecord]:
        return [r for r in self.history if r.ok and r.objective is not None]

    def incumbent(self) -> RunRecord | None:
        """Best result at the highest fidelity evaluated so far (what we would report to the researcher)."""
        ok = self.ok_runs()
        if not ok:
            return None
        top = max(r.fidelity_level for r in ok)
        return max((r for r in ok if r.fidelity_level == top), key=lambda r: r.objective)

    def cheapest_cost(self) -> float:
        return min(f.cost for f in self.sim.fidelities)
