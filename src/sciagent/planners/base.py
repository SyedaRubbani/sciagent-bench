from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..state import Plan, RunRecord, StudyState


class Planner(ABC):
    """Turns the current study state into the next batch of runs."""

    name = "base"

    def __init__(self, options: dict | None = None, seed: int = 0):
        self.options = dict(options or {})
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def propose(self, state: StudyState) -> Plan: ...

    def observe(self, records: list[RunRecord]) -> None:  # noqa: B027 - optional hook
        """Called with the results of the batch just executed."""

    def feedback(self, issues: list[dict], n_accepted: int) -> None:  # noqa: B027 - optional hook
        """Called after the guard has checked the last proposal: what it dropped, clipped or accepted."""

    # helpers -----------------------------------------------------------------------
    def sample_point(self, state: StudyState) -> dict:
        return {p.name: p.sample(self.rng) for p in state.effective_params()}

    def to_unit(self, state: StudyState, params: dict) -> np.ndarray:
        return np.array([p.to_unit(params[p.name]) for p in state.effective_params()])

    def from_unit(self, state: StudyState, u: np.ndarray) -> dict:
        return {p.name: p.from_unit(float(np.clip(x, 0, 1))) for p, x in zip(state.effective_params(), u)}
