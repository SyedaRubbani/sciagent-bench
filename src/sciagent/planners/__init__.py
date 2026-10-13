from __future__ import annotations

from ..spec import StudySpec
from .base import Planner
from .classic import BayesOptPlanner, RandomPlanner, SuccessiveHalvingPlanner


def make_planner(spec: StudySpec, provider=None) -> Planner:
    kind = spec.planner.kind
    opts = spec.planner.options
    if kind == "random":
        return RandomPlanner(opts, spec.seed)
    if kind == "sh":
        return SuccessiveHalvingPlanner(opts, spec.seed)
    if kind == "bo":
        return BayesOptPlanner(opts, spec.seed)
    if kind == "llm":
        from .llm_planner import LLMPlanner

        return LLMPlanner(spec, provider=provider)
    raise ValueError(kind)


__all__ = ["Planner", "make_planner", "RandomPlanner", "SuccessiveHalvingPlanner", "BayesOptPlanner"]
