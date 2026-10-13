"""Simulator registry. Add your own with :func:`register` (see docs/ADDING_A_SIMULATOR.md)."""

from __future__ import annotations

import importlib
import json
from functools import lru_cache
from pathlib import Path

from .base import Fidelity, Param, SimResult, Simulator
from .heat import HeatSimulator
from .mimo import MimoSimulator
from .reactor import ReactorSimulator

_REGISTRY: dict[str, type[Simulator]] = {
    ReactorSimulator.name: ReactorSimulator,
    HeatSimulator.name: HeatSimulator,
    MimoSimulator.name: MimoSimulator,
}


def register(cls: type[Simulator]) -> type[Simulator]:
    """Class decorator: make a simulator available by name to studies, the CLI, the API and MCP."""
    _REGISTRY[cls.name] = cls
    return cls


def get_simulator(name: str) -> Simulator:
    if name not in _REGISTRY and ":" in name:  # "my_pkg.my_module:MySim" plugin form
        mod, attr = name.split(":", 1)
        register(getattr(importlib.import_module(mod), attr))
        name = getattr(importlib.import_module(mod), attr).name
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown simulator {name!r}; registered: {sorted(_REGISTRY)}") from None


def list_simulators() -> list[str]:
    return sorted(_REGISTRY)


@lru_cache(maxsize=1)
def references() -> dict:
    """Best-known optima (computed by scripts/compute_references.py) used for regret scoring."""
    path = Path(__file__).with_name("references.json")
    return json.loads(path.read_text()) if path.exists() else {}


__all__ = ["Fidelity", "Param", "Simulator", "SimResult", "get_simulator", "list_simulators", "register", "references"]
