"""Template: wrap your lab's simulation code as a SciAgent simulator.

Copy this file into your own package, fill in the three TODOs, then reference it from a study spec as
    simulator: "my_lab.my_sim:MySimulator"
(no changes to sciagent itself are needed), or call ``sciagent.simulators.register(MySimulator)``.
"""

from __future__ import annotations

import numpy as np

from sciagent.simulators import Fidelity, Param, Simulator


class MySimulator(Simulator):
    name = "my_sim"            # used in specs, traces and the cache key
    version = "0.1"            # bump when results change -> invalidates cached runs
    stochastic = False         # True if the seed changes the result (Monte-Carlo, stochastic solvers...)
    description = "TODO: one paragraph a planner (human or LLM) needs: what is simulated, what is good."
    objective_name = "TODO_objective"   # always maximised; negate quantities you want to minimise

    # TODO 1: the knobs a researcher can turn, with physical units
    params = (
        Param("x", 0.0, 1.0, unit="", description="TODO"),
        Param("n_layers", 1, 8, kind="int", description="TODO"),
    )

    # TODO 2: accuracy levels, cheapest first, with a declared cost (e.g. core-minutes) and honest notes
    fidelities = (
        Fidelity("coarse", 0, 1.0, "fast; known to over-estimate X by ~5%"),
        Fidelity("fine", 1, 10.0, "production settings"),
    )

    def _evaluate(self, p: dict, fidelity: Fidelity, rng: np.random.Generator) -> tuple[float, dict]:
        # TODO 3: call your code (a Python function, a subprocess, a container, a job on a cluster).
        # Return (objective, metrics). Raise on failure: the run is recorded as failed, budget is charged,
        # and the campaign continues.
        resolution = 16 if fidelity.name == "coarse" else 128
        value = -((p["x"] - 0.3) ** 2) - 0.01 * p["n_layers"] + rng.normal(0, 1 / resolution) * self.stochastic
        return float(value), {"resolution": resolution}

    # Optional: a noise-free ground truth, only for benchmarking planners (never shown to them).
    # def true_objective(self, params): ...
