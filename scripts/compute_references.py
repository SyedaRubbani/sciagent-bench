"""Compute best-known optima of the built-in simulators for regret scoring.

reactor: exact (closed form + multi-start L-BFGS-B)
heat:    differential evolution on the 64x64 solver ("best known", not a proof of global optimality)
mimo:    exhaustive over n_tx x bounded scalar search over power on the 40k-draw estimator

Run:  python scripts/compute_references.py   (about 3 minutes on a laptop CPU)
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

from sciagent.simulators import get_simulator

out = {}

s = get_simulator("reactor")
best = None
rng = np.random.default_rng(0)
for _ in range(300):
    x0 = [p.sample(rng) for p in s.params]
    r = minimize(lambda x: -s.true_objective(dict(zip(["T", "tau", "cat"], x))), x0,
                 bounds=[(p.low, p.high) for p in s.params], method="L-BFGS-B")
    if best is None or r.fun < best.fun:
        best = r
out["reactor"] = {"optimum": -float(best.fun), "argmax": dict(zip(["T", "tau", "cat"], map(float, best.x))),
                  "method": "closed form + 300-start L-BFGS-B", "exact": True}

s = get_simulator("heat")
r = differential_evolution(lambda x: -s.true_objective(dict(pin_x=x[0], pin_y=x[1], pin_r=x[2])),
                           [(p.low, p.high) for p in s.params], seed=0, maxiter=80, popsize=15, tol=1e-9)
out["heat"] = {"optimum": -float(r.fun), "argmax": dict(zip(["pin_x", "pin_y", "pin_r"], map(float, r.x))),
               "method": "differential evolution on grid64", "exact": False}

s = get_simulator("mimo")
cands = []
for k in range(1, 9):
    r = minimize_scalar(lambda p, k=k: -s.true_objective(dict(p_dbm=p, n_tx=k)), bounds=(0, 30), method="bounded",
                        options={"xatol": 1e-3})
    cands.append((-r.fun, k, r.x))
v, k, p = max(cands)
out["mimo"] = {"optimum": float(v), "argmax": {"p_dbm": float(p), "n_tx": int(k)},
               "method": "exhaustive n_tx x bounded power search, 40k MC draws", "exact": False}

path = Path(__file__).resolve().parents[1] / "src/sciagent/simulators/references.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
