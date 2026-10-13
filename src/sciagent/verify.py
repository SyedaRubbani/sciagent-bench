"""Verification of the benchmark's reference solutions, with uncertainty.

    sciagent verify-references [--device cuda] [--mimo-draws 20000000] [--heat-max-n 1024]

Writes ``references_verification.json`` and ``.md`` with:

* reactor  - stationarity (gradient ~ 0) and negative-definite Hessian at the closed-form optimum
* heat     - optimiser robustness across independent global searches, and a grid-convergence study at the
             optimum (observed order, Richardson-extrapolated continuum value, discretisation error of the
             grid64 task definition)
* mimo     - Monte-Carlo standard error of the reference value with many draws, and a batch-means test
             that the best antenna count beats the runner-up by more than the noise
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .simulators import get_simulator, references


def verify_reactor() -> dict:
    sim = get_simulator("reactor")
    ref = references()["reactor"]
    x0 = np.array([ref["argmax"][k] for k in ("T", "tau", "cat")])
    scale = np.array([120.0, 10.0, 1.0])

    def f(u):
        x = x0 + u * scale
        return sim.true_objective(dict(T=x[0], tau=x[1], cat=x[2]))

    h = 1e-4
    g = np.array([(f(np.eye(3)[i] * h) - f(-np.eye(3)[i] * h)) / (2 * h) for i in range(3)])
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * h, np.eye(3)[j] * h
            H[i, j] = (f(ei + ej) - f(ei - ej) - f(-ei + ej) + f(-ei - ej)) / (4 * h * h)
    eig = np.linalg.eigvalsh((H + H.T) / 2)
    return {"optimum": ref["optimum"], "argmax": ref["argmax"], "grad_norm_scaled": float(np.linalg.norm(g)),
            "hessian_eigenvalues_scaled": eig.tolist(), "is_strict_local_max": bool(np.all(eig < 0)),
            "exact": True, "note": "closed-form objective; interior stationary point with negative-definite Hessian"}


def verify_heat(device: str, max_n: int, n_searches: int = 5) -> dict:
    from scipy.optimize import differential_evolution, minimize

    from .gpu.heat_torch import heat_objective_torch

    sim = get_simulator("heat")
    bounds = [(p.low, p.high) for p in sim.params]
    f = lambda x: -sim.true_objective(dict(pin_x=x[0], pin_y=x[1], pin_r=x[2]))  # noqa: E731
    searches = []
    for s in range(n_searches):
        t0 = time.perf_counter()
        r = differential_evolution(f, bounds, seed=100 + s, maxiter=60, popsize=12, tol=1e-9, polish=False)
        r2 = minimize(f, r.x, method="Nelder-Mead", bounds=bounds, options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 400})
        best = r2 if r2.fun < r.fun else r
        searches.append({"seed": 100 + s, "value": -float(best.fun), "argmax": best.x.tolist(),
                         "seconds": time.perf_counter() - t0})
    vals = np.array([s["value"] for s in searches])
    ref = references()["heat"]
    best_found = float(vals.max())
    xb = ref["argmax"]
    ns = [n for n in (32, 64, 128, 256, 512, 1024, 2048) if n <= max_n]
    conv = []
    for n in ns:
        t0 = time.perf_counter()
        obj, info = heat_objective_torch(xb["pin_x"], xb["pin_y"], xb["pin_r"], n, device=device)
        conv.append({"n": n, "objective": obj, "cg_iterations": info["iterations"], "rel_residual": info["rel_residual"],
                     "seconds": time.perf_counter() - t0})
    out = {"reference": ref["optimum"], "best_of_searches": best_found,
           "search_values": vals.tolist(), "search_spread": float(vals.max() - vals.min()),
           "reference_minus_best_search": ref["optimum"] - best_found, "searches": searches, "convergence": conv,
           "device": device}
    # observed order from the three finest grids (note: h = 1/(n+1), close to halving)
    if len(conv) >= 3:
        f1, f2, f3 = (c["objective"] for c in conv[-3:])
        d1, d2 = f1 - f2, f2 - f3
        if d1 * d2 > 0 and abs(d2) > 0:
            p = math.log(abs(d1 / d2), 2)
            f_inf = f3 + d2 / (2**p - 1)
            g64 = next(c["objective"] for c in conv if c["n"] == 64)
            out.update(observed_order=p, richardson_continuum=f_inf, grid64_discretisation_error=g64 - f_inf)
    out["note"] = ("the benchmark task is *defined* by the grid64 solver; the convergence study quantifies how far "
                   "that definition is from the continuum problem. The pin's diffuse-interface width is max(h, 1/128), so "
                   "grids below n = 127 also solve a slightly smoother problem; the observed order uses the finest grids, "
                   "where the width is fixed")
    return out


def verify_mimo(device: str, draws: int) -> dict:
    from .gpu.mimo_torch import energy_efficiency_torch, ergodic_capacity_torch
    from .simulators.mimo import BANDWIDTH_MHZ, GAIN_OVER_NOISE, _power

    sim = get_simulator("mimo")
    ref = references()["mimo"]
    p6, k6 = ref["argmax"]["p_dbm"], ref["argmax"]["n_tx"]
    t0 = time.perf_counter()
    ee, se = energy_efficiency_torch(p6, k6, draws, seed=7, device=device)
    secs = time.perf_counter() - t0
    # runner-up: best power for each other antenna count (coarse search on the 40k estimator, then CRN check)
    rivals = []
    for k in range(1, 9):
        if k == k6:
            continue
        ps = np.linspace(0, 30, 121)
        vals = [sim.true_objective(dict(p_dbm=float(p), n_tx=k)) for p in ps]
        rivals.append((max(vals), k, float(ps[int(np.argmax(vals))])))
    rivals.sort(reverse=True)
    v2, k2, p2 = rivals[0]
    # gap to the runner-up, with a batch-means standard error over 20 independent batches
    import torch

    n = min(draws, 2_000_000)
    g = torch.Generator(device=device).manual_seed(11)
    diffs = []
    chunk = max(n // 20, 1000)
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g, device=device))
        a, _ = ergodic_capacity_torch(10 ** (p6 / 10) / 1000 * GAIN_OVER_NOISE, k6, m, seed, device)
        b, _ = ergodic_capacity_torch(10 ** (p2 / 10) / 1000 * GAIN_OVER_NOISE, k2, m, seed, device)
        diffs.append((BANDWIDTH_MHZ * a / _power(p6, k6) - BANDWIDTH_MHZ * b / _power(p2, k2), m))
    d = np.array([x for x, _ in diffs])
    w = np.array([m for _, m in diffs], dtype=float)
    mean_d = float(np.sum(d * w) / w.sum())
    se_d = float(np.sqrt(np.sum(w * (d - mean_d) ** 2) / w.sum() / max(len(d) - 1, 1)))
    return {"reference_40k_estimator": ref["optimum"], "argmax": ref["argmax"], "high_precision_value": ee,
            "high_precision_stderr": se, "ci95": [ee - 1.96 * se, ee + 1.96 * se], "draws": draws, "seconds": secs,
            "estimator_bias_40k_vs_high_precision": ref["optimum"] - ee,
            "runner_up": {"n_tx": k2, "p_dbm": p2, "value_40k": v2},
            "gap_to_runner_up": mean_d, "gap_stderr_batch_means": se_d,
            "gap_z": mean_d / se_d if se_d > 0 else float("inf"), "device": device,
            "note": "scoring uses the fixed-seed 40k-draw estimator as the task's ground truth; "
                    "this records how close that estimator is to the infinite-draw value"}


def run(device: str = "auto", mimo_draws: int = 2_000_000, heat_max_n: int = 512, out: str | Path = ".",
        heat_searches: int = 5) -> dict:
    from .gpu import device_info, pick_device

    dev = pick_device(device)
    res = {"device": device_info(dev), "reactor": verify_reactor(), "heat": verify_heat(dev, heat_max_n, heat_searches),
           "mimo": verify_mimo(dev, mimo_draws)}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "references_verification.json").write_text(json.dumps(res, indent=2))
    (out / "references_verification.md").write_text(to_markdown(res))
    return res


def to_markdown(r: dict) -> str:
    re_, he, mi = r["reactor"], r["heat"], r["mimo"]
    L = ["# Reference verification", "", f"Device: `{r['device'].get('name', r['device']['device'])}`", "",
         "## reactor (exact)", "",
         f"Optimum {re_['optimum']:.10f} at {json.dumps({k: round(v, 5) for k, v in re_['argmax'].items()})}. "
         f"Scaled gradient norm {re_['grad_norm_scaled']:.2e}; Hessian eigenvalues "
         f"{', '.join(f'{e:.3g}' for e in re_['hessian_eigenvalues_scaled'])} "
         f"({'strict local maximum' if re_['is_strict_local_max'] else 'NOT a strict maximum'}).", "",
         "## heat (best known, grid64 task definition)", "",
         f"Reference {he['reference']:.8f}. {len(he['search_values'])} independent global searches found "
         f"{', '.join(f'{v:.8f}' for v in he['search_values'])} (spread {he['search_spread']:.2e}; "
         f"reference minus best search {he['reference_minus_best_search']:+.2e}).", "",
         "| grid n | objective at reference design | CG iterations | rel. residual | seconds |", "|---|---|---|---|---|"]
    L += [f"| {c['n']} | {c['objective']:.8f} | {c['cg_iterations']} | {c['rel_residual']:.1e} | {c['seconds']:.2f} |"
          for c in he["convergence"]]
    if "observed_order" in he:
        L += ["", f"Observed order {he['observed_order']:.2f}; Richardson continuum estimate "
              f"{he['richardson_continuum']:.8f}; discretisation error of the grid64 definition "
              f"{he['grid64_discretisation_error']:+.2e}."]
    L += ["", "## mimo (40k-draw estimator as task definition)", "",
          f"High-precision value at the reference design: {mi['high_precision_value']:.6f} ± {mi['high_precision_stderr']:.1e} "
          f"(SE, {mi['draws']:,} draws, {mi['seconds']:.1f} s). The 40k estimator differs by "
          f"{mi['estimator_bias_40k_vs_high_precision']:+.2e}.", "",
          f"Runner-up design n_tx={mi['runner_up']['n_tx']} at {mi['runner_up']['p_dbm']:.2f} dBm. Gap "
          f"{mi['gap_to_runner_up']:.4f} ± {mi['gap_stderr_batch_means']:.4f} (batch-means SE over 20 independent batches; z = {mi['gap_z']:.1f}).", ""]
    return "\n".join(L) + "\n"
