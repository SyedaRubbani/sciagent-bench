"""Statistical analysis of benchmark results.

    sciagent analyze bench/results_b300 [bench/results_llm ...] --out bench/analysis_b300

For every task and planner: mean, median, IQR and a seeded bootstrap 95% CI of the mean regret, the
success rate (regret below a task tolerance), and the normalised area under the log-regret curve (anytime
performance). Between planners: paired Wilcoxon signed-rank tests against the best planner of each task
(paired by seed, Holm-corrected), and a win-rate matrix.
"""

from __future__ import annotations

import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

# "solved" if regret is below this (task objective units): ~0.5% of each task's optimum scale
TOLERANCE = {"reactor": 0.005, "heat": 0.005, "mimo": 0.05}


def load(dirs: list[str | Path]) -> tuple[list[dict], dict]:
    rows, curves = [], {}
    for d in dirs:
        d = Path(d)
        with open(d / "results.csv") as fh:
            for r in csv.DictReader(fh):
                r["seed"] = int(r["seed"])
                r["budget"] = float(r["budget"])
                r["regret"] = float(r["regret"]) if r["regret"] not in ("", "None") else None
                rows.append(r)
        if (d / "curves.json").exists():
            for task, by_p in json.loads((d / "curves.json").read_text()).items():
                for p, cs in by_p.items():
                    curves.setdefault(task, {}).setdefault(p, []).extend(cs)
    return rows, curves


def bootstrap_ci(x: np.ndarray, n: int = 10_000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def auc_log_regret(curves: list[list[dict]], budget: float, floor: float = 1e-6) -> float | None:
    """Mean over seeds of the average log10 regret over the budget (lower = better anytime performance).
    Before the first incumbent exists the regret is taken as the worst value observed in that run."""
    if not curves:
        return None
    grid = np.linspace(0, budget, 201)[1:]
    vals = []
    for c in curves:
        if not c:
            continue
        xs = np.array([p["spent"] for p in c])
        ys = np.maximum(np.array([p["regret"] for p in c], dtype=float), floor)
        idx = np.searchsorted(xs, grid, side="right") - 1
        y = np.where(idx >= 0, ys[np.clip(idx, 0, None)], ys.max())
        vals.append(float(np.mean(np.log10(y))))
    return float(np.mean(vals)) if vals else None


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def analyze(rows: list[dict], curves: dict) -> dict:
    from scipy.stats import wilcoxon

    budgets = sorted({r["budget"] for r in rows})
    if len(budgets) > 1:  # never pool different budgets: label planners by budget instead
        rows = [{**r, "planner": f"{r['planner']}@{r['budget']:g}"} for r in rows]
        curves = {}
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    planners = list(dict.fromkeys(r["planner"] for r in rows))
    budget = budgets[0] if len(budgets) == 1 else None
    res: dict = {"budget": budget, "budgets": budgets, "tasks": {}, "planners": planners, "tolerance": TOLERANCE}
    for t in tasks:
        by_p = {p: {r["seed"]: r["regret"] for r in rows if r["task"] == t and r["planner"] == p and r["regret"] is not None}
                for p in planners}
        failed = {p: sum(1 for r in rows if r["task"] == t and r["planner"] == p and r["regret"] is None)
                  for p in planners}
        by_p = {p: v for p, v in by_p.items() if v}
        stats = {}
        for p, d in by_p.items():
            x = np.array(list(d.values()))
            lo, hi = bootstrap_ci(x)
            stats[p] = {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
                        "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75)), "ci95": [lo, hi],
                        "success_rate": float(np.mean(x < TOLERANCE.get(t, 0.0))),
                        "auc_log10_regret": auc_log_regret(curves.get(t, {}).get(p, []), budget) if budget else None,
                        "n_no_result": failed.get(p, 0)}
        best = min(stats, key=lambda p: stats[p]["mean"])
        raw = {}
        for p in by_p:
            if p == best:
                continue
            seeds = sorted(set(by_p[p]) & set(by_p[best]))
            a = np.array([by_p[best][s] for s in seeds])
            b = np.array([by_p[p][s] for s in seeds])
            if len(seeds) >= 5 and np.any(a != b):
                raw[p] = float(wilcoxon(a, b, zero_method="zsplit").pvalue)
            else:
                raw[p] = float("nan")
        adj = holm({k: v for k, v in raw.items() if not math.isnan(v)})
        win = {}
        for p, q in combinations(by_p, 2):
            seeds = sorted(set(by_p[p]) & set(by_p[q]))
            if seeds:
                w = np.mean([by_p[p][s] < by_p[q][s] for s in seeds]) + 0.5 * np.mean([by_p[p][s] == by_p[q][s] for s in seeds])
                win[f"{p}>{q}"] = float(w)
                win[f"{q}>{p}"] = float(1 - w)
        no_result = {p: k for p, k in failed.items() if k and p not in stats}
        res["tasks"][t] = {"stats": stats, "best": best, "wilcoxon_vs_best": raw, "holm_vs_best": adj, "win_rate": win,
                           "no_result": no_result}
    return res


def _f(x, d=4):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}g}"


def to_markdown(a: dict) -> str:
    btxt = f"a budget of {a['budget']:g}" if a.get("budget") else "budgets " + ", ".join(f"{b:g}" for b in a["budgets"])
    L = [f"# Planner comparison at {btxt} compute units", "",
         "Regret of the reported incumbent (task objective units; lower is better). CI = seeded bootstrap 95% "
         "interval of the mean. Solved = regret below the task tolerance "
         f"({', '.join(f'{k} {v:g}' for k, v in a['tolerance'].items())}). AUC = mean log10 regret over the budget "
         "(anytime performance; lower is better). p = paired Wilcoxon signed-rank test against the task's best "
         "planner over seeds, Holm-corrected.", ""]
    for t, r in a["tasks"].items():
        L += [f"## {t}", "", "| planner | mean | 95% CI | median [IQR] | solved | AUC | p vs best (Holm) |",
              "|---|---|---|---|---|---|---|"]
        for p, s in sorted(r["stats"].items(), key=lambda kv: kv[1]["mean"]):
            pv = "best" if p == r["best"] else _f(r["holm_vs_best"].get(p), 2)
            L.append(f"| {p} | {_f(s['mean'])} | [{_f(s['ci95'][0])}, {_f(s['ci95'][1])}] | {_f(s['median'])} "
                     f"[{_f(s['q25'])}, {_f(s['q75'])}] | {s['success_rate']:.0%} ({s['n']}) | {_f(s['auc_log10_regret'], 3)} | {pv} |")
        for p, k in r.get("no_result", {}).items():
            L.append(f"| {p} | no valid result in {k} studies | | | | | |")
        L.append("")
    return "\n".join(L)


def run(dirs: list[str], out: str | Path) -> dict:
    rows, curves = load(dirs)
    a = analyze(rows, curves)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(a, indent=2))
    (out / "analysis.md").write_text(to_markdown(a))
    return a
