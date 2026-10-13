"""Benchmark harness: planners x tasks x seeds at a fixed compute budget.

Scoring uses each simulator's noise-free, top-accuracy ``true_objective`` of the *reported* incumbent and
the best-known optimum in ``references.json``:  regret = f* - f_true(incumbent)  (lower is better).
LLM planners are configured with a YAML block per model so open and closed models are compared on the
same tasks, budgets and seeds.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
import warnings
from pathlib import Path

import yaml

from .orchestrator import Orchestrator
from .spec import LLMSpec, PlannerSpec, StudySpec

DEFAULT_PLANNERS = {
    "random": {"kind": "random"},
    "sh": {"kind": "sh"},
    "bo": {"kind": "bo"},
    # the full LLM-agent code path driven by the deterministic MockProvider (a scripted multi-fidelity
    # heuristic, not a language model): a pipeline check and a hand-written baseline
    "scripted-mf": {"kind": "llm", "llm": {"provider": "mock"}},
}


def load_planners(path: str | None) -> dict[str, dict]:
    if not path:
        return dict(DEFAULT_PLANNERS)
    return yaml.safe_load(Path(path).read_text())["planners"]


def run_bench(tasks: list[str], planners: dict[str, dict], seeds: list[int], budget: float, batch_size: int,
              out_dir: str | Path, keep_runs: bool = False, verbose: bool = True) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    curves: dict = {}
    for task in tasks:
        for pname, pcfg in planners.items():
            for seed in seeds:
                cfg = dict(pcfg)
                if cfg.get("llm"):
                    cfg["llm"] = LLMSpec.model_validate(cfg["llm"])
                spec = StudySpec(name=f"{task}-{pname}-s{seed}", simulator=task, budget={"compute_units": budget},
                                 batch_size=batch_size, seed=seed, planner=PlannerSpec.model_validate(cfg))
                run_dir = out / "runs" / spec.name if keep_runs else None
                t0 = time.perf_counter()
                try:
                    s = Orchestrator(spec, run_dir).run()
                    err = None
                except Exception as e:  # keep the sweep going; record the failure
                    s, err = {}, f"{type(e).__name__}: {e}"
                row = {"task": task, "planner": pname, "seed": seed, "budget": budget,
                       "regret": s.get("regret"), "incumbent_true": s.get("incumbent_true_objective"),
                       "n_runs": s.get("n_runs"), "spent": s.get("spent_compute"),
                       "runs_per_fidelity": json.dumps(s.get("runs_per_fidelity", {})),
                       "llm_tokens_in": s.get("llm_tokens_in"), "llm_tokens_out": s.get("llm_tokens_out"),
                       "llm_usd": s.get("llm_usd"), "wall_s": round(time.perf_counter() - t0, 3),
                       "stop_reason": s.get("stop_reason"), "error": err}
                rows.append(row)
                curves.setdefault(task, {}).setdefault(pname, []).append(s.get("curve", []))
                if verbose:
                    rg = row["regret"]
                    print(f"{task:8s} {pname:14s} seed {seed}: regret {rg if rg is None else f'{rg:.5f}'}"
                          f"  runs {row['n_runs']}  {row['wall_s']}s" + (f"  ERROR {err}" if err else ""))
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "curves.json").write_text(json.dumps(curves))
    (out / "results.md").write_text(markdown_table(rows, budget, len(seeds)))
    try:
        plot_curves(curves, out / "regret_curves.png", budget)
    except ImportError:
        pass
    return rows


def aggregate(rows: list[dict]) -> dict:
    agg: dict = {}
    for r in rows:
        if r["regret"] is None:
            continue
        agg.setdefault(r["task"], {}).setdefault(r["planner"], []).append(r["regret"])
    return agg


def markdown_table(rows: list[dict], budget: float, n_seeds: int) -> str:
    agg = aggregate(rows)
    tasks = list(agg)
    planners = list(dict.fromkeys(r["planner"] for r in rows))
    lines = [f"Simple regret of the reported incumbent at a budget of {budget:g} compute units "
             f"(mean ± sd over {n_seeds} seeds; lower is better; **bold** = best mean per task).", "",
             "| planner | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    best = {t: min(statistics.mean(v) for v in agg[t].values()) for t in tasks}
    for p in planners:
        cells = []
        for t in tasks:
            v = agg[t].get(p)
            if not v:
                cells.append("–")
                continue
            m = statistics.mean(v)
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            cell = f"{m:.4f} ± {sd:.4f}"
            cells.append(f"**{cell}**" if m == best[t] else cell)
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    usd = [r for r in rows if r.get("llm_usd")]
    if usd:
        lines += ["", "LLM planner cost per study (USD, mean): " + ", ".join(
            f"{p}: {statistics.mean(r['llm_usd'] for r in usd if r['planner'] == p):.4f}"
            for p in dict.fromkeys(r['planner'] for r in usd))]
    return "\n".join(lines) + "\n"


def plot_curves(curves: dict, path: Path, budget: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = list(curves)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.6), squeeze=False)
    grid = np.linspace(0, budget, 101)
    for ax, task in zip(axes[0], tasks):
        for pname, runs in curves[task].items():
            mat = []
            for c in runs:
                if not c:
                    continue
                xs = np.array([p["spent"] for p in c])
                ys = np.maximum(np.array([p["regret"] for p in c], dtype=float), 1e-6)
                idx = np.searchsorted(xs, grid, side="right") - 1
                mat.append(np.where(idx >= 0, ys[np.clip(idx, 0, None)], np.nan))
            if not mat:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # NaN before the first incumbent exists
                m = np.nanmedian(np.array(mat), axis=0)
            ax.plot(grid, m, label=pname, lw=1.8)
        ax.set_yscale("log")
        ax.set_title(task)
        ax.set_xlabel("compute units spent")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("regret of incumbent (median)")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
