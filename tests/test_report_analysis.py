import json

from sciagent.analysis import analyze, bootstrap_ci, holm, load
from sciagent.bench import run_bench
from sciagent.experiments import run as run_experiments
from sciagent.report import build


def test_holm_and_bootstrap():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06
    import numpy as np

    lo, hi = bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0]))
    assert 1.0 <= lo < 2.5 < hi <= 4.0


def test_analysis_detects_a_clear_winner(tmp_path):
    rows = [{"task": "reactor", "planner": p, "seed": s, "budget": 100.0, "regret": (0.001 if p == "good" else 0.1) + s * 1e-4}
            for p in ("good", "bad") for s in range(8)]
    a = analyze(rows, {})
    r = a["tasks"]["reactor"]
    assert r["best"] == "good" and r["holm_vs_best"]["bad"] < 0.05 and r["win_rate"]["good>bad"] == 1.0


def test_mixed_budgets_are_not_pooled():
    rows = [{"task": "heat", "planner": "sh", "seed": s, "budget": b, "regret": 0.1} for s in range(3) for b in (60.0, 120.0)]
    a = analyze(rows, {})
    assert set(a["tasks"]["heat"]["stats"]) == {"sh@60", "sh@120"}


def test_report_from_pipeline(tmp_path):
    R = tmp_path / "results"
    status = run_experiments("cpu-smoke", R, ["env", "bench_cpu"])
    assert all(v.startswith("ok") for v in status.values())
    text = build([R], tmp_path / "REPORT.md")
    assert "## Summary" in text and "Classical planners, budget 60" in text and "| bench_cpu | results |" in text
    assert "| bench_llm | not run |" in text
    # resumable: a second call skips finished steps
    assert run_experiments("cpu-smoke", R, ["bench_cpu"]) == {"bench_cpu": "skipped"}
    rows, curves = load([R / "bench_cpu" / "b60"])
    assert rows and json.loads((R / "bench_cpu" / "b60" / "analysis" / "analysis.json").read_text())["budget"] == 60


def test_bench_rows_carry_digests(tmp_path):
    rows = run_bench(["reactor"], {"sh": {"kind": "sh"}}, [0], 40, 4, tmp_path, verbose=False)
    assert rows[0]["results_digest"]
