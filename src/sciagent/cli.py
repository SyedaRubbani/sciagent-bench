"""Command-line interface:  sciagent <command> ...   (run `sciagent -h`)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _cmd_sims(a):
    from .simulators import get_simulator, list_simulators

    for name in list_simulators():
        card = get_simulator(name).card()
        if a.json:
            print(json.dumps(card, indent=2))
            continue
        print(f"{name} v{card['version']}: {card['description']}")
        for p in card["params"]:
            print(f"    param  {p['name']:8s} [{p['low']:g}, {p['high']:g}] {p['kind']} {p['unit']}")
        for f in card["fidelities"]:
            print(f"    fid    {f['name']:13s} cost {f['cost']:5g}  {f['notes']}")


def _default_out(spec_name: str) -> Path:
    return Path("runs") / f"{spec_name}-{time.strftime('%Y%m%d-%H%M%S')}"


def _cmd_run(a):
    from .orchestrator import Orchestrator
    from .spec import StudySpec

    spec = StudySpec.from_yaml(a.spec)
    if a.planner:
        spec.planner.kind = a.planner
        if a.planner == "llm" and spec.planner.llm is None:
            from .spec import LLMSpec

            spec.planner.llm = LLMSpec()
    if a.budget:
        spec.budget.compute_units = a.budget
    if a.seed is not None:
        spec.seed = a.seed
    if a.record and spec.planner.llm:
        spec.planner.llm.mode = "record"
    out = Path(a.out) if a.out else _default_out(spec.name)
    print(f"run dir: {out}")
    s = Orchestrator(spec, out, verbose=True).run()
    _print_summary(s)


def _print_summary(s: dict):
    inc = s.get("incumbent") or {}
    print("\n=== summary ===")
    print(f"stop reason     : {s['stop_reason']}")
    print(f"runs            : {s['n_runs']}  per fidelity {s['runs_per_fidelity']}  cached {s['n_cached']}")
    print(f"compute spent   : {s['spent_compute']:g} / {s['compute_budget']:g}")
    if s.get("llm_tokens_in"):
        print(f"LLM             : {s['llm_tokens_in']} in / {s['llm_tokens_out']} out tokens, ${s['llm_usd']:.4f}")
    print(f"incumbent       : {inc.get('params')} @ {inc.get('fidelity')} -> {inc.get('objective')}")
    if s.get("regret") is not None:
        print(f"true objective  : {s['incumbent_true_objective']:.6g}   reference {s['reference_optimum']:.6g}"
              f"   regret {s['regret']:.4g}")
    print(f"results digest  : {s['results_digest']}")


def _cmd_replay(a):
    """Re-execute a finished run from its saved spec (and LLM cassette) and check the results match."""
    from .orchestrator import Orchestrator
    from .spec import StudySpec

    src = Path(a.run_dir)
    spec = StudySpec.from_yaml(src / "spec.yaml")
    original = json.loads((src / "summary.json").read_text())
    if spec.planner.kind == "llm":
        cas = spec.planner.llm.cassette or str(src / "llm_cassette.jsonl")
        spec.planner.llm.mode, spec.planner.llm.cassette = "replay", cas
    spec.cache_dir = None
    spec.budget.wall_clock_s = None
    out = Path(a.out) if a.out else src / f"replay-{time.strftime('%Y%m%d-%H%M%S')}"
    s = Orchestrator(spec, out).run()
    same = s["results_digest"] == original["results_digest"]
    print(f"original digest {original['results_digest']}  replay digest {s['results_digest']}  ->",
          "IDENTICAL" if same else "DIFFERENT")
    sys.exit(0 if same else 1)


def _cmd_bench(a):
    from .bench import load_planners, run_bench

    planners = load_planners(a.planners)
    if a.only:
        planners = {k: v for k, v in planners.items() if k in a.only}
    run_bench(a.tasks, planners, list(range(a.seed_start, a.seed_start + a.seeds)), a.budget, a.batch_size, a.out, keep_runs=a.keep_runs)
    print((Path(a.out) / "results.md").read_text())


def _cmd_steer(a):
    from .steering import steer

    bounds = {}
    for b in a.bound or []:
        name, rng = b.split("=")
        lo, hi = rng.split(",")
        bounds[name] = (float(lo), float(hi))
    pause = True if a.pause else (False if a.resume else None)
    print(json.dumps(steer(a.run_dir, pause=pause, stop=True if a.stop else None, note=a.note, bounds=bounds or None,
                           compute_units=a.budget)))


def _cmd_trace(a):
    for line in (Path(a.run_dir) / "trace.jsonl").read_text().splitlines():
        ev = json.loads(line)
        k, d = ev["kind"], ev["data"]
        if k == "plan":
            print(f"{ev['span_id']} PLAN   [{d['planner']}] {d['n_proposed']} runs: {d['rationale'][:100]}")
        elif k == "guard":
            for i in d["issues"]:
                print(f"{ev['span_id']} GUARD  run {i['run']} {i['action']}: {i['reason']}")
        elif k == "run":
            o = d["objective"]
            print(f"{ev['span_id']} RUN    #{d['index']:<4d} {d['fidelity']:13s} "
                  f"{json.dumps({x: round(v, 4) for x, v in d['params'].items()})} -> {o if o is None else round(o, 5)}"
                  f"{' (cached)' if d['cached'] else ''}")
        elif k == "llm.call":
            print(f"{ev['span_id']} LLM    {d['model']} attempt {d['attempt']} {d['tokens_in']}/{d['tokens_out']} tok"
                  f"{' (replayed)' if d.get('replayed') else ''}")
        elif k in ("steer", "pause", "error", "study.start", "study.end"):
            print(f"{ev['span_id']} {k.upper():6s} {json.dumps(d)[:160]}")


def _cmd_serve(a):
    import uvicorn

    from .service import create_app

    uvicorn.run(create_app(a.runs_dir), host=a.host, port=a.port)


def _cmd_mcp(a):
    from .mcp_server import main as mcp_main

    mcp_main(a.runs_dir)


def _cmd_surrogate(a):
    from .surrogate import pipeline

    pipeline.cli(a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sciagent", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sims", help="list simulators")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_sims)

    p = sub.add_parser("run", help="run a study from a YAML spec")
    p.add_argument("spec")
    p.add_argument("--out")
    p.add_argument("--planner", choices=["random", "sh", "bo", "llm"])
    p.add_argument("--budget", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--record", action="store_true", help="record LLM calls to a cassette for replay")
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("replay", help="re-run a finished study and verify identical results")
    p.add_argument("run_dir")
    p.add_argument("--out")
    p.set_defaults(fn=_cmd_replay)

    p = sub.add_parser("bench", help="benchmark planners across tasks and seeds")
    p.add_argument("--tasks", nargs="+", default=["reactor", "heat", "mimo"])
    p.add_argument("--planners", help="YAML file with a 'planners:' mapping (see bench/planners.example.yaml)")
    p.add_argument("--only", nargs="+", help="subset of planner names")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0, help="first seed (use held-out seeds for distilled models)")
    p.add_argument("--budget", type=float, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out", default="bench_results")
    p.add_argument("--keep-runs", action="store_true")
    p.set_defaults(fn=_cmd_bench)

    p = sub.add_parser("steer", help="steer a running study")
    p.add_argument("run_dir")
    p.add_argument("--pause", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop", action="store_true")
    p.add_argument("--note")
    p.add_argument("--bound", action="append", help="name=lo,hi (repeatable)")
    p.add_argument("--budget", type=float, help="new total compute budget")
    p.set_defaults(fn=_cmd_steer)

    p = sub.add_parser("trace", help="print a readable timeline of a run")
    p.add_argument("run_dir")
    p.set_defaults(fn=_cmd_trace)

    p = sub.add_parser("serve", help="HTTP API + trace viewer")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=_cmd_serve)

    p = sub.add_parser("mcp", help="MCP server (stdio) exposing simulators and studies as tools")
    p.add_argument("--runs-dir", default="runs")
    p.set_defaults(fn=_cmd_mcp)

    p = sub.add_parser("surrogate", help="curate data, train / distil / quantise a surrogate")
    p.add_argument("action", choices=["make-data", "train", "report"])
    p.add_argument("--sim", default="reactor")
    p.add_argument("--n", type=int, default=6000, help="samples per fidelity for make-data")
    p.add_argument("--runs", nargs="*", default=[], help="run dirs whose runs.jsonl to include in the dataset")
    p.add_argument("--data-dir", default="surrogate_data")
    p.add_argument("--model-dir", default="surrogate_models")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_cmd_surrogate)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
