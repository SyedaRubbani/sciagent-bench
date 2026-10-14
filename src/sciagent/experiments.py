"""The full experimental pipeline, as resumable steps (run on Kaggle/Colab GPUs or, scaled down, on a CPU).

    sciagent experiments --profile kaggle-t4 --results /kaggle/working/results            # everything
    sciagent experiments --profile kaggle-t4 --steps verify gpu_bench                     # some steps
    sciagent experiments --profile cpu-smoke --results /tmp/smoke                         # plumbing check

Each step writes its outputs under ``<results>/<step>/`` and a ``DONE.json``; finished steps are skipped on
re-runs (use ``--force``), so an interrupted GPU session can resume where it stopped.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

RECIPES = Path(__file__).resolve().parents[2] / "recipes" / "distill_planner_lora"

PROFILES: dict[str, dict] = {
    "kaggle-t4": {
        "device": "cuda",
        "verify": {"mimo_draws": 20_000_000, "heat_max_n": 2048, "heat_searches": 5},
        "gpu_bench": {"quick": False},
        "bench_cpu": {"seeds": 20, "budgets": [120, 300], "parallel": 4},
        "surrogate": {"sims": ["reactor", "mimo"], "n": 4096, "epochs": 300},
        "teacher": {"model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "quant": "awq", "hf_fallback_model": "Qwen/Qwen2.5-3B-Instruct",
                    "backend": "auto", "port": 8001, "gpu": 0, "gpu_mem": 0.85},
        "vllm": {"install": True, "version": ""},
        "max_model_len": 8192,
        "bench_llm": {"seeds": 10, "budget": 300, "parallel": 8},
        "traces": {"seeds": list(range(1000, 1012)), "budgets": [150, 300], "parallel": 8},
        "student": {"base": "Qwen/Qwen2.5-0.5B-Instruct", "epochs": 3, "lr": 2e-4, "rank": 16, "max_len": 8192,
                    "grad_accum": 8, "max_steps": -1},
        "inference": {"n": 48, "concurrency": 8, "max_tokens": 1024},
        "bench_distill": {"seeds": 10, "budget": 300, "parallel": 8},
    },
}
PROFILES["quick"] = copy.deepcopy(PROFILES["kaggle-t4"])
PROFILES["quick"].update({
    "verify": {"mimo_draws": 2_000_000, "heat_max_n": 1024, "heat_searches": 2},
    "gpu_bench": {"quick": True},
    "bench_cpu": {"seeds": 5, "budgets": [120, 300], "parallel": 4},
    "surrogate": {"sims": ["mimo"], "n": 1024, "epochs": 100},
    "teacher": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "quant": None, "hf_fallback_model": None, "backend": "auto",
                "port": 8001, "gpu": 0, "gpu_mem": 0.6},
    "bench_llm": {"seeds": 3, "budget": 300, "parallel": 8},
    "traces": {"seeds": list(range(1000, 1004)), "budgets": [300], "parallel": 8},
    "student": {"base": "Qwen/Qwen2.5-0.5B-Instruct", "epochs": 1, "lr": 2e-4, "rank": 16, "max_len": 8192,
                "grad_accum": 4, "max_steps": -1},
    "inference": {"n": 16, "concurrency": 8, "max_tokens": 1024},
    "bench_distill": {"seeds": 3, "budget": 300, "parallel": 8},
})
PROFILES["cpu-smoke"] = {
    "device": "cpu",
    "verify": {"mimo_draws": 100_000, "heat_max_n": 128, "heat_searches": 1},
    "gpu_bench": {"quick": True},
    "bench_cpu": {"seeds": 2, "budgets": [60], "parallel": 2},
    "surrogate": {"sims": ["mimo"], "n": 96, "epochs": 5},
    "teacher": {"model": "fake", "quant": None, "backend": "fake", "port": 8791, "gpu": None, "gpu_mem": 0.5},
    "vllm": {"install": False, "version": ""},
    "max_model_len": 16384,
    "bench_llm": {"seeds": 2, "budget": 60, "parallel": 2},
    "traces": {"seeds": [1000, 1001], "budgets": [100], "parallel": 2},
    "student": {"base": "tiny", "epochs": 1, "lr": 1e-3, "rank": 4, "max_len": 16384, "grad_accum": 1, "max_steps": 3},
    "inference": {"n": 3, "concurrency": 2, "max_tokens": 16},
    "bench_distill": {"seeds": 2, "budget": 60, "parallel": 2},
}

# CPU-only research run (what can be measured without a GPU), used for the results committed to the repo
PROFILES["cpu-full"] = {
    "device": "cpu",
    "verify": {"mimo_draws": 2_000_000, "heat_max_n": 512, "heat_searches": 5},
    "gpu_bench": {"quick": False},
    "bench_cpu": {"seeds": 20, "budgets": [120, 300], "parallel": 2},
    "surrogate": {"sims": ["reactor", "mimo"], "n": 4096, "epochs": 300},
    "max_model_len": 8192,
}

STEPS = ["env", "tests", "verify", "gpu_bench", "bench_cpu", "surrogate", "bench_llm", "traces", "train_student",
         "inference", "bench_distill", "report"]


class Ctx:
    def __init__(self, profile: str, results: Path, force: bool):
        self.name = profile
        self.cfg = PROFILES[profile]
        self.R = results
        self.force = force
        self.R.mkdir(parents=True, exist_ok=True)
        self.logs = self.R / "logs"
        self.logs.mkdir(exist_ok=True)
        self._vllm: str | None | bool = False  # False = not checked yet

    def vllm_python(self) -> str | None:
        if self._vllm is False:
            v = self.cfg.get("vllm", {})
            if not v.get("install"):
                self._vllm = None
            else:
                from .serving.launch import ensure_vllm

                print("installing vLLM into its own virtualenv (first time: several minutes)...", flush=True)
                self._vllm = ensure_vllm(Path(os.environ.get("SCIAGENT_VLLM_VENV", "/tmp/vllm-env")), v.get("version", ""),
                                         self.logs / "vllm_install.log")
                print("vLLM:", self._vllm or "unavailable -> transformers fallback", flush=True)
        return self._vllm or None

    def gpus(self) -> int:
        from .serving.launch import gpu_count

        return gpu_count()


def _sh(cmd: list[str], log: Path, env: dict | None = None, check: bool = True) -> int:
    print("$", " ".join(cmd), flush=True)
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}; see {log}")
    return p.returncode


# ----------------------------------------------------------------------------------------------- servers
def _teacher(ctx: Ctx):
    from .serving.launch import Server, start_server

    t = ctx.cfg["teacher"]
    if t["backend"] == "fake":
        log = ctx.logs / f"fake_{t['port']}.log"
        proc = subprocess.Popen([sys.executable, "-m", "sciagent.testing.fake_openai", "--port", str(t["port"])],
                                stdout=open(log, "w"), stderr=subprocess.STDOUT)
        from .serving.launch import _wait

        url = f"http://127.0.0.1:{t['port']}/v1"
        if not _wait(url, proc, 60):
            raise RuntimeError(f"fake server failed; see {log}")
        return Server(proc, url, "fake", "fake-openai (mock policy)", log)
    return start_server(t["model"], t["port"], served_name="teacher", backend=t["backend"], vllm_python=ctx.vllm_python(),
                        gpu=t["gpu"], quant=t["quant"], max_model_len=ctx.cfg["max_model_len"], gpu_mem=t["gpu_mem"],
                        log_dir=ctx.logs, hf_fallback_model=t.get("hf_fallback_model"))


def _student_base(ctx: Ctx) -> str:
    b = ctx.cfg["student"]["base"]
    if b == "tiny":
        from .testing.tiny_model import make_tiny_model

        texts = []
        sft = ctx.R / "traces" / "sft.jsonl"
        if sft.exists():
            for line in sft.read_text().splitlines()[:50]:
                texts += [m["content"] for m in json.loads(line)["messages"]]
        return make_tiny_model(ctx.R / "tiny_base", texts or None)
    return b


def _student_gpu(ctx: Ctx):
    if ctx.cfg["device"] != "cuda":
        return None
    return 1 if ctx.gpus() >= 2 else 0


def _llm_planner(url: str, model: str, advisors: list[str], record: bool = True, max_tokens: int = 1024) -> dict:
    return {"kind": "llm", "advisors": advisors, "options": {"min_stop_fraction": 0.6},
            "llm": {"provider": "openai", "base_url": url, "model": model, "mode": "record" if record else "live",
                    "max_tokens": max_tokens, "timeout_s": 900}}


# ----------------------------------------------------------------------------------------------- steps
def step_env(ctx: Ctx, out: Path):
    from .gpu import device_info, pick_device
    from .tracing import environment_info

    info = {"profile": ctx.name, "config": ctx.cfg, "environment": environment_info(),
            "device": device_info(pick_device(ctx.cfg["device"])), "python": sys.version, "gpus": ctx.gpus()}
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        info["nvidia_smi"] = None
    pkgs = {}
    for m in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "onnxruntime", "numpy", "scipy", "sklearn"):
        try:
            pkgs[m] = __import__(m).__version__
        except Exception:
            pkgs[m] = None
    info["packages"] = pkgs
    (out / "env.json").write_text(json.dumps(info, indent=2, default=str))


def step_tests(ctx: Ctx, out: Path):
    tests = Path(__file__).resolve().parents[2] / "tests"
    env = {"SCIAGENT_TEST_DEVICE": ctx.cfg["device"]}
    rc = _sh([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tests),
              f"--junitxml={out / 'junit.xml'}"], out / "pytest.log", env=env, check=False)
    tail = (out / "pytest.log").read_text().splitlines()[-3:]
    (out / "summary.json").write_text(json.dumps({"returncode": rc, "tail": tail}, indent=2))
    if rc != 0:
        raise RuntimeError(f"tests failed: {tail}")


def step_verify(ctx: Ctx, out: Path):
    from .verify import run

    v = ctx.cfg["verify"]
    run(ctx.cfg["device"], v["mimo_draws"], v["heat_max_n"], out, v["heat_searches"])


def step_gpu_bench(ctx: Ctx, out: Path):
    from .gpu.throughput import run

    run(ctx.cfg["device"], out, ctx.cfg["gpu_bench"]["quick"])


def step_bench_cpu(ctx: Ctx, out: Path):
    from .analysis import run as analyze
    from .bench import DEFAULT_PLANNERS, run_bench

    b = ctx.cfg["bench_cpu"]
    for budget in b["budgets"]:
        d = out / f"b{budget}"
        run_bench(["reactor", "heat", "mimo"], DEFAULT_PLANNERS, list(range(b["seeds"])), budget, 4, d,
                  verbose=False, parallel=b["parallel"])
        analyze([d], d / "analysis")


def step_surrogate(ctx: Ctx, out: Path):
    from .surrogate.pipeline import make_data, report, train

    s = ctx.cfg["surrogate"]
    for sim in s["sims"]:
        make_data(sim, s["n"], [], out / "data", seed=0)
        train(sim, out / "data", out / "models", s["epochs"], 0, ctx.cfg["device"])
        report(sim, out / "data", out / "models")


def step_bench_llm(ctx: Ctx, out: Path):
    from .analysis import run as analyze
    from .bench import run_bench

    b = ctx.cfg["bench_llm"]
    with _teacher(ctx) as srv:
        (out / "server.json").write_text(json.dumps({"backend": srv.backend, "model": ctx.cfg["teacher"]["model"]}))
        planners = {
            "teacher+gp-advisor": _llm_planner(srv.base_url, srv.model, ["bo"]),
            "teacher": _llm_planner(srv.base_url, srv.model, []),
            "bo": {"kind": "bo"},
            "scripted-mf": {"kind": "llm", "llm": {"provider": "mock"}},
        }
        run_bench(["reactor", "heat", "mimo"], planners, list(range(b["seeds"])), b["budget"], 4, out, keep_runs=True,
                  verbose=True, parallel=b["parallel"])
    analyze([out], out / "analysis")


def step_traces(ctx: Ctx, out: Path):
    from .bench import run_bench

    t = ctx.cfg["traces"]
    with _teacher(ctx) as srv:
        for budget in t["budgets"]:
            run_bench(["reactor", "heat", "mimo"], {"teacher": _llm_planner(srv.base_url, srv.model, ["bo"])},
                      t["seeds"], budget, 4, out / f"b{budget}", keep_runs=True, verbose=True, parallel=t["parallel"])
    allruns = out / "all_runs"
    allruns.mkdir(exist_ok=True)
    for budget in t["budgets"]:
        for d in (out / f"b{budget}" / "runs").iterdir():
            link = allruns / f"b{budget}-{d.name}"
            if not link.exists():
                link.symlink_to(d.resolve())
    _sh([sys.executable, str(RECIPES / "make_sft_data.py"), str(allruns), "--out", str(out / "sft.jsonl")],
        out / "make_sft.log")


def step_train_student(ctx: Ctx, out: Path):
    s = ctx.cfg["student"]
    sft = ctx.R / "traces" / "sft.jsonl"
    if not sft.exists():
        raise RuntimeError("run the 'traces' step first")
    gpu = _student_gpu(ctx)
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {"CUDA_VISIBLE_DEVICES": ""}
    _sh([sys.executable, str(RECIPES / "train_lora.py"), "--data", str(sft), "--base", _student_base(ctx),
         "--out", str(out / "planner-lora"), "--epochs", str(s["epochs"]), "--lr", str(s["lr"]), "--rank", str(s["rank"]),
         "--max-len", str(s["max_len"]), "--grad-accum", str(s["grad_accum"]), "--max-steps", str(s["max_steps"])],
        out / "train.log", env=env)


def _student_servers(ctx: Ctx):
    """(label, kwargs for start_server) for the base model, the distilled student, and its 4-bit version."""
    merged = str(ctx.R / "train_student" / "planner-lora" / "merged")
    base = _student_base(ctx)
    gpu = _student_gpu(ctx)
    common = {"vllm_python": ctx.vllm_python(), "gpu": gpu, "max_model_len": ctx.cfg["max_model_len"], "gpu_mem": 0.4,
              "log_dir": ctx.logs, "backend": "auto"}
    out = [("base-0.5b", dict(model=base, port=8003, served_name="base", **common)),
           ("distilled-0.5b", dict(model=merged, port=8002, served_name="student", **common))]
    if ctx.cfg["device"] == "cuda":
        out.append(("distilled-0.5b-4bit", dict(model=merged, port=8004, served_name="student4", quant="bnb4", **common)))
    return out


def step_inference(ctx: Ctx, out: Path):
    from .serving.launch import start_server

    i = ctx.cfg["inference"]
    val = ctx.R / "traces" / "sft.val.jsonl"
    rows = []

    def measure(label, srv):
        f = out / f"{label}.json"
        _sh([sys.executable, str(RECIPES / "bench_inference.py"), "--base-url", srv.base_url, "--model", srv.model,
             "--data", str(val), "--n", str(i["n"]), "--concurrency", str(i["concurrency"]), "--max-tokens",
             str(i["max_tokens"]), "--label", label, "--out", str(f)], out / "inference.log")
        r = json.loads(f.read_text())
        r["backend"] = srv.backend
        f.write_text(json.dumps(r, indent=2))
        rows.append(r)

    with _teacher(ctx) as srv:
        measure("teacher", srv)
    for label, kw in _student_servers(ctx):
        try:
            with start_server(**kw) as srv:
                measure(label, srv)
        except RuntimeError as e:  # e.g. 4-bit kernels unavailable: record, keep going
            rows.append({"label": label, "error": str(e)[:500]})
    (out / "inference_summary.json").write_text(json.dumps(rows, indent=2))


def step_bench_distill(ctx: Ctx, out: Path):
    from .analysis import run as analyze
    from .bench import run_bench
    from .serving.launch import start_server

    b = ctx.cfg["bench_distill"]
    servers = []
    try:
        planners = {"bo": {"kind": "bo"}}
        for label, kw in _student_servers(ctx)[:2]:  # base and fp16 student, held-out seeds 0..n-1
            srv = start_server(**kw)
            servers.append(srv)
            planners[label] = _llm_planner(srv.base_url, srv.model, ["bo"], max_tokens=ctx.cfg["inference"]["max_tokens"])
        run_bench(["reactor", "heat", "mimo"], planners, list(range(b["seeds"])), b["budget"], 4, out, keep_runs=True,
                  verbose=True, parallel=b["parallel"])
    finally:
        for s in servers:
            s.stop()
    dirs = [out]
    if (ctx.R / "bench_llm" / "results.csv").exists():
        dirs.append(ctx.R / "bench_llm")  # teacher results on the same seeds and budget
    analyze(dirs, out / "analysis")


def step_report(ctx: Ctx, out: Path):
    from .report import build

    build([ctx.R], out / "TECHNICAL_REPORT.md")
    import shutil

    shutil.make_archive(str(ctx.R.parent / f"{ctx.R.name}_bundle"), "zip", ctx.R)


def run(profile: str, results: str | Path, steps: list[str] | None = None, force: bool = False) -> dict:
    ctx = Ctx(profile, Path(results), force)
    todo = steps or STEPS
    status = {}
    for name in todo:
        out = ctx.R / name
        done = out / "DONE.json"
        if done.exists() and not force:
            print(f"== {name}: already done, skipping", flush=True)
            status[name] = "skipped"
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"== {name}", flush=True)
        t0 = time.time()
        try:
            globals()[f"step_{name}"](ctx, out)
            done.write_text(json.dumps({"seconds": time.time() - t0, "profile": profile, "finished": time.time()}))
            status[name] = f"ok ({time.time() - t0:.0f}s)"
        except Exception as e:
            (out / "ERROR.txt").write_text(traceback.format_exc())
            status[name] = f"FAILED: {type(e).__name__}: {str(e)[:300]}"
            print(status[name], flush=True)
            if name in ("traces",) and "train_student" in todo:
                print("   (later distillation steps need this one)", flush=True)
    sf = ctx.R / "experiments_status.json"
    merged = json.loads(sf.read_text()) if sf.exists() else {}
    merged.update({k: v for k, v in status.items() if v != "skipped" or k not in merged})
    sf.write_text(json.dumps(merged, indent=2))
    print(json.dumps(status, indent=2))
    return status
