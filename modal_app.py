"""Run the GPU experiments on Modal (https://modal.com).

Results and the Hugging Face model cache live on Modal Volumes, so they survive between runs, models are
downloaded once, and an interrupted run resumes where it stopped (finished steps are skipped).

    pip install modal && modal setup                                   # once
    modal run --detach modal_app.py::main --profile quick              # ~1-1.5 h on an L4
    modal run --detach modal_app.py::main --profile kaggle-t4          # full run, ~3-4 h
    modal run modal_app.py::status --profile quick                     # what has finished so far
    modal volume get sciagent-results quick results/gpu-quick          # download results

Options: --steps "verify gpu_bench" runs only some steps. Set SCIAGENT_MODAL_GPU=A10G (or A100, H100) before
`modal run` to use another GPU.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "sciagent-bench"
REPO = "/root/sciagent-bench"
RESULTS = "/results"
GPU = os.environ.get("SCIAGENT_MODAL_GPU", "L4")

STEPS = ["env", "tests", "verify", "gpu_bench", "bench_cpu", "surrogate", "bench_llm", "traces", "train_student",
         "inference", "bench_distill", "report"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # vLLM in its own virtualenv, so its pinned torch never clashes with the project's environment
    .run_commands("python -m venv /opt/vllm-env", "/opt/vllm-env/bin/pip install --no-cache-dir vllm")
    .pip_install("transformers>=4.44", "peft>=0.12", "accelerate", "bitsandbytes", "hypothesis")
    .add_local_dir(".", REPO, copy=True,
                   ignore=["results/gpu*", "runs/**", "**/__pycache__/**", "**/*.pyc", ".venv/**", "build/**",
                           "*.egg-info/**", "*_bundle.zip"])
    .run_commands(f"pip install --no-cache-dir -e '{REPO}[dev,surrogate,mcp,plot]'")
    .env({"SCIAGENT_VLLM_VENV": "/opt/vllm-env", "HF_HOME": "/hf-cache", "PYTHONUNBUFFERED": "1"})
)

results_vol = modal.Volume.from_name("sciagent-results", create_if_missing=True)
hf_vol = modal.Volume.from_name("sciagent-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU, timeout=24 * 3600, volumes={RESULTS: results_vol, "/hf-cache": hf_vol},
              retries=modal.Retries(max_retries=2, initial_delay=30.0), cpu=4.0, memory=16384)
def run_pipeline(profile: str, steps: list[str]) -> dict:
    """Run the steps one by one, saving results to the Volume after each, so a preempted or retried container
    resumes after the last finished step."""
    import json

    out = f"{RESULTS}/{profile}"
    status = {}
    for step in steps:
        print(f"\n===== {step} =====", flush=True)
        rc = subprocess.run(["sciagent", "experiments", "--profile", profile, "--results", out, "--steps", step],
                            cwd=REPO).returncode
        status[step] = "ok" if rc == 0 else f"exit {rc}"
        results_vol.commit()  # persist after every step
        if step in ("bench_llm", "traces", "train_student", "inference", "bench_distill"):
            hf_vol.commit()   # keep downloaded models for the next run
    p = Path(out) / "experiments_status.json"
    final = json.loads(p.read_text()) if p.exists() else status
    print(json.dumps(final, indent=2))
    return final


@app.function(volumes={RESULTS: results_vol})
def read_status(profile: str) -> str:
    import json

    p = Path(f"{RESULTS}/{profile}/experiments_status.json")
    done = sorted(d.name for d in Path(f"{RESULTS}/{profile}").glob("*") if (d / "DONE.json").exists())
    return json.dumps({"finished_steps": done, "status": json.loads(p.read_text()) if p.exists() else None}, indent=2)


@app.local_entrypoint()
def main(profile: str = "quick", steps: str = ""):
    chosen = steps.split() if steps else STEPS
    unknown = [s for s in chosen if s not in STEPS]
    if unknown:
        raise SystemExit(f"unknown steps {unknown}; choose from {STEPS}")
    print(f"profile={profile} gpu={GPU} steps={chosen}")
    print(run_pipeline.remote(profile, chosen))


@app.local_entrypoint()
def status(profile: str = "quick"):
    print(read_status.remote(profile))