"""Start and stop model servers (vLLM preferred, transformers fallback) as subprocesses."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class Server:
    proc: subprocess.Popen
    base_url: str
    model: str
    backend: str
    log: Path

    def stop(self, timeout: float = 60):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


def _wait(url: str, proc: subprocess.Popen, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            if httpx.get(f"{url}/models", timeout=3).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(3)
    return False


def ensure_vllm(venv: str | Path, version: str = "", log: Path | None = None) -> str | None:
    """Install vLLM into its own virtualenv (keeps the notebook's torch untouched). Returns its python or None."""
    venv = Path(venv)
    py = venv / "bin" / "python"
    if py.exists() and subprocess.run([str(py), "-c", "import vllm"], capture_output=True).returncode == 0:
        return str(py)
    out = open(log, "a") if log else subprocess.DEVNULL
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, stdout=out, stderr=out)
        spec = f"vllm=={version}" if version else "vllm"
        subprocess.run([str(py), "-m", "pip", "install", "-q", spec], check=True, stdout=out, stderr=out, timeout=3600)
        subprocess.run([str(py), "-c", "import vllm"], check=True, stdout=out, stderr=out)
        return str(py)
    except Exception:
        return None


def start_server(model: str, port: int, *, served_name: str | None = None, backend: str = "auto",
                 vllm_python: str | None = None, gpu: int | str | None = None, quant: str | None = None,
                 lora: str | None = None, max_model_len: int = 8192, gpu_mem: float = 0.85, dtype: str = "auto",
                 log_dir: str | Path = ".", timeout: float = 1500, max_batch: int = 8,
                 hf_fallback_model: str | None = None) -> Server:
    """Start an OpenAI-compatible server. ``backend='auto'`` tries vLLM, then falls back to transformers.

    quant: None | "awq" | "gptq" | "bnb4" (bitsandbytes 4-bit).
    """
    served_name = served_name or model
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{port}/v1"
    tried = []
    if backend in ("auto", "vllm") and vllm_python and not lora:
        log = log_dir / f"vllm_{port}.log"
        cmd = [vllm_python, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--served-model-name",
               served_name, "--port", str(port), "--host", "127.0.0.1", "--max-model-len", str(max_model_len),
               "--gpu-memory-utilization", str(gpu_mem), "--dtype", "half" if dtype == "auto" else dtype,
               "--seed", "0"]
        if quant in ("awq", "gptq"):
            cmd += ["--quantization", quant]
        elif quant == "bnb4":
            cmd += ["--quantization", "bitsandbytes"]
        proc = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        if _wait(url, proc, timeout):
            return Server(proc, url, served_name, "vllm", log)
        proc.kill()
        tried.append(f"vllm failed (see {log})")
        if backend == "vllm":
            raise RuntimeError("; ".join(tried))
    log = log_dir / f"hf_{port}.log"
    hf_model = model
    if quant in ("awq", "gptq") and hf_fallback_model:
        hf_model = hf_fallback_model  # AWQ/GPTQ checkpoints need vLLM-style kernels; use the fallback model
    cmd = [sys.executable, "-m", "sciagent.serving.hf_server", "--model", hf_model, "--served-model-name", served_name,
           "--port", str(port), "--max-model-len", str(max_model_len), "--max-batch", str(max_batch), "--dtype", dtype]
    if quant == "bnb4":
        cmd.append("--load-in-4bit")
    if lora:
        cmd += ["--lora", lora]
    proc = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    if _wait(url, proc, timeout):
        return Server(proc, url, served_name, "transformers" + ("" if hf_model == model else f" ({hf_model})"), log)
    proc.kill()
    tried.append(f"transformers failed (see {log}): {log.read_text()[-800:] if log.exists() else ''}")
    raise RuntimeError("; ".join(tried))


def gpu_count() -> int:
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except Exception:
        return 0
