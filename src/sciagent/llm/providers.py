"""LLM providers behind one small interface, plus record/replay for reproducible agent runs.

* ``anthropic``  -- Anthropic Messages API
* ``openai``     -- any OpenAI-compatible Chat Completions endpoint: OpenAI, a local vLLM server
                    (``vllm serve <model>``), Ollama (``http://localhost:11434/v1``), etc.
* ``mock``       -- deterministic offline planner used in tests, CI and demos (no network, no key)

A :class:`CassetteProvider` wraps any of them. In ``record`` mode it stores every request/response pair
keyed by a hash of the request; in ``replay`` mode it serves them back, so an LLM-driven campaign can be
re-executed bit-for-bit without calling the model again.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from ..spec import LLMSpec


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_s: float = 0.0
    cached: bool = False


class Provider:
    model: str = ""

    def complete(self, system: str, messages: list[dict], temperature: float, max_tokens: int) -> LLMResponse:
        raise NotImplementedError


def _sleep(seconds: float) -> None:
    """Back-off pause between retries (a separate function so tests can skip the wait)."""
    time.sleep(seconds)


def _post_with_retry(url: str, headers: dict, body: dict, timeout: float, tries: int = 4) -> dict:
    delay = 2.0
    for attempt in range(tries):
        try:
            r = httpx.post(url, headers=headers, json=body, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 529) and attempt < tries - 1:
                _sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            return r.json()
        except httpx.TransportError:
            if attempt == tries - 1:
                raise
            _sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


class AnthropicProvider(Provider):
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, spec: LLMSpec):
        self.model = spec.model
        self.spec = spec
        key_env = spec.api_key_env or "ANTHROPIC_API_KEY"
        self.key = os.environ.get(key_env)
        if not self.key:
            raise RuntimeError(f"set {key_env} to use the anthropic provider")

    def complete(self, system, messages, temperature, max_tokens):
        t0 = time.perf_counter()
        body = {"model": self.model, "system": system, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature}
        headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = _post_with_retry(self.spec.base_url or self.URL, headers, body, self.spec.timeout_s)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage", {})
        return LLMResponse(text, u.get("input_tokens", 0), u.get("output_tokens", 0), data.get("model", self.model),
                           time.perf_counter() - t0)


class OpenAICompatibleProvider(Provider):
    def __init__(self, spec: LLMSpec):
        self.model = spec.model
        self.spec = spec
        self.base = (spec.base_url or "https://api.openai.com/v1").rstrip("/")
        key_env = spec.api_key_env or "OPENAI_API_KEY"
        self.key = os.environ.get(key_env, "EMPTY")  # local vLLM/Ollama servers accept any key

    def complete(self, system, messages, temperature, max_tokens):
        t0 = time.perf_counter()
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, *messages],
                "temperature": temperature, "max_tokens": max_tokens}
        headers = {"Authorization": f"Bearer {self.key}", "content-type": "application/json"}
        data = _post_with_retry(f"{self.base}/chat/completions", headers, body, self.spec.timeout_s)
        text = data["choices"][0]["message"].get("content") or ""
        u = data.get("usage") or {}
        return LLMResponse(text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), data.get("model", self.model),
                           time.perf_counter() - t0)


class MockProvider(Provider):
    """A deterministic, rule-based stand-in for an LLM.

    It reads the STATE_JSON block that the LLM planner puts in every prompt and answers with a valid plan:
    explore at the cheapest fidelity, refine around the best points at the middle fidelity, then confirm
    at the top fidelity. It exists so the full agent loop (prompting, parsing, repair, budgeting, tracing,
    record/replay) can be tested without network access -- it is *not* a claim about LLM quality.
    """

    model = "mock-planner"

    def __init__(self, spec: LLMSpec | None = None, fail_first: int = 0):
        if spec is not None:
            self.model = spec.model
        self.fail_first = fail_first
        self.calls = 0

    def complete(self, system, messages, temperature, max_tokens):
        self.calls += 1
        prompt = messages[-1]["content"]
        if self.calls <= self.fail_first:
            return LLMResponse("Sure! I think we should run some simulations.", 50, 10, self.model)
        m = re.search(r"STATE_JSON\s*```json\s*(\{.*?\})\s*```", prompt, re.S)
        if not m:  # repair turn: the state is in the first message
            m = re.search(r"STATE_JSON\s*```json\s*(\{.*?\})\s*```", messages[0]["content"], re.S)
        state = json.loads(m.group(1))
        plan = _mock_policy(state)
        text = json.dumps(plan)
        return LLMResponse(text, len(prompt) // 4, len(text) // 4, self.model)


def _mock_policy(s: dict) -> dict:
    import numpy as np

    rng = np.random.default_rng([s.get("seed", 0), s["history_total"], s["batch"]])
    params = s["params"]
    fids = s["fidelities"]
    n = s["batch_size"]
    remaining = s["budget"]["remaining_compute_units"]
    hist = [h for h in s["history"] if h["objective"] is not None]

    def sample():
        out = {}
        for p in params:
            v = rng.uniform(p["low"], p["high"])
            out[p["name"]] = int(round(v)) if p["kind"] == "int" else float(v)
        return out

    def perturb(x, scale):
        out = {}
        for p in params:
            span = p["high"] - p["low"]
            v = min(max(x[p["name"]] + rng.normal(0, scale * span), p["low"]), p["high"])
            out[p["name"]] = int(round(v)) if p["kind"] == "int" else float(v)
        return out

    total = s["budget"]["compute_units"]
    used_frac = 1 - remaining / total
    if used_frac < 0.25 or len(hist) < 2 * len(params):
        fid, why = fids[0], "explore broadly at the cheapest fidelity"
        runs = [sample() for _ in range(n)]
    elif used_frac < 0.6:
        fid, why = fids[min(1, len(fids) - 1)], "refine around the best cheap results at a mid fidelity"
        best = sorted(hist, key=lambda h: h["objective"], reverse=True)[:3]
        runs = [perturb(best[i % len(best)]["params"], 0.05) for i in range(n)]
    else:
        fid, why = fids[-1], "confirm the most promising candidates at the top fidelity"
        top_lvl = fid["level"]
        done = {json.dumps(h["params"], sort_keys=True) for h in hist if h["fidelity_level"] == top_lvl}
        ranked = sorted(hist, key=lambda h: (h["fidelity_level"], h["objective"]), reverse=True)
        fresh = [h["params"] for h in ranked if json.dumps(h["params"], sort_keys=True) not in done][:1]
        seeds = [h["params"] for h in ranked[:2]]
        runs = fresh + [perturb(seeds[i % len(seeds)], 0.02) for i in range(n - len(fresh))]
    k = int(min(n, remaining // fid["cost"]))
    return {"reasoning_summary": f"mock policy: {why}", "stop": k == 0,
            "runs": [{"params": r, "fidelity": fid["name"], "why": why} for r in runs[:k]]}


class CassetteProvider(Provider):
    def __init__(self, inner: Provider | None, path: str | Path, mode: str, model: str):
        self.inner, self.path, self.mode = inner, Path(path), mode
        self.model = model
        self.tape: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.tape[rec["key"]] = rec["response"]
        elif mode == "replay":
            raise FileNotFoundError(f"cassette {self.path} not found for replay")

    @staticmethod
    def key(model, system, messages, temperature, max_tokens) -> str:
        blob = json.dumps([model, system, messages, temperature, max_tokens], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, system, messages, temperature, max_tokens):
        k = self.key(self.model, system, messages, temperature, max_tokens)
        if k in self.tape:
            resp = LLMResponse(**self.tape[k])
            resp.cached = True
            return resp
        if self.mode == "replay":
            raise KeyError("replay miss: the prompt differs from the recorded run (did the code or spec change?)")
        resp = self.inner.complete(system, messages, temperature, max_tokens)
        self.tape[k] = asdict(resp)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": k, "model": self.model, "system": system, "messages": messages,
                                 "response": asdict(resp)}) + "\n")
        return resp


def make_provider(spec: LLMSpec, **mock_kwargs) -> Provider:
    if spec.mode == "replay":
        if not spec.cassette:
            raise ValueError("replay mode needs llm.cassette")
        inner = None
    elif spec.provider == "anthropic":
        inner = AnthropicProvider(spec)
    elif spec.provider == "openai":
        inner = OpenAICompatibleProvider(spec)
    else:
        inner = MockProvider(spec, **mock_kwargs)
    if spec.cassette and spec.mode in ("record", "replay"):
        return CassetteProvider(inner, spec.cassette, spec.mode, inner.model if inner else spec.model)
    return inner
