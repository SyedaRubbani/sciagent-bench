"""Measure planner serving efficiency on an OpenAI-compatible endpoint (vLLM): latency per plan, output
tokens/s, and JSON validity, replaying real planning prompts from the SFT validation set.

    python bench_inference.py --base-url http://localhost:8002/v1 --model sciagent-planner-0.5b --data sft.val.jsonl
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="sft.val.jsonl")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=1500)
ap.add_argument("--label", default=None, help="name for this configuration in the output")
ap.add_argument("--out", default=None, help="write the JSON result here")
a = ap.parse_args()

rows = [json.loads(line) for line in open(a.data) if line.strip()][: a.n]


def one(ex):
    t0 = time.perf_counter()
    r = httpx.post(f"{a.base_url}/chat/completions", timeout=900, json={
        "model": a.model, "messages": ex["messages"][:-1], "temperature": 0.0, "max_tokens": a.max_tokens}).json()
    dt = time.perf_counter() - t0
    text = r["choices"][0]["message"]["content"] or ""
    try:
        start, end = text.find("{"), text.rfind("}")
        ok = isinstance(json.loads(text[start: end + 1]).get("runs"), list)
    except Exception:
        ok = False
    return dt, r["usage"]["completion_tokens"], ok


t0 = time.perf_counter()
with ThreadPoolExecutor(a.concurrency) as pool:
    res = list(pool.map(one, rows))
wall = time.perf_counter() - t0
lat = [r[0] for r in res]
out = {"label": a.label or a.model, "model": a.model, "requests": len(res), "concurrency": a.concurrency,
       "latency_p50_s": statistics.median(lat), "latency_p90_s": sorted(lat)[int(0.9 * (len(lat) - 1))],
       "aggregate_output_tok_per_s": sum(r[1] for r in res) / wall, "valid_json_plan_rate": sum(r[2] for r in res) / len(res)}
out["wall_s"] = wall
print(json.dumps(out, indent=2))
if a.out:
    from pathlib import Path

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
