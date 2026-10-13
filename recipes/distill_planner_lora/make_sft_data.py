"""Turn recorded teacher-LLM planning calls into a supervised fine-tuning set for a small planner.

Keeps only calls that (1) parsed on the first attempt, (2) passed the guard without dropped runs, and
(3) came from studies whose final regret is in the best `--keep-frac` of the teacher's own runs per task,
so the student imitates the teacher's *good* decisions rather than all of them.

    python make_sft_data.py teacher_runs/runs --out sft.jsonl --keep-frac 0.7
"""

import argparse
import json
import random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("runs_root")
ap.add_argument("--out", default="sft.jsonl")
ap.add_argument("--keep-frac", type=float, default=0.7)
ap.add_argument("--val-frac", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

studies = []
for d in sorted(Path(a.runs_root).iterdir()):
    if (d / "llm_cassette.jsonl").exists() and (d / "summary.json").exists():
        s = json.loads((d / "summary.json").read_text())
        studies.append((d, s))

by_task = {}
for d, s in studies:
    by_task.setdefault(s["simulator"], []).append((s.get("regret") or 0.0, d))
keep_dirs = set()
for _task, lst in by_task.items():
    lst.sort()
    keep_dirs |= {d for _, d in lst[: max(1, int(len(lst) * a.keep_frac))]}

examples, stats = [], {"studies": len(studies), "kept_studies": len(keep_dirs), "calls": 0, "kept_calls": 0}
for d, _s in studies:
    if d not in keep_dirs:
        continue
    trace = [json.loads(line) for line in (d / "trace.jsonl").read_text().splitlines()]
    bad_batches = set()
    for ev in trace:  # batches where the guard had to drop or clip something
        if ev["kind"] == "guard" and any(i["action"] in ("dropped", "clipped") for i in ev["data"]["issues"]):
            bad_batches.add(ev["parent_id"])
        if ev["kind"] == "plan" and ev["data"]["usage"].get("repairs"):
            bad_batches.add(ev["parent_id"])
    good_responses = set()
    for ev in trace:
        if ev["kind"] == "llm.call" and ev["parent_id"] not in bad_batches and ev["data"]["attempt"] == 0:
            good_responses.add(ev["data"]["response"])
    for line in (d / "llm_cassette.jsonl").read_text().splitlines():
        rec = json.loads(line)
        stats["calls"] += 1
        if len(rec["messages"]) != 1:  # repair turns are not imitation targets
            continue
        text = rec["response"]["text"]
        if text[:4000] not in good_responses:
            continue
        examples.append({"messages": [{"role": "system", "content": rec["system"]}, rec["messages"][0],
                                      {"role": "assistant", "content": text}]})
        stats["kept_calls"] += 1

random.Random(a.seed).shuffle(examples)
n_val = max(1, int(len(examples) * a.val_frac))
Path(a.out).write_text("\n".join(json.dumps(e) for e in examples[n_val:]) + "\n")
Path(a.out).with_suffix(".val.jsonl").write_text("\n".join(json.dumps(e) for e in examples[:n_val]) + "\n")
print(json.dumps(stats, indent=2))
