"""Surrogate pipeline: curate simulation data -> train a PyTorch teacher -> distil a small student ->
quantise (PyTorch dynamic int8, ONNX Runtime int8) -> benchmark accuracy, size and latency.

The surrogate is then usable as an *advisor* to the planning agent (see advisor.py): a learned model that
answers "what would the top-fidelity simulator probably say here?" in microseconds.

    sciagent surrogate make-data --sim reactor --n 6000 [--runs runs/*]
    sciagent surrogate train     --sim reactor
    sciagent surrogate report    --sim reactor
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np

from ..cache import run_key
from ..simulators import get_simulator

# ----------------------------------------------------------------------------- data curation


def make_data(sim_name: str, n: int, run_dirs: list[str], data_dir: str | Path, seed: int = 0) -> dict:
    from scipy.stats import qmc

    sim = get_simulator(sim_name)
    out = Path(data_dir) / sim_name
    out.mkdir(parents=True, exist_ok=True)
    rows, log = [], {"generated": 0, "from_runs": 0, "dropped_failed": 0, "dropped_nonfinite": 0,
                     "dropped_duplicate": 0, "dropped_unstable": 0}
    seen = set()

    def add(params, fid, objective, metrics, ok, source, seed_):
        if not ok:
            log["dropped_failed"] += 1
            return
        if objective is None or not math.isfinite(objective):
            log["dropped_nonfinite"] += 1
            return
        if metrics.get("numerically_unstable"):
            log["dropped_unstable"] += 1  # a diverged explicit solve is not physics; keep it out of training
            return
        key = run_key(sim.name, sim.version, params, fid, seed_ if sim.stochastic else None)
        if key in seen:
            log["dropped_duplicate"] += 1
            return
        seen.add(key)
        rows.append({**{p.name: params[p.name] for p in sim.params}, "fidelity": fid,
                     "fidelity_level": sim.fidelity(fid).level, "objective": objective, "source": source,
                     "config_id": run_key(sim.name, sim.version, params, "-", None)[:12]})

    d = len(sim.params)
    for fid in sim.fidelities:
        u = qmc.Sobol(d, scramble=True, seed=seed + fid.level).random(n)
        for j, x in enumerate(u):
            params = {p.name: p.from_unit(x[i]) for i, p in enumerate(sim.params)}
            s = seed * 1_000_003 + fid.level * 100_000 + j
            r = sim.run(params, fid.name, s)
            add(params, fid.name, r.objective, r.metrics, r.ok, "sobol", s)
            log["generated"] += 1
    for rd in run_dirs:
        f = Path(rd) / "runs.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            log["from_runs"] += 1
            add(rec["params"], rec["fidelity"], rec["objective"], rec.get("metrics", {}), rec["ok"], f"run:{Path(rd).name}",
                rec["seed"])

    # split by configuration (not by row) so the same design never appears in train and test
    def split_of(cid: str) -> str:
        h = int(cid[:8], 16) % 10
        return "test" if h == 0 else ("val" if h == 1 else "train")

    for r in rows:
        r["split"] = split_of(r["config_id"])
    path = out / "dataset.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    card = {"simulator": sim.card(), "rows": len(rows), "sha256": sha, "curation": log,
            "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
            "per_fidelity": {f.name: sum(r["fidelity"] == f.name for r in rows) for f in sim.fidelities},
            "objective_range": [min(r["objective"] for r in rows), max(r["objective"] for r in rows)]}
    (out / "datacard.json").write_text(json.dumps(card, indent=2))
    (out / "datacard.md").write_text(_datacard_md(card))
    return card


def _datacard_md(c: dict) -> str:
    sim = c["simulator"]
    lines = [f"# Data card: `{sim['name']}` surrogate dataset", "",
             f"*Simulator* {sim['name']} v{sim['version']} - {sim['description']}", "",
             f"*Rows* {c['rows']}  *SHA-256* `{c['sha256'][:16]}...`", "",
             "## Curation", "", "| step | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in c["curation"].items()]
    lines += ["", "## Splits (by configuration hash - no design leaks between splits)", "",
              "| split | rows |", "|---|---|"] + [f"| {k} | {v} |" for k, v in c["splits"].items()]
    lines += ["", "## Rows per fidelity", "", "| fidelity | rows |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in c["per_fidelity"].items()]
    lines += ["", "## Features", "", "| name | range | unit |", "|---|---|---|"]
    lines += [f"| {p['name']} | [{p['low']:g}, {p['high']:g}] | {p['unit']} |" for p in sim["params"]]
    lines += ["| fidelity | one-hot over levels | |", "", f"Target: `{sim['objective']['name']}` "
              f"(range {c['objective_range'][0]:.4g} to {c['objective_range'][1]:.4g})."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- models


def _load(sim, data_dir):
    rows = list(csv.DictReader(open(Path(data_dir) / sim.name / "dataset.csv")))
    nf = len(sim.fidelities)

    def feats(r):
        x = [sim.param(p.name).to_unit(float(r[p.name])) for p in sim.params]
        oh = [0.0] * nf
        oh[int(r["fidelity_level"])] = 1.0
        return x + oh

    out = {}
    for s in ("train", "val", "test"):
        sel = [r for r in rows if r["split"] == s]
        out[s] = (np.array([feats(r) for r in sel], dtype=np.float32), np.array([float(r["objective"]) for r in sel],
                                                                                dtype=np.float32),
                  np.array([int(r["fidelity_level"]) for r in sel]))
    return out


def _mlp(d_in, widths):
    import torch.nn as nn

    layers, prev = [], d_in
    for w in widths:
        layers += [nn.Linear(prev, w), nn.SiLU()]
        prev = w
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class _Wrapped:
    """Model + target normalisation."""

    def __init__(self, net, y_mu, y_sd):
        self.net, self.y_mu, self.y_sd = net, y_mu, y_sd


def _fit(net, X, y, Xv, yv, epochs, lr, seed, soft=None, alpha=0.0, device="cpu"):
    import torch

    torch.manual_seed(seed)
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(y)[:, None].to(device)
    st = None if soft is None else torch.from_numpy(soft)[:, None].to(device)
    Xv_, yv_ = torch.from_numpy(Xv).to(device), torch.from_numpy(yv)[:, None].to(device)
    best, best_state = float("inf"), None
    g = torch.Generator().manual_seed(seed)
    bs = 512
    for _ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=g).to(device)
        for i in range(0, len(Xt), bs):
            idx = perm[i: i + bs]
            pred = net(Xt[idx])
            loss = torch.mean((pred - yt[idx]) ** 2)
            if st is not None:
                loss = (1 - alpha) * loss + alpha * torch.mean((pred - st[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = float(torch.mean((net(Xv_) - yv_) ** 2))
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.to("cpu")
    return best


def train(sim_name: str, data_dir, model_dir, epochs: int = 300, seed: int = 0, device: str = "cpu") -> dict:
    import torch

    from ..gpu import device_info, pick_device

    device = pick_device(device)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    sim = get_simulator(sim_name)
    D = _load(sim, data_dir)
    Xtr, ytr, _ = D["train"]
    Xv, yv, _ = D["val"]
    mu, sd = float(ytr.mean()), float(ytr.std())
    n = lambda y: ((y - mu) / sd).astype(np.float32)  # noqa: E731
    d_in = Xtr.shape[1]
    out = Path(model_dir) / sim_name
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    t0 = time.perf_counter()
    teacher = _mlp(d_in, [128, 128, 128])
    _fit(teacher, Xtr, n(ytr), Xv, n(yv), epochs, 3e-3, seed, device=device)
    timings["teacher_s"] = time.perf_counter() - t0

    # distillation: student matches the teacher on many fresh inputs (cheap to label) + true labels
    rng = np.random.default_rng(seed)
    nf = len(sim.fidelities)
    Xaug = rng.uniform(0, 1, (len(Xtr) * 4, d_in - nf)).astype(np.float32)
    oh = np.eye(nf, dtype=np.float32)[rng.integers(0, nf, len(Xaug))]
    Xaug = np.hstack([Xaug, oh])
    with torch.no_grad():
        teacher.eval().to("cpu")
        soft_aug = teacher(torch.from_numpy(Xaug)).numpy()[:, 0]
        soft_tr = teacher(torch.from_numpy(Xtr)).numpy()[:, 0]
    Xd = np.vstack([Xtr, Xaug])
    yd = np.concatenate([n(ytr), soft_aug]).astype(np.float32)  # aug rows: teacher labels only
    sd_ = np.concatenate([soft_tr, soft_aug]).astype(np.float32)
    t0 = time.perf_counter()
    student = _mlp(d_in, [32, 32])
    _fit(student, Xd, yd, Xv, n(yv), epochs, 3e-3, seed + 1, soft=sd_, alpha=0.5, device=device)
    timings["student_distilled_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    scratch = _mlp(d_in, [32, 32])  # ablation: same small net, labels only, same number of gradient steps
    scratch_epochs = int(round(epochs * len(Xd) / len(Xtr)))
    _fit(scratch, Xtr, n(ytr), Xv, n(yv), scratch_epochs, 3e-3, seed + 2, device=device)
    timings["student_scratch_s"] = time.perf_counter() - t0

    meta = {"simulator": sim_name, "sim_version": sim.version, "features": [p.name for p in sim.params],
            "fidelities": [f.name for f in sim.fidelities], "y_mu": mu, "y_sd": sd,
            "bounds": {p.name: [p.low, p.high] for p in sim.params}, "train_timings_s": timings, "seed": seed,
            "train_device": device_info(device),
            "epochs": epochs}
    for name, net in [("teacher", teacher), ("student", student), ("student_scratch", scratch)]:
        torch.save(net.state_dict(), out / f"{name}.pt")
        net.eval()
        _export_onnx(net, d_in, out / f"{name}.onnx")
    import logging

    from onnxruntime.quantization import QuantType, quantize_dynamic

    logging.getLogger().setLevel(logging.ERROR)  # silence ORT's "consider pre-processing" hint
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantize_dynamic(str(out / "student.onnx"), str(out / "student.int8.onnx"), weight_type=QuantType.QInt8)
    meta["widths"] = {"teacher": [128, 128, 128], "student": [32, 32], "student_scratch": [32, 32]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _export_onnx(net, d_in, path):
    import torch

    x = torch.zeros(1, d_in)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(net, x, str(path), input_names=["x"], output_names=["y"],
                              dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}, opset_version=17, dynamo=False)
        except TypeError:  # older torch without the dynamo flag
            torch.onnx.export(net, x, str(path), input_names=["x"], output_names=["y"],
                              dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}, opset_version=17)


# ----------------------------------------------------------------------------- evaluation


def _metrics(pred, y):
    err = pred - y
    ss = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2))),
            "r2": 1.0 - float(np.sum(err**2)) / ss}


def _latency(fn, X, repeats):
    fn(X)  # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(X)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def report(sim_name: str, data_dir, model_dir) -> dict:
    import onnxruntime as ort
    import torch
    import torch.ao.quantization as tq

    sim = get_simulator(sim_name)
    out = Path(model_dir) / sim_name
    meta = json.loads((out / "meta.json").read_text())
    D = _load(sim, data_dir)
    Xte, yte, lte = D["test"]
    mu, sd = meta["y_mu"], meta["y_sd"]
    d_in = Xte.shape[1]
    top = len(sim.fidelities) - 1
    torch.set_num_threads(1)  # fair single-thread latency comparison

    nets = {}
    for name in ("teacher", "student", "student_scratch"):
        net = _mlp(d_in, meta["widths"][name])
        net.load_state_dict(torch.load(out / f"{name}.pt"))
        nets[name] = net.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nets["student_int8_torch"] = tq.quantize_dynamic(nets["student"], {torch.nn.Linear}, dtype=torch.qint8)

    def torch_fn(net):
        def f(X):
            with torch.no_grad():
                return net(torch.from_numpy(X)).numpy()[:, 0] * sd + mu
        return f

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    variants = {
        "teacher (torch fp32)": (torch_fn(nets["teacher"]), out / "teacher.pt"),
        "student distilled (torch fp32)": (torch_fn(nets["student"]), out / "student.pt"),
        "student from scratch (torch fp32)": (torch_fn(nets["student_scratch"]), out / "student_scratch.pt"),
        "student distilled (torch int8 dynamic)": (torch_fn(nets["student_int8_torch"]), None),
    }
    for label, fname in [("student distilled (ONNX Runtime fp32)", "student.onnx"),
                         ("student distilled (ONNX Runtime int8)", "student.int8.onnx")]:
        sess = ort.InferenceSession(str(out / fname), so, providers=["CPUExecutionProvider"])
        variants[label] = ((lambda s: lambda X: s.run(None, {"x": X})[0][:, 0] * sd + mu)(sess), out / fname)

    n_params = {k: sum(p.numel() for p in nets[k].parameters()) for k in ("teacher", "student", "student_scratch")}
    rows = []
    X1 = Xte[:1].copy()
    Xb = np.repeat(Xte, max(1, 4096 // len(Xte) + 1), axis=0)[:4096].copy()
    for label, (fn, path) in variants.items():
        pred = fn(Xte)
        m_all = _metrics(pred, yte)
        m_top = _metrics(pred[lte == top], yte[lte == top])
        size = path.stat().st_size if path is not None and Path(path).exists() else None
        if "int8 dynamic" in label:
            import io

            buf = io.BytesIO()
            torch.save(nets["student_int8_torch"].state_dict(), buf)
            size = buf.getbuffer().nbytes
        rows.append({"model": label, "mae_all": m_all["mae"], "r2_all": m_all["r2"], "mae_top_fidelity": m_top["mae"],
                     "r2_top_fidelity": m_top["r2"], "size_bytes": size,
                     "latency_b1_us": 1e6 * _latency(fn, X1, 300), "latency_b4096_ms": 1e3 * _latency(fn, Xb, 30)})

    # what the surrogate replaces: one top-fidelity simulator call
    rng = np.random.default_rng(0)
    ts = []
    for _ in range(20):
        params = {p.name: p.sample(rng) for p in sim.params}
        t0 = time.perf_counter()
        sim.run(params, sim.top_fidelity.name, 0)
        ts.append(time.perf_counter() - t0)
    sim_us = 1e6 * float(np.median(ts))
    res = {"simulator": sim_name, "test_rows": int(len(yte)), "test_rows_top_fidelity": int((lte == top).sum()),
           "n_params": n_params, "simulator_top_fidelity_latency_us": sim_us, "models": rows,
           "train_timings_s": meta["train_timings_s"], "train_device": meta.get("train_device"),
           "torch": torch.__version__, "onnxruntime": ort.__version__}
    (out / "report.json").write_text(json.dumps(res, indent=2))
    (out / "report.md").write_text(_report_md(res))
    return res


def _report_md(r: dict) -> str:
    def size(b):
        return "–" if b is None else (f"{b / 1024:.1f} KiB")

    lines = [f"# Surrogate report: `{r['simulator']}`", "",
             f"Held-out test set: {r['test_rows']} rows ({r['test_rows_top_fidelity']} at top fidelity), split by "
             f"design so no test configuration was seen in training. Latency is single-threaded CPU, median.", "",
             f"Parameters: teacher {r['n_params']['teacher']:,}, student {r['n_params']['student']:,} "
             f"({r['n_params']['teacher'] / r['n_params']['student']:.0f}x smaller).", "",
             "| model | MAE (all) | R² (all) | MAE (top fid.) | R² (top fid.) | size | latency b=1 (µs) | latency b=4096 (ms) |",
             "|---|---|---|---|---|---|---|---|"]
    for m in r["models"]:
        lines.append(f"| {m['model']} | {m['mae_all']:.4g} | {m['r2_all']:.4f} | {m['mae_top_fidelity']:.4g} | "
                     f"{m['r2_top_fidelity']:.4f} | {size(m['size_bytes'])} | {m['latency_b1_us']:.1f} | "
                     f"{m['latency_b4096_ms']:.2f} |")
    best = min(m["latency_b4096_ms"] for m in r["models"]) / 4096 * 1e3
    lines += ["", f"One top-fidelity simulator call: {r['simulator_top_fidelity_latency_us']:.0f} µs. The fastest "
              f"surrogate variant amortises to {best:.2f} µs per prediction in batch, i.e. ~"
              f"{r['simulator_top_fidelity_latency_us'] / best:,.0f}x cheaper per query.", "",
              f"Trained on {(r.get('train_device') or {}).get('name', (r.get('train_device') or {}).get('device', 'cpu'))} "
              f"in {sum(r['train_timings_s'].values()):.0f} s. Environment: torch {r['torch']}, onnxruntime {r['onnxruntime']}."]
    return "\n".join(lines) + "\n"


def cli(a):
    if a.action == "make-data":
        card = make_data(a.sim, a.n, a.runs, a.data_dir, a.seed)
        print(json.dumps({k: card[k] for k in ("rows", "curation", "splits", "per_fidelity")}, indent=2))
    elif a.action == "train":
        meta = train(a.sim, a.data_dir, a.model_dir, a.epochs, a.seed, a.device)
        print(json.dumps(meta["train_timings_s"], indent=2))
    else:
        r = report(a.sim, a.data_dir, a.model_dir)
        print(_report_md(r))
