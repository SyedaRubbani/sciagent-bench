"""Throughput of the simulator kernels: NumPy/SciPy (CPU) vs PyTorch on a device.

    sciagent gpu-bench --device cuda --out gpu_results
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import device_info, pick_device, sync


def _time(fn, repeats=3):
    fn()  # warm-up (allocations, kernels, caches)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def mimo(device: str, draws_list=(65_536, 1_048_576, 8_388_608), k: int = 6) -> list[dict]:
    from ..simulators.mimo import ergodic_capacity
    from .mimo_torch import ergodic_capacity_torch

    rows = []
    for n in draws_list:
        row = {"draws": n}
        if n <= 1_048_576:  # numpy baseline gets slow and memory-hungry beyond this
            row["numpy_s"] = _time(lambda n=n: ergodic_capacity(10.0, k, n, np.random.default_rng(0)), 1)
        for name, dt in (("torch_fp64", None), ("torch_fp32", "c64")):
            import torch

            cd = torch.complex64 if dt else torch.complex128

            def f(n=n, cd=cd):
                ergodic_capacity_torch(10.0, k, n, 0, device, dtype=cd)
                sync(device)
            row[f"{name}_s"] = _time(f, 2)
        row["draws_per_s_best"] = n / min(v for kk, v in row.items() if kk.endswith("_s"))
        rows.append(row)
    return rows


def heat(device: str, ns=(64, 128, 256, 512, 1024)) -> list[dict]:
    from ..simulators.heat import solve_plate
    from .heat_torch import solve_plate_torch

    rows = []
    for n in ns:
        row = {"n": n, "unknowns": n * n}
        if n <= 512:
            row["scipy_direct_s"] = _time(lambda n=n: solve_plate(0.52, 0.5, 0.04, n), 1)
        holder = {}

        def f(n=n, holder=holder):
            _, holder["info"] = solve_plate_torch(0.52, 0.5, 0.04, n, device=device)
            sync(device)
        row["torch_cg_s"] = _time(f, 1)
        row["cg_iterations"] = holder["info"]["iterations"]
        rows.append(row)
    return rows


def run(device: str = "auto", out: str | Path = "gpu_results", quick: bool = False) -> dict:
    dev = pick_device(device)
    res = {"device": device_info(dev),
           "mimo": mimo(dev, (65_536, 262_144) if quick else (65_536, 1_048_576, 8_388_608)),
           "heat": heat(dev, (64, 128) if quick else (64, 128, 256, 512, 1024))}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "throughput.json").write_text(json.dumps(res, indent=2))
    (out / "throughput.md").write_text(to_markdown(res))
    return res


def to_markdown(r: dict) -> str:
    d = r["device"]
    L = [f"# Kernel throughput on `{d.get('name', d['device'])}`", "", "## MIMO Monte-Carlo capacity (8 x 6 complex channel)", "",
         "| draws | NumPy (s) | torch fp64 (s) | torch fp32 (s) | best draws/s |", "|---|---|---|---|---|"]
    for x in r["mimo"]:
        npy = "–" if x.get("numpy_s") is None else f"{x['numpy_s']:.3f}"
        L.append(f"| {x['draws']:,} | {npy} | {x['torch_fp64_s']:.3f} | "
                 f"{x['torch_fp32_s']:.3f} | {x['draws_per_s_best']:,.0f} |")
    L += ["", "## Heat plate: sparse direct (SciPy) vs matrix-free PCG (torch, fp64, rtol 1e-10)", "",
          "| grid | unknowns | SciPy direct (s) | torch PCG (s) | PCG iterations |", "|---|---|---|---|---|"]
    for x in r["heat"]:
        sd = x.get("scipy_direct_s")
        L.append(f"| {x['n']} | {x['unknowns']:,} | {'–' if sd is None else f'{sd:.3f}'} | {x['torch_cg_s']:.3f} | {x['cg_iterations']} |")
    return "\n".join(L) + "\n"
