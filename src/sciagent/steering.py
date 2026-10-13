"""Researcher steering of a live campaign through ``control.json`` (used by the CLI, the API and MCP)."""

from __future__ import annotations

import json
from pathlib import Path


def steer(run_dir: str | Path, *, pause: bool | None = None, stop: bool | None = None, note: str | None = None,
          bounds: dict[str, tuple[float, float]] | None = None, compute_units: float | None = None) -> dict:
    path = Path(run_dir) / "control.json"
    ctl = json.loads(path.read_text()) if path.exists() else {"version": 0}
    ctl["version"] = int(ctl.get("version", 0)) + 1
    for k in ("note", "bounds", "compute_units"):
        ctl.pop(k, None)  # one-shot instructions apply once per version
    if pause is not None:
        ctl["pause"] = pause
    if stop is not None:
        ctl["stop"] = stop
    if note:
        ctl["note"] = note
    if bounds:
        ctl["bounds"] = {k: [float(a), float(b)] for k, (a, b) in bounds.items()}
    if compute_units is not None:
        ctl["compute_units"] = float(compute_units)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ctl))
    tmp.replace(path)
    return ctl
