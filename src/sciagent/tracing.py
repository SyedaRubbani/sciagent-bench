"""Traceable runs: an append-only JSONL event log plus a run manifest.

Every planner decision, LLM call, guard intervention, simulation and steering action is an event with a
deterministic span id and a parent id, so a run can be audited, diffed and replayed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import __version__


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item"):  # numpy scalars
        return x.item()
    if isinstance(x, float) and x != x:
        return None
    return x


class Tracer:
    def __init__(self, run_dir: Path | None):
        self.run_dir = Path(run_dir) if run_dir else None
        self._n = 0
        self._fh = None
        self.events: list[dict] = []
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.run_dir / "trace.jsonl", "a", encoding="utf-8")

    def event(self, kind: str, data: dict | None = None, parent: str | None = None, **extra) -> str:
        self._n += 1
        sid = f"s{self._n:05d}"
        ev = {"span_id": sid, "parent_id": parent, "kind": kind, "ts": time.time(), "data": _jsonable(data or {})}
        ev.update(_jsonable(extra))
        self.events.append(ev)
        if self._fh:
            self._fh.write(json.dumps(ev) + "\n")
            self._fh.flush()
        return sid

    @contextmanager
    def span(self, kind: str, data: dict | None = None, parent: str | None = None):
        sid = self.event(kind + ".start", data, parent)
        t0 = time.perf_counter()
        holder: dict = {}
        try:
            yield sid, holder
        finally:
            self.event(kind + ".end", {"duration_s": time.perf_counter() - t0, **holder}, parent=sid)

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], stderr=subprocess.DEVNULL,
                                      cwd=Path(__file__).parent, text=True)
        return bool(out.strip())
    except Exception:
        return None


def environment_info() -> dict:
    info = {"sciagent": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
            "git_sha": _git_sha(), "git_dirty": _git_dirty()}
    for mod in ("numpy", "scipy", "sklearn", "torch"):
        m = sys.modules.get(mod)
        if m is not None:
            info[mod] = getattr(m, "__version__", "?")
    return info


def results_digest(records: list[dict]) -> str:
    """Hash of what was run and what came out -- ignores timings, so replays can be compared."""
    core = [(r["params"], r["fidelity"], None if r["objective"] is None else round(r["objective"], 12))
            for r in records]
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]
