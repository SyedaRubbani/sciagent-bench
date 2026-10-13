"""MCP server: expose the simulators and campaign control as tools to any MCP client
(Claude Desktop, Claude Code, an IDE agent...).

    sciagent mcp --runs-dir runs

Claude Desktop config (claude_desktop_config.json):
    {"mcpServers": {"sciagent": {"command": "sciagent", "args": ["mcp", "--runs-dir", "/abs/path/runs"]}}}
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .cache import RunCache, run_key
from .orchestrator import Orchestrator
from .simulators import get_simulator, list_simulators
from .spec import StudySpec
from .steering import steer


def build_server(runs_dir: str | Path = "runs"):
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP

    root = Path(runs_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache = RunCache(root / ".mcp_cache")
    mcp = FastMCP("sciagent")

    def _dir(study_id: str) -> Path:
        if not re.match(r"^[A-Za-z0-9_.\-]+$", study_id) or not (root / study_id).is_dir():
            raise ValueError(f"unknown study {study_id!r}")
        return root / study_id

    @mcp.tool()
    def simulators() -> list[dict]:
        """List available scientific simulators with their parameters, fidelity levels and costs."""
        return [get_simulator(n).card() for n in list_simulators()]

    @mcp.tool()
    def run_simulation(simulator: str, params: dict[str, float], fidelity: str, seed: int = 0) -> dict:
        """Run ONE simulation at a chosen fidelity (cached by content hash). Use for quick what-if checks."""
        sim = get_simulator(simulator)
        p = sim.normalize(params)
        key = run_key(sim.name, sim.version, p, fidelity, seed if sim.stochastic else None)
        hit = cache.get(key)
        if hit:
            return {**hit, "cached": True}
        res = asdict(sim.run(p, fidelity, seed))
        if res["ok"]:
            cache.put(key, res)
        return {**res, "cached": False}

    @mcp.tool()
    def start_study(spec: dict[str, Any]) -> dict:
        """Launch a budget-bounded campaign in the background from a StudySpec (same schema as the YAML files)."""
        s = StudySpec.model_validate(spec)
        sid = f"{re.sub(r'[^A-Za-z0-9_.-]', '-', s.name)}-{time.strftime('%Y%m%d-%H%M%S')}"
        orch = Orchestrator(s, root / sid)
        threading.Thread(target=orch.run, daemon=True).start()
        return {"study_id": sid}

    @mcp.tool()
    def study_status(study_id: str) -> dict:
        """Budget, incumbent and (when finished) summary of a study."""
        d = _dir(study_id)
        out = {"status": json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else None}
        if (d / "summary.json").exists():
            out["summary"] = json.loads((d / "summary.json").read_text())
        return out

    @mcp.tool()
    def steer_study(study_id: str, note: str | None = None, pause: bool | None = None, stop: bool | None = None,
                    bounds: dict[str, list[float]] | None = None, compute_units: float | None = None) -> dict:
        """Steer a running study: add a note for the planner, narrow parameter bounds, change budget, pause or stop."""
        return steer(_dir(study_id), note=note, pause=pause, stop=stop,
                     bounds={k: (v[0], v[1]) for k, v in (bounds or {}).items()} or None, compute_units=compute_units)

    @mcp.tool()
    def study_trace(study_id: str, kinds: str = "plan,guard,steer", limit: int = 50) -> list[dict]:
        """Recent decision-trace events of a study (comma-separated kinds: plan, guard, run, llm.call, steer...)."""
        f = _dir(study_id) / "trace.jsonl"
        want = set(kinds.split(","))
        evs = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
        return [e for e in evs if e["kind"] in want][-limit:]

    return mcp


def main(runs_dir: str | Path = "runs"):
    build_server(runs_dir).run()
