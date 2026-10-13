"""HTTP API + trace viewer for launching, watching and steering studies.

    sciagent serve --runs-dir runs      # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .orchestrator import Orchestrator
from .simulators import get_simulator, list_simulators
from .spec import StudySpec
from .steering import steer

_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")


class SteerRequest(BaseModel):
    pause: bool | None = None
    stop: bool | None = None
    note: str | None = None
    bounds: dict[str, tuple[float, float]] | None = None
    compute_units: float | None = None


def create_app(runs_dir: str | Path = "runs") -> FastAPI:
    root = Path(runs_dir)
    root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="SciAgent-Bench", version="0.1.0",
                  description="Launch, inspect and steer budget-aware simulation campaigns.")
    threads: dict[str, threading.Thread] = {}
    errors: dict[str, str] = {}

    def _dir(study_id: str) -> Path:
        if not _ID.match(study_id):
            raise HTTPException(400, "invalid study id")
        d = root / study_id
        if not d.is_dir():
            raise HTTPException(404, f"no study {study_id}")
        return d

    def _read_json(p: Path) -> Any:
        return json.loads(p.read_text()) if p.exists() else None

    @app.get("/", response_class=HTMLResponse)
    def viewer():
        return (Path(__file__).parent / "static" / "viewer.html").read_text()

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/simulators")
    def simulators():
        return [get_simulator(n).card() for n in list_simulators()]

    @app.post("/api/studies", status_code=201)
    def create_study(spec: dict = Body(..., description="a StudySpec as JSON")):
        try:
            s = StudySpec.model_validate(spec)
            get_simulator(s.simulator)
        except Exception as e:
            raise HTTPException(422, str(e)) from e
        sid = f"{re.sub(r'[^A-Za-z0-9_.-]', '-', s.name)}-{time.strftime('%Y%m%d-%H%M%S')}-{len(threads)}"
        run_dir = root / sid
        orch = Orchestrator(s, run_dir)

        def work():
            try:
                orch.run()
            except Exception as e:  # surfaced through the API
                errors[sid] = f"{type(e).__name__}: {e}"

        t = threading.Thread(target=work, name=sid, daemon=True)
        threads[sid] = t
        t.start()
        return {"id": sid}

    @app.get("/api/studies")
    def list_studies():
        out = []
        for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir() or not (d / "spec.yaml").exists():
                continue
            st = _read_json(d / "status.json") or {}
            sm = _read_json(d / "summary.json") or {}
            out.append({"id": d.name, "status": st.get("status", "unknown"), "n_runs": st.get("n_runs"),
                        "spent_compute": st.get("spent_compute"), "compute_budget": st.get("compute_budget"),
                        "simulator": sm.get("simulator"), "planner": sm.get("planner"), "regret": sm.get("regret")})
        return out

    @app.get("/api/studies/{study_id}")
    def get_study(study_id: str):
        d = _dir(study_id)
        return {"id": study_id, "status": _read_json(d / "status.json"), "summary": _read_json(d / "summary.json"),
                "manifest": _read_json(d / "manifest.json"), "spec": (d / "spec.yaml").read_text(),
                "error": errors.get(study_id)}

    @app.get("/api/studies/{study_id}/runs")
    def get_runs(study_id: str):
        f = _dir(study_id) / "runs.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []

    @app.get("/api/studies/{study_id}/trace")
    def get_trace(study_id: str, kinds: str | None = Query(None, description="comma-separated event kinds"),
                  after: int = 0):
        f = _dir(study_id) / "trace.jsonl"
        if not f.exists():
            return []
        want = set(kinds.split(",")) if kinds else None
        evs = [json.loads(x) for x in f.read_text().splitlines()[after:]]
        return [e for e in evs if want is None or e["kind"] in want]

    @app.post("/api/studies/{study_id}/steer")
    def steer_study(study_id: str, req: SteerRequest):
        d = _dir(study_id)
        st = _read_json(d / "status.json") or {}
        if st.get("status") in ("finished", "failed"):
            raise HTTPException(409, f"study is {st['status']}")
        return steer(d, **req.model_dump())

    return app
