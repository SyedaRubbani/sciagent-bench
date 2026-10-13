import asyncio
import time

import pytest

from sciagent.bench import run_bench


def test_service_end_to_end(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sciagent.service import create_app

    c = TestClient(create_app(tmp_path))
    assert c.get("/api/health").json() == {"ok": True}
    assert len(c.get("/api/simulators").json()) >= 3
    assert "SciAgent" in c.get("/").text
    r = c.post("/api/studies", json={"name": "api", "simulator": "reactor", "budget": {"compute_units": 60},
                                     "planner": {"kind": "sh"}})
    assert r.status_code == 201
    sid = r.json()["id"]
    for _ in range(100):
        st = c.get(f"/api/studies/{sid}").json()
        if st["status"] and st["status"]["status"] == "finished":
            break
        time.sleep(0.1)
    assert st["summary"]["spent_compute"] <= 60
    assert c.get(f"/api/studies/{sid}/runs").json()
    assert c.get(f"/api/studies/{sid}/trace", params={"kinds": "plan"}).json()[0]["kind"] == "plan"
    assert c.post(f"/api/studies/{sid}/steer", json={"note": "x"}).status_code == 409
    assert c.get("/api/studies/..%2Fetc").status_code in (400, 404)
    assert c.post("/api/studies", json={"name": "bad", "simulator": "nope", "budget": {"compute_units": 1}}).status_code == 422
    assert c.get("/api/studies").json()[0]["id"] == sid


def test_mcp_tools(tmp_path):
    pytest.importorskip("mcp")
    from sciagent.mcp_server import build_server

    srv = build_server(tmp_path)

    async def go():
        names = {t.name for t in await srv.list_tools()}
        assert {"simulators", "run_simulation", "start_study", "study_status", "steer_study", "study_trace"} <= names
        await srv.call_tool("run_simulation", {"simulator": "reactor", "params": {"T": 320, "tau": 5, "cat": 0.5},
                                               "fidelity": "rk4_medium"})
    asyncio.run(go())


def test_bench_outputs(tmp_path):
    rows = run_bench(["reactor"], {"random": {"kind": "random"}, "sh": {"kind": "sh"}}, [0, 1], 64, 4, tmp_path,
                     verbose=False)
    assert len(rows) == 4 and all(r["regret"] is not None for r in rows)
    assert (tmp_path / "results.csv").exists() and "| sh |" in (tmp_path / "results.md").read_text()
