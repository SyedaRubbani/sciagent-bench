import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from sciagent.orchestrator import Orchestrator  # noqa: E402
from sciagent.testing.fake_openai import serve_in_thread  # noqa: E402


def test_llm_planner_over_real_http_with_retries(make_spec, monkeypatch):
    from sciagent.llm import providers

    monkeypatch.setattr(providers.time, "sleep", lambda s: None)  # fast retries
    url, server = serve_in_thread(fail_every=4)  # every 4th request returns 503
    try:
        spec = make_spec("reactor", planner={"kind": "llm", "llm": {"provider": "openai", "base_url": url,
                                                                    "model": "fake"}}, budget=80)
        http = Orchestrator(spec).run()
        local = Orchestrator(make_spec("reactor", planner={"kind": "llm"}, budget=80)).run()
    finally:
        server.should_exit = True
    assert http["n_runs"] > 0 and http["llm_tokens_in"] > 0
    assert http["results_digest"] == local["results_digest"]  # same policy over HTTP == in-process


def test_api_token_auth(tmp_path):
    from sciagent.service import create_app

    c = TestClient(create_app(tmp_path, token="s3cret"))
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/studies").status_code == 401
    assert c.get("/api/studies", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/api/studies", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    r = c.post("/api/studies", json={"name": "a", "simulator": "reactor", "budget": {"compute_units": 20}})
    assert r.status_code == 401
