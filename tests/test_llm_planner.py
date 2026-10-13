import json

import pytest

from sciagent.cli import main
from sciagent.llm.providers import CassetteProvider, LLMResponse, MockProvider, Provider
from sciagent.orchestrator import Orchestrator
from sciagent.planners.llm_planner import LLMPlanner, extract_json
from sciagent.spec import LLMSpec


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_repair_loop_recovers(make_spec):
    spec = make_spec("reactor", planner={"kind": "llm", "options": {"mock_fail_first": 1}}, budget=40)
    o = Orchestrator(spec)
    s = o.run()
    assert s["n_runs"] > 0
    assert o.planner.provider.calls >= 2


class Garbage(Provider):
    model = "garbage"

    def complete(self, system, messages, temperature, max_tokens):
        return LLMResponse('{"runs": [{"params": {"T": 1}, "fidelity": "warp_drive"}]}', 10, 10, self.model)


def test_invalid_plans_stop_gracefully(make_spec):
    spec = make_spec("reactor", planner={"kind": "llm"}, budget=40)
    s = Orchestrator(spec, provider=Garbage()).run()
    assert s["n_runs"] == 0 and "no runnable work" in s["stop_reason"]


def test_llm_cost_budget_and_bo_fallback(make_spec):
    spec = make_spec("reactor", planner={"kind": "llm", "options": {"fallback": "bo"},
                                         "llm": {"provider": "mock", "price_in_per_mtok": 1000.0}}, budget=200)
    spec.budget.llm_usd = 0.5
    o = Orchestrator(spec)
    s = o.run()
    assert s["spent_compute"] > 0
    rationales = [e["data"]["rationale"] for e in o.tracer.events if e["kind"] == "plan"]
    assert any("-> BO fallback" in r for r in rationales)


def test_prompt_contains_state_and_steering(make_spec):
    spec = make_spec("reactor", planner={"kind": "llm", "advisors": ["bo"]}, budget=100, description="find B")
    o = Orchestrator(spec)
    o.state.notes.append("avoid hot runs")
    msg = o.planner.build_messages(o.state)[0]["content"]
    assert "STATE_JSON" in msg and "avoid hot runs" in msg and "find B" in msg and "euler_coarse" in msg


def test_record_then_replay_is_identical(make_spec, tmp_path):
    cas = tmp_path / "cas.jsonl"
    spec = make_spec("heat", planner={"kind": "llm", "advisors": ["bo"],
                                      "llm": {"provider": "mock", "mode": "record", "cassette": str(cas)}}, budget=60)
    a = Orchestrator(spec).run()
    spec2 = make_spec("heat", planner={"kind": "llm", "advisors": ["bo"],
                                       "llm": {"provider": "mock", "mode": "replay", "cassette": str(cas)}}, budget=60)
    b = Orchestrator(spec2).run()
    assert a["results_digest"] == b["results_digest"]


def test_replay_miss_raises(tmp_path):
    cas = tmp_path / "c.jsonl"
    p = CassetteProvider(MockProvider(), cas, "record", "m")
    with pytest.raises(AttributeError):  # the mock finds no STATE_JSON block to plan from
        p.complete("s", [{"role": "user", "content": "no state"}], 0, 10)
    cas.write_text("")
    r = CassetteProvider(None, cas, "replay", "m")
    with pytest.raises(KeyError):
        r.complete("s", [{"role": "user", "content": "x"}], 0, 10)


def test_cli_run_and_replay(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "s.yaml"
    spec.write_text("name: cli\nsimulator: reactor\nbudget: {compute_units: 40}\nplanner: {kind: llm}\n")
    out = tmp_path / "run"
    main(["run", str(spec), "--out", str(out), "--record"])
    assert (out / "llm_cassette.jsonl").exists()
    with pytest.raises(SystemExit) as e:
        main(["replay", str(out)])
    assert e.value.code == 0
    assert "IDENTICAL" in capsys.readouterr().out
    main(["trace", str(out)])


def test_openai_provider_request_shape(monkeypatch):
    from sciagent.llm import providers

    seen = {}

    def fake_post(url, headers, body, timeout, tries=4):
        seen.update(url=url, body=body)
        return {"choices": [{"message": {"content": '{"runs": []}'}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    monkeypatch.setattr(providers, "_post_with_retry", fake_post)
    p = providers.OpenAICompatibleProvider(LLMSpec(provider="openai", model="m", base_url="http://localhost:8001/v1/"))
    r = p.complete("sys", [{"role": "user", "content": "hi"}], 0.0, 10)
    assert seen["url"] == "http://localhost:8001/v1/chat/completions"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert (r.input_tokens, r.output_tokens) == (7, 3)


def test_anthropic_provider_request_shape(monkeypatch):
    from sciagent.llm import providers

    seen = {}

    def fake_post(url, headers, body, timeout, tries=4):
        seen.update(url=url, headers=headers, body=body)
        return {"content": [{"type": "text", "text": "{}"}], "usage": {"input_tokens": 5, "output_tokens": 2}}
    monkeypatch.setattr(providers, "_post_with_retry", fake_post)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    p = providers.AnthropicProvider(LLMSpec(provider="anthropic", model="claude-x"))
    r = p.complete("sys", [{"role": "user", "content": "hi"}], 0.0, 10)
    assert seen["body"]["system"] == "sys" and seen["headers"]["x-api-key"] == "k"
    assert "anthropic-version" in seen["headers"] and r.input_tokens == 5


def test_mock_state_json_is_deterministic(make_spec):
    spec = make_spec("mimo", planner={"kind": "llm"}, budget=50)
    o1, o2 = Orchestrator(spec), Orchestrator(spec)
    assert isinstance(o1.planner, LLMPlanner)
    assert json.dumps(o1.planner.state_json(o1.state)) == json.dumps(o2.planner.state_json(o2.state))
