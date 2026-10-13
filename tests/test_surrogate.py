import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")


def test_surrogate_pipeline_small(tmp_path, make_spec):
    from sciagent.orchestrator import Orchestrator
    from sciagent.surrogate.pipeline import make_data, report, train

    card = make_data("mimo", 128, [], tmp_path / "d", seed=0)
    assert card["rows"] > 300 and set(card["splits"]) == {"train", "val", "test"}
    assert (tmp_path / "d" / "mimo" / "datacard.md").exists()
    train("mimo", tmp_path / "d", tmp_path / "m", epochs=15)
    r = report("mimo", tmp_path / "d", tmp_path / "m")
    names = [m["model"] for m in r["models"]]
    assert any("int8" in n for n in names) and any("ONNX" in n for n in names)
    fp32 = next(m for m in r["models"] if m["model"] == "student distilled (ONNX Runtime fp32)")
    torch_fp32 = next(m for m in r["models"] if m["model"] == "student distilled (torch fp32)")
    assert abs(fp32["mae_all"] - torch_fp32["mae_all"]) < 1e-4  # ONNX export is faithful

    spec = make_spec("mimo", planner={"kind": "llm", "advisors": ["surrogate"],
                                      "options": {"surrogate_path": str(tmp_path / "m")}}, budget=40)
    o = Orchestrator(spec)
    o.run()
    sj = o.planner.state_json(o.state)
    adv = sj["advisors"]["learned_surrogate"]
    assert len(adv["top_predicted"]) == 3 and adv["quality"] is not None
    json.dumps(sj)
