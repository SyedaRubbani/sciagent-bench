"""Surrogate advisor: a distilled ONNX model the LLM planner can consult each batch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class SurrogateAdvisor:
    def __init__(self, model_dir: str | Path, sim, variant: str = "student.onnx"):
        import onnxruntime as ort

        d = Path(model_dir)
        if (d / sim.name / "meta.json").exists():
            d = d / sim.name
        self.meta = json.loads((d / "meta.json").read_text())
        if self.meta["simulator"] != sim.name:
            raise ValueError(f"surrogate was trained for {self.meta['simulator']}, not {sim.name}")
        self.sim = sim
        self.sess = ort.InferenceSession(str(d / variant), providers=["CPUExecutionProvider"])
        rep = d / "report.json"
        self.quality = None
        if rep.exists():
            r = json.loads(rep.read_text())
            m = next((m for m in r["models"] if "ONNX Runtime fp32" in m["model"]), None)
            if m:
                self.quality = {"test_r2_top_fidelity": round(m["r2_top_fidelity"], 4),
                                "test_mae_top_fidelity": round(m["mae_top_fidelity"], 5)}

    def predict(self, params_list: list[dict], fidelity_level: int | None = None) -> np.ndarray:
        nf = len(self.sim.fidelities)
        lvl = nf - 1 if fidelity_level is None else fidelity_level
        X = np.array([[self.sim.param(n).to_unit(p[n]) for n in self.meta["features"]] + list(np.eye(nf)[lvl])
                      for p in params_list], dtype=np.float32)
        y = self.sess.run(None, {"x": X})[0][:, 0]
        return y * self.meta["y_sd"] + self.meta["y_mu"]

    def advise(self, state, seed: int = 0, n_cand: int = 4000, k: int = 3) -> dict:
        rng = np.random.default_rng(seed)
        cands = [{p.name: p.sample(rng) for p in state.effective_params()} for _ in range(n_cand)]
        pred = self.predict(cands)
        top = np.argsort(pred)[-k:][::-1]
        out = {"note": "predicted top-fidelity objective from a distilled surrogate; verify with real runs",
               "quality": self.quality,
               "top_predicted": [{"params": {kk: float(f"{v:.5g}") for kk, v in cands[i].items()},
                                  "predicted": float(f"{pred[i]:.5g}")} for i in top]}
        inc = state.incumbent()
        if inc is not None:
            out["incumbent_predicted_at_top_fidelity"] = float(f"{self.predict([inc.params])[0]:.5g}")
        return out
