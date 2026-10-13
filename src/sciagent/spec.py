"""Study specification: what a researcher writes (YAML) to launch a campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class Budget(BaseModel):
    compute_units: float = Field(..., gt=0, description="total declared simulator cost allowed")
    llm_usd: float | None = Field(None, ge=0, description="max spend on planner LLM calls")
    wall_clock_s: float | None = Field(None, gt=0, description="stop after this many seconds (non-reproducible)")
    max_batches: int = Field(50, gt=0)


class LLMSpec(BaseModel):
    provider: Literal["anthropic", "openai", "mock"] = "mock"
    model: str = "mock-planner"
    base_url: str | None = Field(None, description="OpenAI-compatible endpoint, e.g. a vLLM or Ollama server")
    api_key_env: str | None = Field(None, description="environment variable holding the API key")
    temperature: float = 0.0
    max_tokens: int = 1500
    price_in_per_mtok: float = Field(0.0, ge=0, description="USD per million input tokens (set from your provider)")
    price_out_per_mtok: float = Field(0.0, ge=0, description="USD per million output tokens")
    cassette: str | None = Field(None, description="path for record/replay of LLM calls")
    mode: Literal["live", "record", "replay"] = "live"
    max_repairs: int = 2
    timeout_s: float = 120.0


class PlannerSpec(BaseModel):
    kind: Literal["random", "sh", "bo", "llm"] = "sh"
    options: dict[str, Any] = Field(default_factory=dict)
    llm: LLMSpec | None = None
    advisors: list[Literal["bo", "surrogate"]] = Field(
        default_factory=list, description="tools whose suggestions the LLM planner sees each batch")

    @model_validator(mode="after")
    def _llm_needs_spec(self):
        if self.kind == "llm" and self.llm is None:
            self.llm = LLMSpec()
        return self


class ExecutorSpec(BaseModel):
    kind: Literal["local", "slurm"] = "local"
    workers: int = Field(1, ge=1)
    slurm: dict[str, Any] = Field(default_factory=dict, description="passed to submitit, e.g. {partition: gpu, timeout_min: 30}")
    folder: str = "slurm_logs"


class StudySpec(BaseModel):
    name: str
    simulator: str
    description: str = Field("", description="the research goal in the researcher's own words")
    budget: Budget
    batch_size: int = Field(4, ge=1)
    seed: int = 0
    planner: PlannerSpec = Field(default_factory=PlannerSpec)
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict, description="narrow the parameter space")
    executor: ExecutorSpec = Field(default_factory=ExecutorSpec)
    cache_dir: str | None = Field(None, description="shared result cache; default is per-run")

    @field_validator("bounds")
    @classmethod
    def _ordered(cls, v):
        for k, (lo, hi) in v.items():
            if lo > hi:
                raise ValueError(f"bounds for {k}: low > high")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> StudySpec:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def digest(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
