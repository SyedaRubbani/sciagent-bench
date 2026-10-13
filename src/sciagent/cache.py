"""Content-addressed result cache: never pay twice for the same deterministic simulation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def run_key(sim_name: str, sim_version: str, params: dict, fidelity: str, seed: int | None) -> str:
    canon = {k: (float(f"{float(v):.12g}") if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
             for k, v in sorted(params.items())}
    blob = json.dumps({"sim": sim_name, "v": sim_version, "p": canon, "f": fidelity, "seed": seed}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class RunCache:
    def __init__(self, directory: str | Path | None):
        self.dir = Path(directory) if directory else None
        self._mem: dict[str, dict] = {}
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        if key in self._mem:
            return self._mem[key]
        if self.dir:
            p = self.dir / key[:2] / f"{key}.json"
            if p.exists():
                self._mem[key] = json.loads(p.read_text())
                return self._mem[key]
        return None

    def put(self, key: str, value: dict) -> None:
        self._mem[key] = value
        if self.dir:
            d = self.dir / key[:2]
            d.mkdir(exist_ok=True)
            tmp = d / f".{key}.tmp"
            tmp.write_text(json.dumps(value))
            tmp.replace(d / f"{key}.json")
