"""Where simulations run: in-process, a local process pool, or a Slurm cluster (via submitit)."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

from .simulators import get_simulator
from .spec import ExecutorSpec


def run_one(sim_name: str, params: dict, fidelity: str, seed: int) -> dict:
    """Top-level (picklable) entry point executed on a worker or cluster node."""
    return asdict(get_simulator(sim_name).run(params, fidelity, seed))


class Executor:
    def map(self, sim_name: str, jobs: list[tuple[dict, str, int]]) -> list[dict]:
        raise NotImplementedError

    def close(self):
        pass


class InlineExecutor(Executor):
    def map(self, sim_name, jobs):
        return [run_one(sim_name, p, f, s) for p, f, s in jobs]


class PoolExecutor(Executor):
    def __init__(self, workers: int):
        # spawn: safe even when the orchestrator runs inside a threaded server
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))

    def map(self, sim_name, jobs):
        futs = [self.pool.submit(run_one, sim_name, p, f, s) for p, f, s in jobs]
        return [f.result() for f in futs]

    def close(self):
        self.pool.shutdown(wait=True)


class SlurmExecutor(Executor):
    """Submit each batch as a Slurm job array. ``cluster='local'`` or ``'debug'`` runs without Slurm (for tests)."""

    def __init__(self, folder: str, slurm: dict):
        import submitit

        opts = dict(slurm)
        cluster = opts.pop("cluster", None)
        self.ex = submitit.AutoExecutor(folder=folder, cluster=cluster)
        if opts:
            self.ex.update_parameters(**opts)

    def map(self, sim_name, jobs):
        if not jobs:
            return []
        ps, fs, ss = zip(*jobs)
        array = self.ex.map_array(run_one, [sim_name] * len(jobs), list(ps), list(fs), list(ss))
        return [j.result() for j in array]


def make_executor(spec: ExecutorSpec, run_dir=None) -> Executor:
    if spec.kind == "slurm":
        from pathlib import Path

        folder = Path(run_dir or ".") / spec.folder
        return SlurmExecutor(str(folder), spec.slurm)
    return PoolExecutor(spec.workers) if spec.workers > 1 else InlineExecutor()
