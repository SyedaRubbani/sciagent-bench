"""GPU (PyTorch) implementations of the simulator kernels.

Used for high-resolution reference solutions, convergence studies and throughput benchmarks. They run on
CPU too (``device="cpu"``), which is how the test suite checks them against the NumPy/SciPy versions.
The benchmark tasks themselves keep their CPU definitions so that results do not depend on the device.
"""

from __future__ import annotations


def pick_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_info(device: str) -> dict:
    import torch

    info = {"device": device, "torch": torch.__version__}
    if device.startswith("cuda") and torch.cuda.is_available():
        i = torch.device(device).index or 0
        p = torch.cuda.get_device_properties(i)
        info.update(name=p.name, memory_gb=round(p.total_memory / 2**30, 1), capability=f"{p.major}.{p.minor}",
                    count=torch.cuda.device_count(), cuda=torch.version.cuda)
    return info


def sync(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
