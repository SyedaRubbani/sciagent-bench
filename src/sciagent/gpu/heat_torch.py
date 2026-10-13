"""Matrix-free preconditioned conjugate gradient for the heat-plate problem, on any torch device.

Solves (-lap + beta*chi) T = q with zero Dirichlet boundaries on an n x n interior grid, exactly the system
assembled by ``sciagent.simulators.heat.solve_plate``, but without forming a matrix, so grids of
1024 x 1024 and beyond fit on one GPU.
"""

from __future__ import annotations

import math

from ..simulators.heat import BETA, Q_SCALE, SOURCES


def _grid(n: int, device, dtype):
    import torch

    h = 1.0 / (n + 1)
    xs = torch.linspace(h, 1 - h, n, device=device, dtype=dtype)
    Y, X = torch.meshgrid(xs, xs, indexing="ij")  # row = y, col = x (matches numpy meshgrid 'xy')
    return X, Y, h


def solve_plate_torch(px: float, py: float, r: float, n: int, device: str = "cpu", dtype=None, rtol: float = 1e-10,
                      max_iter: int | None = None):
    """Return (T as an n x n tensor, info dict with iterations and final relative residual)."""
    import torch

    dtype = dtype or torch.float64
    X, Y, h = _grid(n, device, dtype)
    eps = max(h, 1.0 / 128) * 0.75
    dist = torch.sqrt((X - px) ** 2 + (Y - py) ** 2)
    chi = 0.5 * (1.0 - torch.tanh((dist - r) / eps))
    q = torch.zeros_like(X)
    for sx, sy, a, w in SOURCES:
        q += a * torch.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (2 * w * w))
    q *= Q_SCALE
    sink = BETA * chi
    inv_h2 = 1.0 / (h * h)

    def A(u):
        up = torch.nn.functional.pad(u, (1, 1, 1, 1))
        lap = 4 * u - up[:-2, 1:-1] - up[2:, 1:-1] - up[1:-1, :-2] - up[1:-1, 2:]
        return lap * inv_h2 + sink * u

    Minv = 1.0 / (4 * inv_h2 + sink)  # Jacobi preconditioner
    x = torch.zeros_like(q)
    rres = q.clone()
    z = Minv * rres
    p = z.clone()
    rz = torch.sum(rres * z)
    bnorm = torch.linalg.vector_norm(q)
    max_iter = max_iter or 20 * n + 200
    it = 0
    rel = 1.0
    for it in range(1, max_iter + 1):  # noqa: B007
        Ap = A(p)
        alpha = rz / torch.sum(p * Ap)
        x = x + alpha * p
        rres = rres - alpha * Ap
        if it % 25 == 0 or it == max_iter:
            rel = float(torch.linalg.vector_norm(rres) / bnorm)
            if rel < rtol:
                break
        z = Minv * rres
        rz_new = torch.sum(rres * z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, {"iterations": it, "rel_residual": rel, "n": n, "h": h, "converged": rel < 10 * rtol}


def heat_objective_torch(px, py, r, n, device="cpu", **kw) -> tuple[float, dict]:
    from ..simulators.heat import AREA_COST

    T, info = solve_plate_torch(px, py, r, n, device=device, **kw)
    peak = float(T.max())
    return -peak - AREA_COST * math.pi * r * r, {"peak_temp": peak, **info}
