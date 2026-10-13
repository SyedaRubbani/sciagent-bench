"""Batched Monte-Carlo ergodic MIMO capacity on any torch device (millions of channel draws on a GPU)."""

from __future__ import annotations

import math

from ..simulators.mimo import BANDWIDTH_MHZ, GAIN_OVER_NOISE, NR, RX_CORR, _power, _rx_corr_sqrt


def ergodic_capacity_torch(snr: float, k: int, draws: int, seed: int = 0, device: str = "cpu", nr: int = NR,
                           rho: float = RX_CORR, chunk: int = 262_144, dtype=None) -> tuple[float, float]:
    """Return (mean, standard error) in bit/s/Hz, accumulated in float64 over chunks."""
    import torch

    cdtype = dtype or torch.complex128
    g = torch.Generator(device=device).manual_seed(seed)
    Rs = torch.as_tensor(_rx_corr_sqrt(nr, rho), dtype=cdtype, device=device)
    eye = torch.eye(k, dtype=cdtype, device=device)
    s1 = s2 = 0.0
    done = 0
    real = torch.float64 if cdtype == torch.complex128 else torch.float32
    while done < draws:
        m = min(chunk, draws - done)
        re = torch.randn((m, nr, k), generator=g, device=device, dtype=real)
        im = torch.randn((m, nr, k), generator=g, device=device, dtype=real)
        H = Rs @ (torch.complex(re, im) / math.sqrt(2))
        G = eye + (snr / k) * (H.mH @ H)
        c = torch.linalg.slogdet(G).logabsdet.to(torch.float64) / math.log(2)
        s1 += float(c.sum())
        s2 += float((c * c).sum())
        done += m
    mean = s1 / draws
    var = max(s2 / draws - mean * mean, 0.0) * draws / max(draws - 1, 1)
    return mean, math.sqrt(var / draws)


def energy_efficiency_torch(p_dbm: float, n_tx: int, draws: int, seed: int = 0, device: str = "cpu", **kw):
    snr = 10 ** (p_dbm / 10) / 1000 * GAIN_OVER_NOISE
    c, se = ergodic_capacity_torch(snr, n_tx, draws, seed, device, **kw)
    P = _power(p_dbm, n_tx)
    return BANDWIDTH_MHZ * c / P, BANDWIDTH_MHZ * se / P
