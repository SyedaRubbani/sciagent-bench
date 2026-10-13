"""Energy efficiency of an uplink 8-receive-antenna MIMO link (information theory / wireless).

Choose the transmit power and the number of active transmit RF chains to maximise energy
efficiency (Mbit per Joule):

    C   = E_H[ log2 det(I + (snr / k) H H^H) ]   (ergodic Shannon capacity, equal power over k streams)
    EE  = B * C / (P_tx / eta + k * P_rf + P_static)

H is an 8 x k i.i.d. Rayleigh channel with mild receive correlation. More antennas and power raise
capacity but also circuit power, so the optimum is interior and lies on a mixed integer/continuous
space. Fidelity = number of Monte-Carlo channel draws: cheap estimates are *noisy but unbiased*
(contrast with the reactor and heat tasks, whose cheap fidelities are biased).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .base import Fidelity, Param, Simulator

NR = 8
BANDWIDTH_MHZ = 1.0
PA_EFF = 0.35
P_RF_W = 0.12
P_STATIC_W = 0.4
GAIN_OVER_NOISE = 100.0  # receive SNR (linear) per Watt transmitted -> 20 dB at 30 dBm
RX_CORR = 0.3


@lru_cache(maxsize=16)
def _rx_corr_sqrt(nr: int = NR, rho: float = RX_CORR) -> np.ndarray:
    """Symmetric square root of the exponential receive-correlation matrix R_ij = rho^|i-j|."""
    idx = np.arange(nr)
    R = rho ** np.abs(idx[:, None] - idx[None, :])
    w, V = np.linalg.eigh(R)
    return (V * np.sqrt(w)) @ V.T


def ergodic_capacity(snr: float, k: int, draws: int, rng: np.random.Generator, nr: int = NR,
                     rho: float = RX_CORR) -> tuple[float, float]:
    """Return (mean, standard error) of capacity in bit/s/Hz."""
    H = (rng.standard_normal((draws, nr, k)) + 1j * rng.standard_normal((draws, nr, k))) / np.sqrt(2)
    H = _rx_corr_sqrt(nr, rho) @ H
    G = np.eye(k) + (snr / k) * np.conj(np.swapaxes(H, 1, 2)) @ H  # k x k form (Sylvester)
    _, logdet = np.linalg.slogdet(G)
    c = logdet / np.log(2)
    return float(c.mean()), float(c.std(ddof=1) / np.sqrt(draws))


def _power(p_dbm: float, k: int) -> float:
    p_w = 10 ** (p_dbm / 10) / 1000
    return p_w / PA_EFF + k * P_RF_W + P_STATIC_W


class MimoSimulator(Simulator):
    name = "mimo"
    version = "1.0"
    description = (
        "Uplink MIMO link with 8 receive antennas. Choose transmit power (dBm) and the number of active "
        "transmit RF chains (1-8) to maximise energy efficiency in Mbit/J. Capacity grows with power and "
        "streams, but every active chain costs circuit power. Cheap fidelities are noisy Monte-Carlo "
        "estimates (unbiased); the metric 'stderr' reports their uncertainty."
    )
    objective_name = "energy_efficiency_Mbit_per_J"
    stochastic = True
    params = (
        Param("p_dbm", 0.0, 30.0, unit="dBm", description="total transmit power"),
        Param("n_tx", 1, 8, kind="int", description="number of active transmit RF chains / streams"),
    )
    fidelities = (
        Fidelity("mc64", 0, 1.0, "64 Monte-Carlo channel draws; noisy, unbiased"),
        Fidelity("mc512", 1, 4.0, "512 draws"),
        Fidelity("mc4096", 2, 16.0, "4096 draws; standard error ~1% of the cheap estimate's"),
    )
    _DRAWS = {"mc64": 64, "mc512": 512, "mc4096": 4096}

    def _evaluate(self, p, fidelity, rng):
        k = int(p["n_tx"])
        snr = 10 ** (p["p_dbm"] / 10) / 1000 * GAIN_OVER_NOISE
        c, se = ergodic_capacity(snr, k, self._DRAWS[fidelity.name], rng)
        P = _power(p["p_dbm"], k)
        ee = BANDWIDTH_MHZ * c / P
        return ee, {"capacity_bps_hz": c, "stderr": BANDWIDTH_MHZ * se / P, "power_W": P, "snr_db": 10 * np.log10(snr)}

    def true_objective(self, params) -> float:
        p = self.normalize(params)
        return _true_ee(round(float(p["p_dbm"]), 6), int(p["n_tx"]))


@lru_cache(maxsize=4096)
def _true_ee(p_dbm: float, k: int) -> float:
    snr = 10 ** (p_dbm / 10) / 1000 * GAIN_OVER_NOISE
    c, _ = ergodic_capacity(snr, k, 40_000, np.random.default_rng(2024))
    return BANDWIDTH_MHZ * c / _power(p_dbm, k)
