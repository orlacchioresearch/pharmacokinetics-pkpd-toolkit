"""Non-compartmental analysis (NCA) of a concentration-time profile (NumPy only).

Computes the standard model-independent PK parameters from sampled
concentration-time data:

* Cmax, Tmax                  -- observed peak and its time.
* AUC(last), AUC(inf)         -- exposure by the linear trapezoidal rule, with
                                 extrapolation to infinity using lambda_z.
* lambda_z, t-half            -- terminal elimination rate (log-linear regression
                                 of the terminal points) and half-life.
* AUMC, MRT                   -- first-moment curve and mean residence time.
* CL/F, Vz/F                  -- apparent clearance and volume (extravascular).

No SciPy: lambda_z is an ordinary least-squares fit of ln(C) vs t over the
terminal phase, chosen as the run of points giving the best adjusted R^2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:  # numpy >= 2.0 renamed trapz -> trapezoid (trapz is removed in newer releases)
    from numpy import trapezoid as _trapezoid
except ImportError:  # numpy < 2.0
    from numpy import trapz as _trapezoid


@dataclass(frozen=True)
class NCAResult:
    cmax: float
    tmax: float
    auc_last: float
    auc_inf: float
    lambda_z: float
    half_life: float
    lambda_z_r2: float
    lambda_z_n_points: int
    aumc_inf: float
    mrt: float
    clearance_f: Optional[float]
    vz_f: Optional[float]

    def as_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def linear_trapezoid_auc(time: np.ndarray, conc: np.ndarray) -> float:
    """Cumulative AUC by the linear trapezoidal rule."""
    return float(_trapezoid(conc, time))


def _fit_lambda_z(time: np.ndarray, conc: np.ndarray,
                  min_points: int = 3) -> Tuple[float, float, int]:
    """Estimate the terminal rate constant by best-adjusted-R^2 regression.

    Tries terminal windows of increasing length (the last k positive-concentration
    points after Tmax) and keeps the slope from the window with the highest
    adjusted R^2, following the common NCA heuristic. Returns
    (lambda_z, r_squared, n_points).
    """
    tmax_idx = int(np.argmax(conc))
    # candidate terminal points: strictly after Tmax with positive concentration
    idx = [i for i in range(tmax_idx + 1, len(time)) if conc[i] > 0]
    if len(idx) < min_points:
        return float("nan"), float("nan"), 0

    t = time[idx]
    y = np.log(conc[idx])
    best = (float("nan"), -np.inf, 0)  # (lambda_z, adj_r2, n)
    for k in range(min_points, len(idx) + 1):
        tt, yy = t[-k:], y[-k:]
        slope, intercept = np.polyfit(tt, yy, 1)
        if slope >= 0:
            continue  # terminal phase must decline
        pred = slope * tt + intercept
        ss_res = float(((yy - pred) ** 2).sum())
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adj = 1.0 - (1.0 - r2) * (k - 1) / (k - 2) if k > 2 else r2
        if adj > best[1]:
            best = (-slope, adj, k)
    lambda_z, _, n = best
    # report the plain (not adjusted) R^2 for the chosen window
    if n:
        tt, yy = t[-n:], y[-n:]
        slope, intercept = np.polyfit(tt, yy, 1)
        pred = slope * tt + intercept
        ss_res = float(((yy - pred) ** 2).sum())
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        r2 = float("nan")
    return lambda_z, r2, n


def run_nca(time: Sequence[float], conc: Sequence[float],
            dose: Optional[float] = None) -> NCAResult:
    """Compute NCA parameters for one concentration-time profile."""
    t = np.asarray(time, dtype=float)
    c = np.asarray(conc, dtype=float)
    if t.shape != c.shape:
        raise ValueError("time and conc must have the same length.")
    if t.size < 3:
        raise ValueError("Need at least three samples for NCA.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must be strictly increasing.")
    if np.any(c < 0):
        raise ValueError("concentrations must be non-negative.")

    cmax = float(c.max())
    tmax = float(t[int(np.argmax(c))])
    auc_last = linear_trapezoid_auc(t, c)
    # first moment (t*C) for AUMC
    aumc_last = float(_trapezoid(t * c, t))

    lambda_z, r2, n = _fit_lambda_z(t, c)
    c_last = float(c[-1])
    if np.isfinite(lambda_z) and lambda_z > 0:
        auc_inf = auc_last + c_last / lambda_z
        # AUMC extrapolation: t_last*C_last/lambda_z + C_last/lambda_z^2
        aumc_inf = aumc_last + t[-1] * c_last / lambda_z + c_last / lambda_z ** 2
        half_life = float(np.log(2.0) / lambda_z)
    else:
        auc_inf = float("nan")
        aumc_inf = float("nan")
        half_life = float("nan")

    mrt = aumc_inf / auc_inf if auc_inf and np.isfinite(auc_inf) else float("nan")
    clearance_f = dose / auc_inf if (dose and np.isfinite(auc_inf)) else None
    vz_f = (clearance_f / lambda_z if (clearance_f is not None
            and np.isfinite(lambda_z) and lambda_z > 0) else None)

    return NCAResult(
        cmax=cmax, tmax=tmax, auc_last=auc_last, auc_inf=auc_inf,
        lambda_z=lambda_z, half_life=half_life, lambda_z_r2=r2,
        lambda_z_n_points=n, aumc_inf=aumc_inf, mrt=mrt,
        clearance_f=clearance_f, vz_f=vz_f,
    )
