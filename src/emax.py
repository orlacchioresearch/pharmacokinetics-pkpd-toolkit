"""Emax and sigmoidal (Hill) Emax PK/PD models (NumPy only).

Relates drug concentration to effect:

    simple Emax    E = E0 + Emax * C / (EC50 + C)
    sigmoid Emax   E = E0 + Emax * C^h / (EC50^h + C^h)

``EC50`` is the concentration at half-maximal effect and ``h`` the Hill slope.
Fitting is by Levenberg-Marquardt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from .least_squares import lm_fit
except ImportError:
    from least_squares import lm_fit


@dataclass(frozen=True)
class EmaxFit:
    e0: float
    emax: float
    ec50: float
    hill: float
    r_squared: float
    sigmoid: bool


def emax_model(conc, e0, emax, ec50, hill=1.0):
    c = np.asarray(conc, dtype=float)
    ch = np.power(c, hill)
    return e0 + emax * ch / (np.power(ec50, hill) + ch)


def fit_emax(conc: Sequence[float], effect: Sequence[float], sigmoid: bool = True
             ) -> EmaxFit:
    """Fit a (sigmoidal) Emax model of effect vs concentration."""
    c = np.asarray(conc, dtype=float)
    e = np.asarray(effect, dtype=float)
    if c.shape != e.shape:
        raise ValueError("conc and effect must have the same length.")
    if np.any(c < 0):
        raise ValueError("concentrations must be non-negative.")

    e0_0 = float(e[np.argmin(c)]) if c.size else 0.0
    emax_0 = float(e.max() - e0_0) or 1.0
    pos = c[c > 0]
    ec50_0 = float(np.median(pos)) if pos.size else 1.0

    if sigmoid:
        def model(cc, p):
            e0, emax, ec50, hill = p
            return emax_model(cc, e0, emax, abs(ec50), abs(hill))
        fit = lm_fit(model, c, e, [e0_0, emax_0, ec50_0, 1.0])
        e0, emax, ec50, hill = fit.params
        ec50, hill = abs(ec50), abs(hill)
    else:
        def model(cc, p):
            e0, emax, ec50 = p
            return emax_model(cc, e0, emax, abs(ec50), 1.0)
        fit = lm_fit(model, c, e, [e0_0, emax_0, ec50_0])
        e0, emax, ec50 = fit.params
        ec50, hill = abs(ec50), 1.0

    return EmaxFit(e0=float(e0), emax=float(emax), ec50=float(ec50),
                   hill=float(hill), r_squared=fit.r_squared, sigmoid=sigmoid)
