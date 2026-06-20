"""One-compartment PK models and fitting (NumPy only).

Two routes:

* **IV bolus** -- ``C(t) = (Dose/V) * exp(-ke * t)``. Fitted by exact log-linear
  regression (the model is linear in log-concentration).
* **First-order oral absorption** -- the Bateman function
  ``C(t) = (F*Dose*ka)/(V*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))``. Fitted by
  Levenberg-Marquardt; ``F`` and ``V`` are confounded for extravascular data, so
  the apparent volume ``V/F`` is reported.

Derived parameters: half-life = ln2/ke, CL(/F) = ke * V(/F).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # numpy >= 2.0 renamed trapz -> trapezoid (trapz is removed in newer releases)
    from numpy import trapezoid as _trapezoid
except ImportError:  # numpy < 2.0
    from numpy import trapz as _trapezoid

try:
    from .least_squares import lm_fit
except ImportError:
    from least_squares import lm_fit


@dataclass(frozen=True)
class OneCompartmentFit:
    ka: float            # absorption rate (nan for IV)
    ke: float            # elimination rate
    v_f: float           # (apparent) volume of distribution
    cl_f: float          # (apparent) clearance = ke * v_f
    half_life: float
    r_squared: float
    route: str


def conc_iv_bolus(t, dose, ke, v):
    t = np.asarray(t, dtype=float)
    return dose / v * np.exp(-ke * t)


def conc_oral_first_order(t, dose, ka, ke, v_f):
    """Bateman one-compartment first-order-absorption concentration."""
    t = np.asarray(t, dtype=float)
    if abs(ka - ke) < 1e-12:                       # flip-flop limit
        return dose * ka / v_f * t * np.exp(-ke * t)
    return (dose * ka) / (v_f * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def fit_iv_bolus(time: Sequence[float], conc: Sequence[float], dose: float
                 ) -> OneCompartmentFit:
    """Exact log-linear fit of an IV-bolus profile."""
    t = np.asarray(time, dtype=float)
    c = np.asarray(conc, dtype=float)
    mask = c > 0
    slope, intercept = np.polyfit(t[mask], np.log(c[mask]), 1)
    ke = -float(slope)
    c0 = float(np.exp(intercept))
    v = dose / c0
    pred = slope * t[mask] + intercept
    ss_res = float(((np.log(c[mask]) - pred) ** 2).sum())
    ss_tot = float(((np.log(c[mask]) - np.log(c[mask]).mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return OneCompartmentFit(ka=float("nan"), ke=ke, v_f=v, cl_f=ke * v,
                             half_life=float(np.log(2) / ke), r_squared=r2,
                             route="iv_bolus")


def fit_oral_first_order(time: Sequence[float], conc: Sequence[float], dose: float
                         ) -> OneCompartmentFit:
    """Levenberg-Marquardt fit of a first-order oral-absorption profile."""
    t = np.asarray(time, dtype=float)
    c = np.asarray(conc, dtype=float)
    if t.size < 4:
        raise ValueError("Need at least four samples to fit ka, ke, V/F.")

    # Initial guesses: ke from the terminal log-linear slope; ka faster than ke;
    # V/F from a Dose/AUC-style scale.
    tmax_idx = int(np.argmax(c))
    term = slice(max(tmax_idx + 1, len(t) - 4), len(t))
    if np.sum(c[term] > 0) >= 2:
        slope, _ = np.polyfit(t[term], np.log(np.clip(c[term], 1e-12, None)), 1)
        ke0 = max(-float(slope), 1e-3)
    else:
        ke0 = 0.1
    ka0 = max(5.0 * ke0, 1.0)
    auc = float(_trapezoid(c, t))
    v_f0 = max(dose / (auc * ke0), 1e-6) if auc > 0 else dose

    def model(tt, p):
        ka, ke, v_f = p
        return conc_oral_first_order(tt, dose, abs(ka), abs(ke), abs(v_f))

    fit = lm_fit(model, t, c, [ka0, ke0, v_f0])
    ka, ke, v_f = np.abs(fit.params)
    if ka < ke:                                    # resolve flip-flop labelling
        ka, ke = ke, ka
    return OneCompartmentFit(ka=float(ka), ke=float(ke), v_f=float(v_f),
                             cl_f=float(ke * v_f), half_life=float(np.log(2) / ke),
                             r_squared=fit.r_squared, route="oral_first_order")
