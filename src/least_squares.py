"""A compact Levenberg-Marquardt nonlinear least-squares fitter (NumPy only).

Used to fit the compartmental PK and Emax PK/PD models without a SciPy
dependency. The Jacobian is computed by central finite differences, so any
Python callable ``model(x, params) -> y`` can be fitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FitResult:
    params: np.ndarray
    r_squared: float
    sse: float
    n_iter: int
    converged: bool

    def standard_errors(self) -> np.ndarray:
        return self._se

    _se: np.ndarray = None  # type: ignore


def _jacobian(model, x, params, y_shape, eps=1e-6):
    jac = np.zeros((y_shape, params.size))
    for k in range(params.size):
        step = eps * max(abs(params[k]), eps)
        up, dn = params.copy(), params.copy()
        up[k] += step
        dn[k] -= step
        jac[:, k] = (model(x, up) - model(x, dn)) / (2 * step)
    return jac


def lm_fit(model: Callable, x: Sequence[float], y: Sequence[float],
           p0: Sequence[float], max_iter: int = 200, tol: float = 1e-10
           ) -> FitResult:
    """Fit ``model(x, params)`` to ``y`` by Levenberg-Marquardt least squares."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p0, dtype=float).copy()

    def sse_of(params):
        r = y - model(x, params)
        return float(r @ r)

    lam = 1e-3
    sse = sse_of(p)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        jac = _jacobian(model, x, p, y.size)
        resid = y - model(x, p)
        jtj = jac.T @ jac
        jtr = jac.T @ resid
        improved = False
        for _ in range(30):  # adjust damping until a step improves the fit
            try:
                step = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj)), jtr)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            new_sse = sse_of(p + step)
            if new_sse < sse:
                p = p + step
                improved = True
                if abs(sse - new_sse) < tol * max(sse, 1e-30):
                    converged = True
                sse = new_sse
                lam = max(lam / 3.0, 1e-12)
                break
            lam *= 3.0
        if converged or not improved:
            break

    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
    dof = max(y.size - p.size, 1)
    jac = _jacobian(model, x, p, y.size)
    try:
        cov = (sse / dof) * np.linalg.inv(jac.T @ jac)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(p.size, np.nan)

    res = FitResult(params=p, r_squared=r2, sse=sse, n_iter=it, converged=converged)
    object.__setattr__(res, "_se", se)
    return res
