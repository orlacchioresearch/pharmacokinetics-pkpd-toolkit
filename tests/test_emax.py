"""Emax PK/PD tests: the half-maximal property and parameter recovery."""
import math

import numpy as np

from emax import emax_model, fit_emax


def test_effect_at_ec50_is_half_maximal():
    e = emax_model(2.0, e0=5.0, emax=80.0, ec50=2.0, hill=1.0)
    assert math.isclose(e, 5.0 + 40.0)


def test_hill_slope_steepens_the_curve():
    low = emax_model(1.0, 0, 100, 2.0, hill=1.0)
    steep = emax_model(1.0, 0, 100, 2.0, hill=3.0)
    # below EC50 a steeper Hill slope gives a smaller effect
    assert steep < low


def test_recovers_known_simple_emax():
    conc = np.array([0, 0.25, 0.5, 1, 2, 4, 8, 16], dtype=float)
    e = emax_model(conc, e0=3.0, emax=90.0, ec50=2.0, hill=1.0)
    fit = fit_emax(conc, e, sigmoid=False)
    assert math.isclose(fit.ec50, 2.0, rel_tol=1e-3)
    assert math.isclose(fit.emax, 90.0, rel_tol=1e-3)
    assert math.isclose(fit.e0, 3.0, abs_tol=1e-2)


def test_recovers_known_sigmoid_emax():
    conc = np.array([0, 0.1, 0.3, 0.5, 1, 1.5, 2, 3, 5, 8, 12], dtype=float)
    e = emax_model(conc, e0=2.0, emax=80.0, ec50=1.5, hill=1.8)
    fit = fit_emax(conc, e, sigmoid=True)
    assert math.isclose(fit.ec50, 1.5, rel_tol=1e-2)
    assert math.isclose(fit.hill, 1.8, rel_tol=1e-2)
    assert fit.r_squared > 0.999
