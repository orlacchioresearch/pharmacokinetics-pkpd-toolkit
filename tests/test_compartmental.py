"""Compartmental-fit tests: exact IV recovery and oral-fit parameter recovery."""
import math

import numpy as np

from compartmental import (conc_oral_first_order, fit_iv_bolus,
                           fit_oral_first_order)


def test_iv_bolus_recovers_parameters_exactly():
    dose, v, ke = 100.0, 10.0, 0.3
    t = np.array([0.5, 1, 2, 4, 6, 8], dtype=float)
    c = dose / v * np.exp(-ke * t)
    fit = fit_iv_bolus(t, c, dose)
    assert math.isclose(fit.ke, ke, rel_tol=1e-9)
    assert math.isclose(fit.v_f, v, rel_tol=1e-9)
    assert fit.r_squared > 0.999999


def test_iv_clearance_is_ke_times_volume():
    fit = fit_iv_bolus([0.5, 1, 2, 4, 6], 5.0 * np.exp(-0.25 * np.array([0.5, 1, 2, 4, 6])), 50.0)
    assert math.isclose(fit.cl_f, fit.ke * fit.v_f, rel_tol=1e-12)


def test_oral_first_order_recovers_known_parameters():
    dose, ka, ke, v_f = 100.0, 1.0, 0.2, 25.0
    t = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 18, 24], dtype=float)
    c = conc_oral_first_order(t, dose, ka, ke, v_f)
    fit = fit_oral_first_order(t, c, dose)
    assert math.isclose(fit.ka, ka, rel_tol=0.02)
    assert math.isclose(fit.ke, ke, rel_tol=0.02)
    assert math.isclose(fit.v_f, v_f, rel_tol=0.02)
    assert fit.r_squared > 0.999


def test_oral_half_life_consistent_with_ke():
    dose, ka, ke, v_f = 100.0, 1.5, 0.1, 40.0
    t = np.array([0.25, 0.5, 1, 2, 4, 6, 9, 12, 18, 24, 36], dtype=float)
    c = conc_oral_first_order(t, dose, ka, ke, v_f)
    fit = fit_oral_first_order(t, c, dose)
    assert math.isclose(fit.half_life, math.log(2) / fit.ke, rel_tol=1e-9)
