"""NCA tests against hand calculations and a known mono-exponential profile."""
import math

import numpy as np
import pytest

from nca import linear_trapezoid_auc, run_nca


def test_trapezoidal_auc_hand_calculation():
    # triangle: up 0->2 over 1h, down 2->0 over 1h => area = 2
    assert math.isclose(linear_trapezoid_auc([0, 1, 2], [0, 2, 0]), 2.0)


def test_monoexponential_recovers_lambda_and_half_life():
    t = np.array([1, 2, 4, 6, 8, 10], dtype=float)
    c = 10.0 * np.exp(-0.2 * t)
    res = run_nca(t, c, dose=100.0)
    assert math.isclose(res.lambda_z, 0.2, rel_tol=1e-6)
    assert math.isclose(res.half_life, math.log(2) / 0.2, rel_tol=1e-6)


def test_cmax_and_tmax_are_observed_values():
    t = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    c = [0.0, 4.0, 6.5, 5.0, 2.0, 0.5]
    res = run_nca(t, c, dose=50.0)
    assert res.cmax == 6.5
    assert res.tmax == 1.0


def test_auc_inf_extrapolation_and_clearance():
    t = np.array([1, 2, 4, 6, 8, 10], dtype=float)
    c = 10.0 * np.exp(-0.2 * t)
    res = run_nca(t, c, dose=100.0)
    # AUClast + C_last/lambda_z, and CL/F = dose / AUCinf
    assert res.auc_inf > res.auc_last
    assert math.isclose(res.clearance_f, 100.0 / res.auc_inf, rel_tol=1e-9)


def test_requires_increasing_time():
    with pytest.raises(ValueError):
        run_nca([0, 2, 1], [1, 2, 3])
