"""Deterministically generate the synthetic PK and PD datasets.

PK: a single-subject oral one-compartment (Bateman) concentration-time profile.
PD: a concentration-effect relationship following a sigmoidal Emax model.
Re-running reproduces the CSVs exactly (fixed seed), keeping the committed data
in sync with the "true" parameters listed below.

True PK parameters:  Dose = 100 mg, ka = 1.2 /h, ke = 0.15 /h, V/F = 30 L
                     (=> CL/F = 4.5 L/h, t-half ≈ 4.6 h, Tmax ≈ 2.0 h)
True PD parameters:  E0 = 2, Emax = 80, EC50 = 1.5 mg/L, Hill = 1.8
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260620
DOSE_MG = 100.0
KA, KE, VF = 1.2, 0.15, 30.0
PD_E0, PD_EMAX, PD_EC50, PD_HILL = 2.0, 80.0, 1.5, 1.8


def _bateman(t, dose, ka, ke, v_f):
    return (dose * ka) / (v_f * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def generate_pk() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    t = np.array([0.0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24], dtype=float)
    c = _bateman(t, DOSE_MG, KA, KE, VF)
    c[0] = 0.0
    noise = rng.normal(1.0, 0.05, size=c.size)        # ~5% proportional error
    c = np.clip(c * noise, 0.0, None)
    return pd.DataFrame({"time_h": t, "concentration_mg_L": np.round(c, 4)})


def generate_pd() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    conc = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
    e = PD_E0 + PD_EMAX * conc ** PD_HILL / (PD_EC50 ** PD_HILL + conc ** PD_HILL)
    e = e + rng.normal(0.0, 1.5, size=e.size)          # additive measurement error
    return pd.DataFrame({"concentration_mg_L": conc, "effect": np.round(e, 3)})


def main() -> None:
    here = Path(__file__).resolve().parent
    pk = generate_pk()
    pd_df = generate_pd()
    pk.to_csv(here / "pk_concentration.csv", index=False)
    pd_df.to_csv(here / "pd_effect.csv", index=False)
    print(f"Wrote pk_concentration.csv ({len(pk)} samples) and "
          f"pd_effect.csv ({len(pd_df)} levels).")


if __name__ == "__main__":
    main()
