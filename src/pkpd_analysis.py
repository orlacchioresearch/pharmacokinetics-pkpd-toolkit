"""End-to-end PK/PD analysis of the synthetic datasets.

Runs non-compartmental analysis and a one-compartment oral fit on the PK
profile, fits a sigmoidal Emax model to the PD data, writes parameter tables,
and saves a concentration-time plot and a concentration-effect plot.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from .nca import run_nca
    from .compartmental import fit_oral_first_order, conc_oral_first_order
    from .emax import fit_emax, emax_model
except ImportError:
    from nca import run_nca
    from compartmental import fit_oral_first_order, conc_oral_first_order
    from emax import fit_emax, emax_model

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PK = ROOT / "data" / "pk_concentration.csv"
PD = ROOT / "data" / "pd_effect.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DOSE_MG = 100.0


def plot_pk(t, c, fit, output: Path) -> None:
    ts = np.linspace(t.min(), t.max(), 300)
    cs = conc_oral_first_order(ts, DOSE_MG, fit.ka, fit.ke, fit.v_f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, scale in zip(axes, ("linear", "log")):
        ax.scatter(t, c, color="#C0392B", label="observed", zorder=3)
        ax.plot(ts, cs, color="#1f77b4", label="1-comp oral fit")
        ax.set_yscale(scale)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Concentration (mg/L)")
        ax.set_title(f"PK profile ({scale} scale)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    logger.info("Saved PK figure to %s", output)


def plot_pd(conc, effect, fit, output: Path) -> None:
    cs = np.linspace(0, conc.max(), 300)
    es = emax_model(cs, fit.e0, fit.emax, fit.ec50, fit.hill)
    plt.figure(figsize=(7, 5))
    plt.scatter(conc, effect, color="#C0392B", label="observed", zorder=3)
    plt.plot(cs, es, color="#1f77b4",
             label=f"Emax fit (EC50={fit.ec50:.2f}, h={fit.hill:.2f})")
    plt.axhline(fit.e0 + fit.emax / 2, color="grey", ls="--", lw=0.8)
    plt.axvline(fit.ec50, color="grey", ls=":", lw=0.8)
    plt.xlabel("Concentration (mg/L)")
    plt.ylabel("Effect")
    plt.title("PK/PD: sigmoidal Emax")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    logger.info("Saved PD figure to %s", output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    pk = pd.read_csv(PK)
    t = pk["time_h"].to_numpy(float)
    c = pk["concentration_mg_L"].to_numpy(float)

    nca = run_nca(t, c, dose=DOSE_MG)
    pd.DataFrame([nca.as_dict()]).to_csv(RESULTS / "nca_parameters.csv", index=False)

    cmp = fit_oral_first_order(t, c, DOSE_MG)
    pd.DataFrame([{
        "ka_per_h": round(cmp.ka, 4), "ke_per_h": round(cmp.ke, 4),
        "V_F_L": round(cmp.v_f, 3), "CL_F_L_per_h": round(cmp.cl_f, 4),
        "half_life_h": round(cmp.half_life, 3), "r_squared": round(cmp.r_squared, 5),
    }]).to_csv(RESULTS / "compartmental_fit.csv", index=False)
    plot_pk(t, c, cmp, FIGURES / "pk_profile.png")

    pdd = pd.read_csv(PD)
    conc = pdd["concentration_mg_L"].to_numpy(float)
    eff = pdd["effect"].to_numpy(float)
    pk_pd = fit_emax(conc, eff, sigmoid=True)
    pd.DataFrame([{
        "E0": round(pk_pd.e0, 3), "Emax": round(pk_pd.emax, 3),
        "EC50_mg_L": round(pk_pd.ec50, 3), "Hill": round(pk_pd.hill, 3),
        "r_squared": round(pk_pd.r_squared, 5),
    }]).to_csv(RESULTS / "emax_fit.csv", index=False)
    plot_pd(conc, eff, pk_pd, FIGURES / "pd_emax.png")

    print("\n=== Non-compartmental analysis ===")
    print(f"Cmax={nca.cmax:.3f} mg/L at Tmax={nca.tmax:.2f} h; "
          f"AUCinf={nca.auc_inf:.2f} mg·h/L; t½={nca.half_life:.2f} h; "
          f"CL/F={nca.clearance_f:.3f} L/h")
    print("\n=== One-compartment oral fit ===")
    print(f"ka={cmp.ka:.3f}/h  ke={cmp.ke:.3f}/h  V/F={cmp.v_f:.1f} L  "
          f"CL/F={cmp.cl_f:.3f} L/h  (R²={cmp.r_squared:.4f})")
    print("\n=== Sigmoidal Emax ===")
    print(f"E0={pk_pd.e0:.2f}  Emax={pk_pd.emax:.1f}  EC50={pk_pd.ec50:.2f} mg/L  "
          f"Hill={pk_pd.hill:.2f}  (R²={pk_pd.r_squared:.4f})")


if __name__ == "__main__":
    main()
