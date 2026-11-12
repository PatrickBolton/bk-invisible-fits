"""Validate the linear theory-uncertainty model against full toys.

The likelihoods describe the form factor and efficiency uncertainties of each binned
signal template as s + J u with u ~ N(0, 1) (lights.signal_rates.signal_template), which
implies the covariance J J^T. This script compares that with the covariance from full,
nonlinear toys (lights.signal_rates.dNdqsq_cov), for the SM and an example NP template.

    uv run python scripts/validate_theory_cov.py
"""

import numpy as np

from lights.load_data import eff_vals, eff_cov, eff_BaBar_K0_vals, eff_BaBar_K0_cov
from lights.signal_rates import dNdqsq_cov, Nrec_BelleII, Nrec_BaBar, signal_template
from lights.models import Model

NTOYS = 5000
SEED = 20261111

# (label, model, include_SM); the NP example is B -> K(*) phi phi-bar with a scalar coupling
cases = [("SM", None, True), ("NP (phiphi CSL, mX = 1.5)", Model("phiphi", 1.5, Lambda = 1e4, CSL = 1e-3), False)]

np.set_printoptions(precision = 3, linewidth = 200)
for name, Nrec, e_vals, e_cov in [("BelleII", Nrec_BelleII, eff_vals, eff_cov),
                                  ("BaBar", Nrec_BaBar, eff_BaBar_K0_vals, eff_BaBar_K0_cov)]:
    for label, model, include_SM in cases:
        _, C_toys = dNdqsq_cov(NTOYS, Nrec, model, include_SM, eff_vals = e_vals, eff_cov = e_cov, rng = SEED)
        s0, J = signal_template(name, model, include_SM)
        C_lin = J @ J.T

        sd_t, sd_l = np.sqrt(np.diag(C_toys)), np.sqrt(np.diag(C_lin))
        live = sd_t > 1e-6 * sd_t.max()
        corr_t = C_toys[np.ix_(live, live)] / np.outer(sd_t[live], sd_t[live])
        corr_l = C_lin[np.ix_(live, live)] / np.outer(sd_l[live], sd_l[live])
        print(f"\n{name}, {label}")
        print("  relative uncertainty (linear):", sd_l[live] / s0[live])
        print("  sd linear / sd toys          :", sd_l[live] / sd_t[live])
        print(f"  max |corr linear - corr toys|: {np.abs(corr_l - corr_t).max():.3f}")
