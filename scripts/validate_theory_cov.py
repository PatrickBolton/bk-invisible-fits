"""Validate the linear theory-uncertainty model against full toys.

The likelihoods describe the form factor and efficiency uncertainties of each binned
signal template as s + J u with u ~ N(0, 1) (lights.signal_rates.signal_template), which
implies the covariance J J^T. This script compares that with the covariance from full,
nonlinear toys (lights.signal_rates.dNdqsq_cov), for the SM and an example NP template.

    uv run python scripts/validate_theory_cov.py
"""

import numpy as np
import numpy.random as npr

from lights.load_data import eff_vals, eff_cov, eff_BaBar_K0_vals, eff_BaBar_K0_cov
from lights.signal_rates import dNdqsq_cov, Nrec_BelleII, Nrec_BaBar, signal_template
from lights.branching_fractions import cdict

NTOYS = 5000
SEED = 20261111

c_NP = np.zeros((6, 14))
c_NP[cdict['CSL_ss']] = 1e-3     # example: B -> K(*) phi phi, scalar coupling
cases = [("SM", 1, 1., np.zeros((6, 14)), 1.), ("NP (CSL_ss, mX = 1.5)", 0, 1.5, c_NP, 1e4)]

np.set_printoptions(precision = 3, linewidth = 200)
for name, Nrec, e_vals, e_cov in [("BelleII", Nrec_BelleII, eff_vals, eff_cov),
                                  ("BaBar", Nrec_BaBar, eff_BaBar_K0_vals, eff_BaBar_K0_cov)]:
    for label, SM, mX, c, Lambda in cases:
        npr.seed(SEED)
        _, C_toys = dNdqsq_cov(NTOYS, Nrec, SM, mX, c, Lambda, eff_vals = e_vals, eff_cov = e_cov)
        s0, J = signal_template(name, SM, mX, c, Lambda)
        C_lin = J @ J.T

        sd_t, sd_l = np.sqrt(np.diag(C_toys)), np.sqrt(np.diag(C_lin))
        live = sd_t > 1e-6 * sd_t.max()
        corr_t = C_toys[np.ix_(live, live)] / np.outer(sd_t[live], sd_t[live])
        corr_l = C_lin[np.ix_(live, live)] / np.outer(sd_l[live], sd_l[live])
        print(f"\n{name}, {label}")
        print("  relative uncertainty (linear):", sd_l[live] / s0[live])
        print("  sd linear / sd toys          :", sd_l[live] / sd_t[live])
        print(f"  max |corr linear - corr toys|: {np.abs(corr_l - corr_t).max():.3f}")
