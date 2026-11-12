import csv

import numpy as np
import pandas as pd
from numpy.linalg import inv, multi_dot
from scipy.interpolate import RectBivariateSpline

from .paths import BTOKEMISS_DIR as DATA_DIR  # Belle II and BaBar inputs


def sq(x):
    return np.squeeze(np.array(x))


def load(file_name):
    return sq(pd.read_table(file_name, header=None))


def load_s(file_name):
    return np.array(pd.io.json.read_json(file_name, typ="series"))


def read_csv_to_array(file_path):
    data = []
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Convert each element in the row to float and
            #  append to the data list
            data.append([float(element) for element in row])
    return data


def corr_mat(M):
    """Returns the correlation matrix corresponding to the covariance matrix M."""
    return multi_dot([np.diag(np.sqrt(1 / np.diag(M))), M, np.diag(np.sqrt(1 / np.diag(M)))])


def factorise(C, rtol=1e-6):
    """Factor a (possibly singular) covariance matrix as C = L L^T.

    Uses the eigendecomposition C = U diag(lam) U^T and returns L = U diag(sqrt(lam)),
    an N x N matrix. Modes with lam < rtol * max(lam) carry no meaningful variance
    (they are numerically zero) and get zero columns, so the corresponding nuisance
    parameters do not affect the prediction. A Gaussian shift delta ~ N(0, C) is then
    delta = L z with z ~ N(0, 1), and its penalty is simply z.z, so C is never inverted.
    """
    lam, U = np.linalg.eigh(C)
    keep = lam > rtol * lam.max()
    return np.ascontiguousarray(U * np.where(keep, np.sqrt(np.abs(lam)), 0.0))


### Belle-II B+ -> K+ Inclusive Data ###
Nbins = 20
bins = np.arange(-1, 21)[: Nbins + 1]
bin_cents = (bins[1:] + bins[:-1]) / 2
bw = (bins[1] - bins[0]) / 2

b_cont = load(DATA_DIR / "bkg_inc_cont.txt")[:Nbins]
b_charged = load(DATA_DIR / "bkg_inc_charged.txt")[:Nbins]
b_neutral = load(DATA_DIR / "bkg_inc_neutral.txt")[:Nbins]
b_tot = b_cont + b_charged + b_neutral
b_errors = load(DATA_DIR / "errors_inc.txt")[:Nbins]

# Background MC uncertainties, uncorrelated between bins: each background gets its share
# of the total error, w_k = b_errors * sqrt(b_k/b_tot), so that the w_k add in quadrature
# to b_errors. In the likelihood the shift is w_k * z_k with z_k ~ N(0, 1).
w_cont = b_errors * np.sqrt(b_cont / b_tot)
w_charged = b_errors * np.sqrt(b_charged / b_tot)
w_neutral = b_errors * np.sqrt(b_neutral / b_tot)

cov_cont = np.diag(w_cont**2)
cov_charged = np.diag(w_charged**2)
cov_neutral = np.diag(w_neutral**2)

obs = load(DATA_DIR / "obs_inc.txt")[:Nbins]

# Efficiencies
eff_data = np.array(pd.io.json.read_json(DATA_DIR / "SignalEfficiency.json"))
effHad_data = np.array(pd.io.json.read_json(DATA_DIR / "SignalEfficiency_HAD.json"))

eff_bins = np.arange(0, 24 + 2, 2)
eff_bin_cents = (eff_bins[1:] + eff_bins[:-1]) / 2.0

eff_vals = eff_data[0] / 100.0
eff_sigma = 1 / 100.0 * (eff_data[2] - eff_data[1]) / 2.0
eff_cov = np.diag(eff_sigma**2)

effHad_vals = effHad_data[0] / 100.0
effHad_sigma = 1 / 100.0 * (effHad_data[2] - effHad_data[1]) / 2.0
effHad_cov = np.diag(effHad_sigma**2)


def eff_vals_gen(Ntoys, eff_vals, eff_cov, rng=None):
    """Generates random values for efficiencies.

    rng: a seed or numpy.random.Generator, for reproducible toys (None: fresh randomness).
    """
    eff_vals_r = np.random.default_rng(rng).multivariate_normal(eff_vals, eff_cov, size=Ntoys)

    return eff_vals_r


# Smearing
smearing_data = np.array(pd.read_table(DATA_DIR / "q2SmearingMapq2genvsq2rec.txt", header=None))
qsqrec_bins = np.linspace(-5, 25, 31)
qsqtrue_bins = np.linspace(-5, 25, 31)
qsqrec_bin_cents = (qsqrec_bins[1:] + qsqrec_bins[:-1]) / 2
qsqtrue_bin_cents = (qsqtrue_bins[1:] + qsqtrue_bins[:-1]) / 2

# Quadratic spline through the smearing map, built once
_smearing_interp = RectBivariateSpline(qsqrec_bin_cents, qsqtrue_bin_cents, np.transpose(smearing_data), kx=2, ky=2)


def sm(qsqrec, qsqtrue):
    """Smearing interpolation function for Belle II (probability density in q2_rec, clipped at 0)."""
    return np.maximum(_smearing_interp(qsqrec, qsqtrue), 0)


### BaBar B0 -> K*0 data ###
# Number of BB-bar pairs at BaBar. The number of signal-side B0 mesons would be
# 2 * f(00) * NB_BaBar; that factor is assumed to be absorbed in the signal efficiencies.
NB_BaBar = 471e6

Nbins_BaBar = 7
bins_BaBar = np.linspace(0, 0.7, 8)[: Nbins_BaBar + 1]
bin_cents_BaBar = (bins_BaBar[1:] + bins_BaBar[:-1]) / 2
bw_BaBar = (bins_BaBar[1] - bins_BaBar[0]) / 2


sigrecB0Ks0 = 0.086
NiBs = 0.028
NinormB0Ks0 = 0.066
MCFactorBack = 7.6

b_BaBar_K0 = load_s(DATA_DIR / "PeakBackgroundBabarB0Ks0_PDF.json")[:Nbins_BaBar]
b_BaBar_comb_K0 = load_s(DATA_DIR / "CombBackgroundBabarB0Ks0_PDF.json")[:Nbins_BaBar]
b_BaBar_peak_K0 = b_BaBar_K0 - b_BaBar_comb_K0
b_BaBar_K0_sigma = np.sqrt(b_BaBar_K0 / MCFactorBack)
b_BaBar_K0_cov = np.diag(b_BaBar_K0_sigma**2)
b_BaBar_K0_invcov = inv(b_BaBar_K0_cov)

rate_BaBar_data = np.array(pd.io.json.read_json(DATA_DIR / "RateBaBarB0Ks0_PDF.json"))
obs_BaBar_K0 = rate_BaBar_data[0][:Nbins_BaBar]
obs_BaBar_down_K0 = rate_BaBar_data[1][:Nbins_BaBar]
obs_BaBar_up_K0 = rate_BaBar_data[2][:Nbins_BaBar]
asymmetric_error_K0 = np.array(list(zip(obs_BaBar_down_K0, obs_BaBar_up_K0))).T

# Efficiencies
eff_bins_BaBar = np.linspace(0, 0.7, 8)
eff_bin_cents_BaBar = (eff_bins_BaBar[1:] + eff_bins_BaBar[:-1]) / 2

eff_BaBar_K0_data = np.array(pd.io.json.read_json(DATA_DIR / "EfficiencyB0toKs0Babar_PDF.json")) * 8 * 1e-4
eff_BaBar_K0_vals = eff_BaBar_K0_data[0]
eff_BaBar_K0_sigma = (eff_BaBar_K0_data[1] + eff_BaBar_K0_data[2]) / 2
eff_BaBar_K0_cov = np.diag(eff_BaBar_K0_sigma**2)
eff_BaBar_K0_invcov = inv(eff_BaBar_K0_cov)
