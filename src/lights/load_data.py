"""Experimental inputs: Belle II (B+ -> K+ E_miss, ITA) and BaBar (B0 -> K*0 E_miss)."""

import json

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .paths import BTOKEMISS_DIR as DATA_DIR  # Belle II and BaBar inputs


def load(file_name):
    "Numbers in a text file (whitespace- or comma-separated), as an array; a single column gives a 1D array."
    with open(file_name) as f:
        delimiter = "," if "," in f.readline() else None
    return np.squeeze(np.loadtxt(file_name, delimiter=delimiter, ndmin=2))


def load_json(file_name):
    """Numbers in a JSON file, as an array.

    A JSON object (keyed by bin) gives its values in file order, with lists of values as
    rows: {"bin 1": [a1, b1], "bin 2": [a2, b2]} -> [[a1, a2], [b1, b2]].
    """
    with open(file_name) as f:
        data = json.load(f)
    return np.array(list(data.values())).T if isinstance(data, dict) else np.array(data)


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
b_errors = load(DATA_DIR / "errors_inc.txt")[:Nbins]
_b_tot = b_cont + b_charged + b_neutral

# Background MC uncertainties, uncorrelated between bins: each background gets its share
# of the total error, w_k = b_errors * sqrt(b_k/b_tot), so that the w_k add in quadrature
# to b_errors. In the likelihood the shift is w_k * z_k with z_k ~ N(0, 1).
w_cont = b_errors * np.sqrt(b_cont / _b_tot)
w_charged = b_errors * np.sqrt(b_charged / _b_tot)
w_neutral = b_errors * np.sqrt(b_neutral / _b_tot)

obs = load(DATA_DIR / "obs_inc.txt")[:Nbins]

# Belle II's published profile likelihood of the SM signal strength (ITA); columns mu, -2 Delta ln L
ita_profile = load_json(DATA_DIR / "ITALikelihoodPoints_PDF.json")

# Efficiencies [%] per 2 GeV^2 bin, rows: value, lower and upper uncertainty (inclusive and hadronic tag)
eff_data = load_json(DATA_DIR / "SignalEfficiency.json")
effHad_data = load_json(DATA_DIR / "SignalEfficiency_HAD.json")

eff_bins = np.arange(0, 24 + 2, 2)
eff_bin_cents = (eff_bins[1:] + eff_bins[:-1]) / 2.0

eff_vals = eff_data[0] / 100.0
eff_sigma = 1 / 100.0 * (eff_data[2] - eff_data[1]) / 2.0
eff_cov = np.diag(eff_sigma**2)

effHad_vals = effHad_data[0] / 100.0


def eff_vals_gen(Ntoys, eff_vals, eff_cov, rng=None):
    """Generates random values for efficiencies.

    rng: a seed or numpy.random.Generator, for reproducible toys (None: fresh randomness).
    """
    eff_vals_r = np.random.default_rng(rng).multivariate_normal(eff_vals, eff_cov, size=Ntoys)

    return eff_vals_r


# Smearing
# Smearing map: rows true q^2, columns reconstructed q^2, 1 GeV^2 bins from -5 to 25 GeV^2
smearing_data = load(DATA_DIR / "q2SmearingMapq2genvsq2rec.txt")
_smearing_bin_cents = np.linspace(-4.5, 24.5, 30)

# Quadratic spline through the smearing map, built once
_smearing_interp = RectBivariateSpline(_smearing_bin_cents, _smearing_bin_cents, smearing_data.T, kx=2, ky=2)


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

b_BaBar_K0 = load_json(DATA_DIR / "PeakBackgroundBabarB0Ks0_PDF.json")[:Nbins_BaBar]  # total background
b_BaBar_comb_K0 = load_json(DATA_DIR / "CombBackgroundBabarB0Ks0_PDF.json")[:Nbins_BaBar]
b_BaBar_peak_K0 = b_BaBar_K0 - b_BaBar_comb_K0
b_BaBar_K0_sigma = np.sqrt(b_BaBar_K0 / MCFactorBack)  # MC statistical uncertainty

obs_BaBar_K0 = load_json(DATA_DIR / "RateBaBarB0Ks0_PDF.json")[0][:Nbins_BaBar]

# Efficiencies
eff_bins_BaBar = np.linspace(0, 0.7, 8)
eff_bin_cents_BaBar = (eff_bins_BaBar[1:] + eff_bins_BaBar[:-1]) / 2

eff_BaBar_K0_data = load_json(DATA_DIR / "EfficiencyB0toKs0BaBar_PDF.json") * 8 * 1e-4
eff_BaBar_K0_vals = eff_BaBar_K0_data[0]
eff_BaBar_K0_sigma = (eff_BaBar_K0_data[1] + eff_BaBar_K0_data[2]) / 2
eff_BaBar_K0_cov = np.diag(eff_BaBar_K0_sigma**2)
