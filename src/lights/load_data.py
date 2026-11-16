"""Experimental inputs: Belle II (B+ -> K+ E_miss, ITA) and BaBar (B0 -> K*0 E_miss).

The data of each experiment is grouped in one object, BelleII or BaBar (see BelleIIData and
BaBarData for the fields), e.g. BelleII.obs, BaBar.eff_vals. BelleII_public holds Belle II's
public likelihood and its reinterpretation material (see BelleIIPublicData). All q^2 values
are in GeV^2.
"""

import json
from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline

from .constants import NB, mB
from .paths import BELLE2_PUBLIC_DIR
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


def eff_vals_gen(Ntoys, eff_vals, eff_cov, rng=None):
    """Random efficiency values, Gaussian with mean eff_vals and covariance eff_cov.

    rng: a seed or numpy.random.Generator, for reproducible toys (None: fresh randomness).
    """
    return np.random.default_rng(rng).multivariate_normal(eff_vals, eff_cov, size=Ntoys)


def _bins(edges):
    "Bin edges and centres."
    return edges, (edges[1:] + edges[:-1]) / 2


### Belle II B+ -> K+ E_miss, inclusive tag ###
@dataclass(frozen=True, eq=False)
class BelleIIData:
    """Belle II B+ -> K+ E_miss data, inclusive tag analysis (ITA).

    NB:             number of BB-bar pairs (the factor 2 f(+-) for signal-side B+ mesons is
                    assumed to be absorbed in the efficiencies)
    Nbins, bins, bin_cents:  number, edges and centres of the reconstructed q^2 bins
    obs:            observed events per bin
    b_cont, b_charged, b_neutral:  expected backgrounds (continuum, B+B-, B0B0-bar) per bin
    b_errors:       total MC uncertainty of the background per bin
    w_cont, w_charged, w_neutral:  each background's share of b_errors (they add in quadrature
                    to b_errors); in the likelihood the shift is w_k z_k with z_k ~ N(0, 1)
    eff_bins, eff_bin_cents:  true-q^2 bins of the signal efficiency
    eff_vals, eff_sigma, eff_cov:  signal efficiency, its uncertainty and (diagonal) covariance
    effHad_vals:    signal efficiency of the hadronic tag analysis (for comparison only)
    smearing_data:  q^2 smearing map, rows true q^2, columns reconstructed q^2, 1 GeV^2 bins
                    from -5 to 25 GeV^2 (see smearing() for the interpolated density)
    ita_profile:    published profile likelihood of the SM signal strength, columns mu, -2 Delta ln L
    """

    NB: float
    Nbins: int
    bins: np.ndarray
    bin_cents: np.ndarray
    obs: np.ndarray
    b_cont: np.ndarray
    b_charged: np.ndarray
    b_neutral: np.ndarray
    b_errors: np.ndarray
    w_cont: np.ndarray
    w_charged: np.ndarray
    w_neutral: np.ndarray
    eff_bins: np.ndarray
    eff_bin_cents: np.ndarray
    eff_vals: np.ndarray
    eff_sigma: np.ndarray
    eff_cov: np.ndarray
    effHad_vals: np.ndarray
    smearing_data: np.ndarray
    ita_profile: np.ndarray


def _load_BelleII(Nbins=20):
    bins, bin_cents = _bins(np.arange(-1.0, 21.0)[: Nbins + 1])

    b_cont = load(DATA_DIR / "bkg_inc_cont.txt")[:Nbins]
    b_charged = load(DATA_DIR / "bkg_inc_charged.txt")[:Nbins]
    b_neutral = load(DATA_DIR / "bkg_inc_neutral.txt")[:Nbins]
    b_errors = load(DATA_DIR / "errors_inc.txt")[:Nbins]
    b_tot = b_cont + b_charged + b_neutral

    # Efficiencies [%] per 2 GeV^2 bin, rows: value, lower and upper uncertainty (inclusive and hadronic tag)
    eff_data = load_json(DATA_DIR / "SignalEfficiency.json")
    effHad_data = load_json(DATA_DIR / "SignalEfficiency_HAD.json")
    eff_bins, eff_bin_cents = _bins(np.arange(0.0, 24.0 + 2, 2))
    eff_sigma = 1 / 100.0 * (eff_data[2] - eff_data[1]) / 2.0

    return BelleIIData(
        NB=NB,
        Nbins=Nbins,
        bins=bins,
        bin_cents=bin_cents,
        obs=load(DATA_DIR / "obs_inc.txt")[:Nbins],
        b_cont=b_cont,
        b_charged=b_charged,
        b_neutral=b_neutral,
        b_errors=b_errors,
        w_cont=b_errors * np.sqrt(b_cont / b_tot),
        w_charged=b_errors * np.sqrt(b_charged / b_tot),
        w_neutral=b_errors * np.sqrt(b_neutral / b_tot),
        eff_bins=eff_bins,
        eff_bin_cents=eff_bin_cents,
        eff_vals=eff_data[0] / 100.0,
        eff_sigma=eff_sigma,
        eff_cov=np.diag(eff_sigma**2),
        effHad_vals=effHad_data[0] / 100.0,
        smearing_data=load(DATA_DIR / "q2SmearingMapq2genvsq2rec.txt"),
        ita_profile=load_json(DATA_DIR / "ITALikelihoodPoints_PDF.json"),
    )


BelleII = _load_BelleII()

# Quadratic spline through the smearing map (1 GeV^2 bins centred at -4.5 ... 24.5), built once
_smearing_bin_cents = np.linspace(-4.5, 24.5, 30)
_smearing_interp = RectBivariateSpline(_smearing_bin_cents, _smearing_bin_cents, BelleII.smearing_data.T, kx=2, ky=2)


def smearing(qsqrec, qsqtrue):
    """Belle II q^2 smearing: probability density in reconstructed q^2 for each true q^2 (clipped at 0).

    qsqrec and qsqtrue are ascending arrays (or numbers); returns shape (len(qsqrec), len(qsqtrue)).
    """
    return np.maximum(_smearing_interp(qsqrec, qsqtrue), 0)


### BaBar B0 -> K*0 E_miss ###
@dataclass(frozen=True, eq=False)
class BaBarData:
    """BaBar B0 -> K*0 E_miss data (no q^2 smearing).

    NB:             number of BB-bar pairs (the factor 2 f(00) for signal-side B0 mesons is
                    assumed to be absorbed in the efficiencies)
    Nbins, bins, bin_cents:  number, edges and centres of the q^2 bins
                    (0.1 mB^2 wide, from 0 to 0.7 mB^2)
    obs:            observed events per bin
    b_tot, b_comb, b_peak:  expected total, combinatorial and peaking backgrounds per bin
    mc_factor:      ratio of MC to data statistics of the background
    b_sigma:        MC statistical uncertainty of b_tot, sqrt(b_tot / mc_factor)
    sigma_rec, sigma_NiB, sigma_Nnorm:  relative uncertainties of the reconstruction
                    efficiency, the peaking background and the background normalisation
    eff_bin_cents:  q^2 at which the signal efficiency is given (the bin centres)
    eff_vals, eff_sigma, eff_cov:  signal efficiency, its uncertainty and (diagonal) covariance
    """

    NB: float
    Nbins: int
    bins: np.ndarray
    bin_cents: np.ndarray
    obs: np.ndarray
    b_tot: np.ndarray
    b_comb: np.ndarray
    b_peak: np.ndarray
    mc_factor: float
    b_sigma: np.ndarray
    sigma_rec: float
    sigma_NiB: float
    sigma_Nnorm: float
    eff_bin_cents: np.ndarray
    eff_vals: np.ndarray
    eff_sigma: np.ndarray
    eff_cov: np.ndarray


def _load_BaBar(Nbins=7):
    bins, bin_cents = _bins(np.linspace(0, 0.7, 8)[: Nbins + 1] * mB**2)

    b_tot = load_json(DATA_DIR / "PeakBackgroundBabarB0Ks0_PDF.json")[:Nbins]
    b_comb = load_json(DATA_DIR / "CombBackgroundBabarB0Ks0_PDF.json")[:Nbins]
    mc_factor = 7.6

    # Efficiencies per bin, rows: value, lower and upper uncertainty
    eff_data = load_json(DATA_DIR / "EfficiencyB0toKs0BaBar_PDF.json")[:, :Nbins] * 8 * 1e-4
    eff_sigma = (eff_data[1] + eff_data[2]) / 2

    return BaBarData(
        NB=471e6,
        Nbins=Nbins,
        bins=bins,
        bin_cents=bin_cents,
        obs=load_json(DATA_DIR / "RateBaBarB0Ks0_PDF.json")[0][:Nbins],
        b_tot=b_tot,
        b_comb=b_comb,
        b_peak=b_tot - b_comb,
        mc_factor=mc_factor,
        b_sigma=np.sqrt(b_tot / mc_factor),
        sigma_rec=0.086,
        sigma_NiB=0.028,
        sigma_Nnorm=0.066,
        eff_bin_cents=bin_cents,
        eff_vals=eff_data[0],
        eff_sigma=eff_sigma,
        eff_cov=np.diag(eff_sigma**2),
    )


BaBar = _load_BaBar()


### Belle II public likelihood (arXiv:2507.12393, HEPData ins2947386) ###
def belle2_null_spectrum(qsq):
    """dB/dq^2(B+ -> K+ nu nu-bar) [GeV^-2] of the SM prediction Belle II generated its signal with.

    This is the null distribution of the released reinterpretation material (the joint number
    densities are binned signal MC of this spectrum): eq. 16 of arXiv:2207.13371 with the f+
    form factor of arXiv:2207.12468 (table V), as implemented in NullPrediction of
    https://github.com/lorenzennio/knunu-reinterpretation-mini (scripts/utils.py). Its constants
    differ slightly from lights.constants (e.g. mK = 0.495644 GeV, so the endpoint is at
    22.885 GeV^2), and it integrates to B = 4.966e-6.
    """
    GF, alpha_ew, Xt, sw2, Vtb_Vts = 1.1663787e-5, 1 / 127.952, 1.468, 0.23124, 0.04185
    m_B, m_K, m_Bs_star, tau_B = 5.279495, 0.4956440, 5.4158, 1.638e-12 / 6.58211e-25
    a, N, log_sB = (0.2545, -0.71, 0.32), 3, 1.304

    qsq = np.asarray(qsq, dtype=float)
    t_plus = (m_B + m_K) ** 2
    z = (np.sqrt(t_plus - qsq) - np.sqrt(t_plus)) / (np.sqrt(t_plus - qsq) + np.sqrt(t_plus))
    fplus = sum(a[n] * (z**n - n / N * (-1) ** (n - N) * z**N) for n in range(N)) / (1 - qsq / m_Bs_star**2) * log_sB
    lam = m_B**4 + m_K**4 + qsq**2 - 2 * (m_B**2 * m_K**2 + m_K**2 * qsq + m_B**2 * qsq)
    norm = GF**2 * alpha_ew**2 * Xt**2 / (32 * np.pi**5 * sw2**2) * tau_B * Vtb_Vts**2
    return norm * np.maximum(lam, 0) ** 1.5 / (2 * m_B) ** 3 * fplus**2


@dataclass(frozen=True, eq=False)
class BelleIIPublicData:
    """Belle II's public B+ -> K+ nu nu-bar likelihood and its reinterpretation material.

    workspace:   the pyhf (HistFactory JSON) workspace of the analysis, restricted to `channels`
    channels:    the workspace channels used (here the ITA: regions A and B, on- and off-resonance)
    q2gen_bins:  edges of the true-q^2 bins of the joint number densities: [-1, 0] for events
                 outside the kinematic range, then 100 equal bins up to 22.885 GeV^2
    nu0:         joint number density: expected SM signal events in each bin of the likelihood
                 (rows, in channel order; zero in channels without signal) from each true-q^2
                 bin (columns). The rows sum to the workspace's signal sample.
    null_binned: Belle II's SM branching fraction in each true-q^2 bin (belle2_null_spectrum;
                 the first, unphysical bin is 0)
    """

    workspace: dict
    channels: tuple
    q2gen_bins: np.ndarray
    nu0: np.ndarray
    null_binned: np.ndarray


def _load_BelleII_public():
    with open(BELLE2_PUBLIC_DIR / "combined_likelihood.json") as f:
        spec = json.load(f)
    # The ITA channels A (eta(BDT2) in [0.94, 1]) and B ([0.92, 0.94]), on- (Y4S) and off-resonance
    channels = tuple(ch["name"] for ch in spec["channels"] if ch["name"][:3] in ("A__", "B__", "C__", "D__"))
    workspace = {
        "channels": [ch for ch in spec["channels"] if ch["name"] in channels],
        "observations": [o for o in spec["observations"] if o["name"] in channels],
        "measurements": spec["measurements"],
        "version": spec["version"],
    }

    # Joint number densities of the signal regions, one row per bin of channels A and B
    maps = {"A__": "number_density_ita_regA.csv", "B__": "number_density_ita_regB.csv"}
    rows = []
    for ch in workspace["channels"]:
        nbins = len(ch["samples"][0]["data"])
        if ch["name"][:3] in maps:
            rows.append(load(BELLE2_PUBLIC_DIR / maps[ch["name"][:3]]).reshape(nbins, -1))
        else:
            rows.append(None)
    ncols = next(r for r in rows if r is not None).shape[1]
    nu0 = np.vstack(
        [
            r if r is not None else np.zeros((len(ch["samples"][0]["data"]), ncols))
            for r, ch in zip(rows, workspace["channels"])
        ]
    )

    q2gen_bins = np.r_[-1.0, np.linspace(0.0, 22.885, ncols)]
    lo, hi = q2gen_bins[1:-1], q2gen_bins[2:]
    q = np.linspace(lo, hi, 41, axis=1)
    null_binned = np.r_[0.0, simpson(belle2_null_spectrum(q), x=q, axis=1)]

    return BelleIIPublicData(workspace, channels, q2gen_bins, nu0, null_binned)


BelleII_public = _load_BelleII_public()
