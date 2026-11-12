from typing import NamedTuple

import numba as nb
import numpy as np
from iminuit import Minuit
from joblib import Parallel, delayed
from tqdm import tqdm

from lights.load_data import (
    Nbins,
    Nbins_BaBar,
    NiBs,
    NinormB0Ks0,
    b_BaBar_comb_K0,
    b_BaBar_K0,
    b_BaBar_K0_invcov,
    b_BaBar_peak_K0,
    b_charged,
    b_cont,
    b_neutral,
    obs,
    obs_BaBar_K0,
    sigrecB0Ks0,
    w_charged,
    w_cont,
    w_neutral,
)
from lights.models import B_Bs, Model, from_legacy
from lights.signal_rates import (
    J_SM_BaBar,
    J_SM_BelleII,
    Nrec_BaBar,  # noqa: F401
    Nrec_BelleII,  # noqa: F401  (re-exported: importable from lights.likelihoods)
    NU_BaBar,
    NU_BelleII,
    s_SM_BaBar,
    s_SM_BelleII,
    signal_template,
)

kwd = {"fastmath": {"reassoc", "contract", "arcp"}}

CNP_zero = np.zeros((6, 14))


def _relative(s, J):
    "Relative response R = J / s of a template (zero where the template is zero)."
    s = np.asarray(s, dtype=float)
    safe = np.where(s > 0, s, 1.0)
    return np.ascontiguousarray(np.where(s[:, None] > 0, J / safe[:, None], 0.0))


# Relative responses of the SM templates to the input nuisance params.
R_SM_BelleII = _relative(s_SM_BelleII, J_SM_BelleII)
R_SM_BaBar = _relative(s_SM_BaBar, J_SM_BaBar)


@nb.njit(**kwd)
def sigdot(x, y):
    "Useful function entering gaussian log likelihoods."
    return x @ y @ x


# Parameter layout
# ----------------
# x[0]        mu_SM, SM signal strength
# x[1], x[2]  NP couplings; the NP signal strengths are x[1]**2 and x[2]**2
# x[3:]       nuisance parameters (one block per experiment, see below)
#
# The NP templates are passed to the likelihoods as arguments, not as fit parameters:
# each template is a binned signal s_NP and its relative response R_NP = (ds_NP/du) / s_NP.
#
# Theory uncertainties use shared input nuisance parameters u ~ N(0, 1): the whitened form
# factor coefficients and efficiency points of each experiment (signal_rates.signal_template).
# Every signal template responds log-normally, s * exp(R u), and the penalty is u.u. To first
# order this is s + J u (J = ds/du), but the prediction stays positive however large u is,
# so the nuisances cannot cancel a large signal. Because the same
# u enter the SM and NP templates, their theory uncertainties are correctly correlated, and
# no covariance matrix is ever inverted.

# Belle II ITA
# Nuisance block p (length NPAR_BelleII), with U = NU_BelleII:
#   p[0:U]            u          input nuisance params. (B -> K form factors, efficiencies)
#   p[U:U+N]          z_cont     background MC modes, shift w_cont * z_cont (uncorrelated)
#   p[U+N:U+2N]       z_charged
#   p[U+2N:U+3N]      z_neutral
#   p[U+3N:U+3N+3]    mu_cont, mu_charged, mu_neutral   background normalisation params.
NPAR_BelleII = NU_BelleII + 3 * Nbins + 3

# Relative uncertainty of each Belle II background normalisation (mu_cont, mu_charged, mu_neutral)
BKG_NORM_SIGMA = 0.02


@nb.njit(**kwd)
def _exp_BelleII(mu_SM, mu_NP_1, mu_NP_2, s_NP_1, s_NP_2, R_NP_1, R_NP_2, p):
    "Expected number of Belle II ITA events in each bin."
    N, U = Nbins, NU_BelleII
    u = p[0:U]
    z_cont, z_charged, z_neutral = p[U : U + N], p[U + N : U + 2 * N], p[U + 2 * N : U + 3 * N]
    mu_cont, mu_charged, mu_neutral = p[U + 3 * N], p[U + 3 * N + 1], p[U + 3 * N + 2]

    return (
        mu_SM * s_SM_BelleII * np.exp(R_SM_BelleII @ u)
        + mu_NP_1 * s_NP_1 * np.exp(R_NP_1 @ u)
        + mu_NP_2 * s_NP_2 * np.exp(R_NP_2 @ u)
        + (1 + mu_cont) * (b_cont + w_cont * z_cont)
        + (1 + mu_charged) * (b_charged + w_charged * z_charged)
        + (1 + mu_neutral) * (b_neutral + w_neutral * z_neutral)
    )


@nb.njit(**kwd)
def _penalty_BelleII(p):
    "Gaussian constraint terms for the Belle II nuisance parameters."
    N, U = Nbins, NU_BelleII
    u, z = p[0:U], p[U : U + 3 * N]
    mu_cont, mu_charged, mu_neutral = p[U + 3 * N], p[U + 3 * N + 1], p[U + 3 * N + 2]

    sigma = BKG_NORM_SIGMA
    normgauss = (mu_cont / sigma) ** 2 + (mu_charged / sigma) ** 2 + (mu_neutral / sigma) ** 2

    return u @ u + z @ z + normgauss


@nb.njit(**kwd)
def Nevents_BelleII(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "Expected number of Belle II ITA events: SM plus NP B+ -> K+ invisible signal and backgrounds."
    return _exp_BelleII(x[0], x[1] ** 2, x[2] ** 2, s_NP_1, s_NP_2, R_NP_1, R_NP_2, x[3:])


@nb.njit(**kwd)
def logLL_BelleII(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "-2 times the log likelihood (minus constant factors) for the Belle II inclusive tag."
    exp = Nevents_BelleII(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2)
    pois = 2 * np.sum(exp - obs * np.log(exp))  # Poisson (exp, obs) over bins
    return pois + _penalty_BelleII(x[3:])


# BaBar
# Nuisance block p (length NPAR_BaBar), with U = NU_BaBar:
#   p[0:U]          u           input nuisance params. (B -> K* form factors, efficiencies)
#   p[U:U+Nb]       th_bkg      statistical background (MC) nuisance params.
#   p[U+Nb]         recB0Ks0    reconstruction nuisance param.
#   p[U+Nb+1]       NiBvar      NiB nuisance param.
#   p[U+Nb+2]       NinormB0Ks0var   background normalisation nuisance param.
NPAR_BaBar = NU_BaBar + Nbins_BaBar + 3


@nb.njit(**kwd)
def _exp_BaBar(mu_SM, mu_NP_1, mu_NP_2, s_NP_1, s_NP_2, R_NP_1, R_NP_2, p):
    "Expected number of BaBar B0 -> K*0 invisible events in each bin."
    N, U = Nbins_BaBar, NU_BaBar
    u, th_bkg = p[0:U], p[U : U + N]
    recB0Ks0, NiBvar, NinormB0Ks0var = p[U + N], p[U + N + 1], p[U + N + 2]

    signal = (
        mu_SM * s_SM_BaBar * np.exp(R_SM_BaBar @ u)
        + mu_NP_1 * s_NP_1 * np.exp(R_NP_1 @ u)
        + mu_NP_2 * s_NP_2 * np.exp(R_NP_2 @ u)
    )
    return (1 + recB0Ks0) * signal + (1 + NinormB0Ks0var) * (1 + th_bkg) * (
        (1 + recB0Ks0) * (1 + NiBvar) * b_BaBar_peak_K0 + b_BaBar_comb_K0
    )


@nb.njit(**kwd)
def _penalty_BaBar(p):
    "Gaussian constraint terms for the BaBar nuisance parameters."
    N, U = Nbins_BaBar, NU_BaBar
    u, th_bkg = p[0:U], p[U : U + N]
    recB0Ks0, NiBvar, NinormB0Ks0var = p[U + N], p[U + N + 1], p[U + N + 2]

    MCgauss = sigdot(th_bkg * b_BaBar_K0, b_BaBar_K0_invcov)
    normgauss = (recB0Ks0 / sigrecB0Ks0) ** 2 + (NiBvar / NiBs) ** 2 + (NinormB0Ks0var / NinormB0Ks0) ** 2

    return u @ u + MCgauss + normgauss


@nb.njit(**kwd)
def Nevents_BaBar(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "Expected number of BaBar events: SM plus NP B0 -> K*0 invisible signal and backgrounds."
    return _exp_BaBar(x[0], x[1] ** 2, x[2] ** 2, s_NP_1, s_NP_2, R_NP_1, R_NP_2, x[3:])


@nb.njit(**kwd)
def logLL_BaBar(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "-2 times the log likelihood (minus constant factors) for BaBar."
    exp = Nevents_BaBar(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2)
    pois = 2 * np.sum(exp - obs_BaBar_K0 * np.log(exp))  # Poisson (exp, obs) over bins
    return pois + _penalty_BaBar(x[3:])


# LEP (Bs -> inv)
LEP_N_norm = 6607.14
LEP_mu = 0.525241
LEP_sigma = 2.47728


@nb.njit(**kwd)
def logLL_LEP(x):
    "-2 times the log likelihood (minus constant factors) for LEP."

    mu_NP_1 = (x[0]) ** 2  # NP signal 1 strength
    mu_NP_2 = (x[1]) ** 2  # NP signal 2 strength

    Br_LEP_1 = x[2]
    Br_LEP_2 = x[3]

    Br_LEP = mu_NP_1 * Br_LEP_1 + mu_NP_2 * Br_LEP_2

    # N_norm = 6296.296296296295
    # mu_LEP = -0.12751
    # sigma_LEP = 2.75253

    N_norm, mu_LEP, sigma_LEP = LEP_N_norm, LEP_mu, LEP_sigma

    # LL_LEP = ((N_norm*Br_LEP)/sigma_LEP)**2 - 2*(N_norm*Br_LEP)*mu_LEP/sigma_LEP**2

    LL_LEP = (N_norm * Br_LEP - mu_LEP) ** 2 / sigma_LEP**2

    return LL_LEP


# Combined Likelihoods
# Layout: x[0:3] as above, then the Belle II nuisance block, then the BaBar nuisance block.
# The Belle II and BaBar inputs (B -> K vs B -> K* form factors, separate efficiencies) are
# independent, so the two blocks share no parameters apart from mu_SM and the NP couplings.
@nb.njit(**kwd)
def logLL_Combined(
    x,
    s_NP_1_BelleII,
    s_NP_2_BelleII,
    R_NP_1_BelleII,
    R_NP_2_BelleII,
    s_NP_1_BaBar,
    s_NP_2_BaBar,
    R_NP_1_BaBar,
    R_NP_2_BaBar,
):
    "-2 times the log likelihood (minus constant factors) for combined experiments."
    mu_SM, mu_NP_1, mu_NP_2 = x[0], x[1] ** 2, x[2] ** 2
    pB = 3  # start of Belle II nuisance block
    pb = 3 + NPAR_BelleII  # start of BaBar nuisance block

    exp_BelleII = _exp_BelleII(
        mu_SM, mu_NP_1, mu_NP_2, s_NP_1_BelleII, s_NP_2_BelleII, R_NP_1_BelleII, R_NP_2_BelleII, x[pB:pb]
    )
    exp_BaBar = _exp_BaBar(mu_SM, mu_NP_1, mu_NP_2, s_NP_1_BaBar, s_NP_2_BaBar, R_NP_1_BaBar, R_NP_2_BaBar, x[pb:])

    return (
        2 * np.sum(exp_BelleII - obs * np.log(exp_BelleII))
        + _penalty_BelleII(x[pB:pb])
        + 2 * np.sum(exp_BaBar - obs_BaBar_K0 * np.log(exp_BaBar))
        + _penalty_BaBar(x[pb:])
    )


# Analytic gradients
# -------------------
# Migrad otherwise estimates the gradient numerically, costing two likelihood calls per free
# parameter per step. With f = 2 sum(e - o ln e) + penalties, df/de = 2 (1 - o/e), and the
# chain rule through e gives the gradient below. Checked against finite differences in
# scripts/check_gradients.py.


@nb.njit(**kwd)
def _grad_BelleII(mu_SM, mu_NP_1, mu_NP_2, s_NP_1, s_NP_2, R_NP_1, R_NP_2, p):
    """Gradient of the Belle II -2 ln L w.r.t. (mu_SM, mu_NP_1, mu_NP_2) and the nuisance block p."""
    N, U = Nbins, NU_BelleII
    u = p[0:U]
    z_cont, z_charged, z_neutral = p[U : U + N], p[U + N : U + 2 * N], p[U + 2 * N : U + 3 * N]
    mu_cont, mu_charged, mu_neutral = p[U + 3 * N], p[U + 3 * N + 1], p[U + 3 * N + 2]

    a0 = s_SM_BelleII * np.exp(R_SM_BelleII @ u)
    a1 = s_NP_1 * np.exp(R_NP_1 @ u)
    a2 = s_NP_2 * np.exp(R_NP_2 @ u)
    B_cont = b_cont + w_cont * z_cont
    B_charged = b_charged + w_charged * z_charged
    B_neutral = b_neutral + w_neutral * z_neutral
    e = (
        mu_SM * a0
        + mu_NP_1 * a1
        + mu_NP_2 * a2
        + (1 + mu_cont) * B_cont
        + (1 + mu_charged) * B_charged
        + (1 + mu_neutral) * B_neutral
    )
    g = 2 * (1 - obs / e)

    dp = np.zeros(NPAR_BelleII)
    dp[0:U] = R_SM_BelleII.T @ (g * mu_SM * a0) + R_NP_1.T @ (g * mu_NP_1 * a1) + R_NP_2.T @ (g * mu_NP_2 * a2) + 2 * u
    dp[U : U + N] = g * (1 + mu_cont) * w_cont + 2 * z_cont
    dp[U + N : U + 2 * N] = g * (1 + mu_charged) * w_charged + 2 * z_charged
    dp[U + 2 * N : U + 3 * N] = g * (1 + mu_neutral) * w_neutral + 2 * z_neutral
    sigma = BKG_NORM_SIGMA
    dp[U + 3 * N] = g @ B_cont + 2 * mu_cont / sigma**2
    dp[U + 3 * N + 1] = g @ B_charged + 2 * mu_charged / sigma**2
    dp[U + 3 * N + 2] = g @ B_neutral + 2 * mu_neutral / sigma**2

    return g @ a0, g @ a1, g @ a2, dp


@nb.njit(**kwd)
def _grad_BaBar(mu_SM, mu_NP_1, mu_NP_2, s_NP_1, s_NP_2, R_NP_1, R_NP_2, p):
    """Gradient of the BaBar -2 ln L w.r.t. (mu_SM, mu_NP_1, mu_NP_2) and the nuisance block p."""
    N, U = Nbins_BaBar, NU_BaBar
    u, th_bkg = p[0:U], p[U : U + N]
    rec, NiB, Nnorm = p[U + N], p[U + N + 1], p[U + N + 2]

    a0 = s_SM_BaBar * np.exp(R_SM_BaBar @ u)
    a1 = s_NP_1 * np.exp(R_NP_1 @ u)
    a2 = s_NP_2 * np.exp(R_NP_2 @ u)
    signal = mu_SM * a0 + mu_NP_1 * a1 + mu_NP_2 * a2
    peak = (1 + rec) * (1 + NiB) * b_BaBar_peak_K0
    bkg_shape = peak + b_BaBar_comb_K0
    e = (1 + rec) * signal + (1 + Nnorm) * (1 + th_bkg) * bkg_shape
    g = 2 * (1 - obs_BaBar_K0 / e)

    dp = np.zeros(NPAR_BaBar)
    dp[0:U] = (1 + rec) * (
        R_SM_BaBar.T @ (g * mu_SM * a0) + R_NP_1.T @ (g * mu_NP_1 * a1) + R_NP_2.T @ (g * mu_NP_2 * a2)
    ) + 2 * u
    tb = th_bkg * b_BaBar_K0
    dp[U : U + N] = g * (1 + Nnorm) * bkg_shape + 2 * b_BaBar_K0 * (b_BaBar_K0_invcov @ tb)
    dp[U + N] = g @ signal + g @ ((1 + Nnorm) * (1 + th_bkg) * (1 + NiB) * b_BaBar_peak_K0) + 2 * rec / sigrecB0Ks0**2
    dp[U + N + 1] = g @ ((1 + Nnorm) * (1 + th_bkg) * (1 + rec) * b_BaBar_peak_K0) + 2 * NiB / NiBs**2
    dp[U + N + 2] = g @ ((1 + th_bkg) * bkg_shape) + 2 * Nnorm / NinormB0Ks0**2

    return (1 + rec) * (g @ a0), (1 + rec) * (g @ a1), (1 + rec) * (g @ a2), dp


@nb.njit(**kwd)
def grad_BelleII(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "Gradient of logLL_BelleII w.r.t. x."
    d0, d1, d2, dp = _grad_BelleII(x[0], x[1] ** 2, x[2] ** 2, s_NP_1, s_NP_2, R_NP_1, R_NP_2, x[3:])
    out = np.zeros(len(x))
    out[0], out[1], out[2] = d0, 2 * x[1] * d1, 2 * x[2] * d2
    out[3 : 3 + NPAR_BelleII] = dp
    return out


@nb.njit(**kwd)
def grad_BaBar(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "Gradient of logLL_BaBar w.r.t. x."
    d0, d1, d2, dp = _grad_BaBar(x[0], x[1] ** 2, x[2] ** 2, s_NP_1, s_NP_2, R_NP_1, R_NP_2, x[3:])
    out = np.zeros(len(x))
    out[0], out[1], out[2] = d0, 2 * x[1] * d1, 2 * x[2] * d2
    out[3 : 3 + NPAR_BaBar] = dp
    return out


@nb.njit(**kwd)
def grad_Combined(
    x,
    s_NP_1_BelleII,
    s_NP_2_BelleII,
    R_NP_1_BelleII,
    R_NP_2_BelleII,
    s_NP_1_BaBar,
    s_NP_2_BaBar,
    R_NP_1_BaBar,
    R_NP_2_BaBar,
):
    "Gradient of logLL_Combined w.r.t. x."
    pB = 3
    pb = 3 + NPAR_BelleII
    m0, m1, m2 = x[0], x[1] ** 2, x[2] ** 2
    B0, B1, B2, dpB = _grad_BelleII(
        m0, m1, m2, s_NP_1_BelleII, s_NP_2_BelleII, R_NP_1_BelleII, R_NP_2_BelleII, x[pB:pb]
    )
    b0, b1, b2, dpb = _grad_BaBar(m0, m1, m2, s_NP_1_BaBar, s_NP_2_BaBar, R_NP_1_BaBar, R_NP_2_BaBar, x[pb:])
    out = np.zeros(len(x))
    out[0], out[1], out[2] = B0 + b0, 2 * x[1] * (B1 + b1), 2 * x[2] * (B2 + b2)
    out[pB:pb] = dpB
    out[pb : pb + NPAR_BaBar] = dpb
    return out


@nb.njit(**kwd)
def grad_LEP(x, Br_LEP_1, Br_LEP_2):
    "Gradient of the LEP term logLL_LEP([x1, x2, Br_LEP_1, Br_LEP_2]) w.r.t. the full x."
    Br = x[1] ** 2 * Br_LEP_1 + x[2] ** 2 * Br_LEP_2
    c = 2 * (LEP_N_norm * Br - LEP_mu) / LEP_sigma**2 * LEP_N_norm
    out = np.zeros(len(x))
    out[1] = c * 2 * x[1] * Br_LEP_1
    out[2] = c * 2 * x[2] * Br_LEP_2
    return out


# Fitted components (for plotting)
def components_BelleII(x, s_NP_1=None, s_NP_2=None, R_NP_1=None, R_NP_2=None):
    "Fitted Belle II signal and background components in each bin, from a parameter vector x."
    N, U = Nbins, NU_BelleII
    s_NP_1, s_NP_2, R_NP_1, R_NP_2 = _fill_zero("BelleII", s_NP_1, s_NP_2, R_NP_1, R_NP_2)
    x = np.asarray(x, dtype=float)
    p = x[3:]
    u = p[0:U]
    z_cont, z_charged, z_neutral = p[U : U + N], p[U + N : U + 2 * N], p[U + 2 * N : U + 3 * N]
    mu_cont, mu_charged, mu_neutral = p[U + 3 * N : U + 3 * N + 3]
    return {
        "SM": x[0] * s_SM_BelleII * np.exp(R_SM_BelleII @ u),
        "NP_1": x[1] ** 2 * s_NP_1 * np.exp(R_NP_1 @ u),
        "NP_2": x[2] ** 2 * s_NP_2 * np.exp(R_NP_2 @ u),
        "cont": (1 + mu_cont) * (b_cont + w_cont * z_cont),
        "charged": (1 + mu_charged) * (b_charged + w_charged * z_charged),
        "neutral": (1 + mu_neutral) * (b_neutral + w_neutral * z_neutral),
    }


def components_BaBar(x, s_NP_1=None, s_NP_2=None, R_NP_1=None, R_NP_2=None):
    "Fitted BaBar signal and background components in each bin, from a parameter vector x."
    N, U = Nbins_BaBar, NU_BaBar
    s_NP_1, s_NP_2, R_NP_1, R_NP_2 = _fill_zero("BaBar", s_NP_1, s_NP_2, R_NP_1, R_NP_2)
    x = np.asarray(x, dtype=float)
    p = x[3:]
    u, th_bkg = p[0:U], p[U : U + N]
    rec, NiB, Ninorm = p[U + N : U + N + 3]
    return {
        "SM": (1 + rec) * x[0] * s_SM_BaBar * np.exp(R_SM_BaBar @ u),
        "NP_1": (1 + rec) * x[1] ** 2 * s_NP_1 * np.exp(R_NP_1 @ u),
        "NP_2": (1 + rec) * x[2] ** 2 * s_NP_2 * np.exp(R_NP_2 @ u),
        "peak": (1 + Ninorm) * (1 + th_bkg) * (1 + rec) * (1 + NiB) * b_BaBar_peak_K0,
        "comb": (1 + Ninorm) * (1 + th_bkg) * b_BaBar_comb_K0,
    }


# Minimisation
_NBINS = {"BelleII": Nbins, "BaBar": Nbins_BaBar}
_NU = {"BelleII": NU_BelleII, "BaBar": NU_BaBar}
_NPAR = {"BelleII": 3 + NPAR_BelleII, "BaBar": 3 + NPAR_BaBar, "Combined": 3 + NPAR_BelleII + NPAR_BaBar}
_LOGLL = {"BelleII": logLL_BelleII, "BaBar": logLL_BaBar, "Combined": logLL_Combined}
_GRAD = {"BelleII": grad_BelleII, "BaBar": grad_BaBar, "Combined": grad_Combined}
_KIND_BY_NAME = {"logLL_BelleII": "BelleII", "logLL_BaBar": "BaBar", "logLL_Combined": "Combined"}


def _kind_of(logLL):
    """Experiment ("BelleII", "BaBar", "Combined") of a likelihood function.

    Looked up by function name, not object identity: joblib workers receive a pickled copy
    of the function, which is a different object from the module-level one.
    """
    if isinstance(logLL, str) and logLL in _LOGLL:
        return logLL
    name = getattr(logLL, "__name__", None)
    if name not in _KIND_BY_NAME:
        raise ValueError(f"Unknown likelihood {logLL!r}; expected logLL_BelleII, logLL_BaBar or logLL_Combined")
    return _KIND_BY_NAME[name]


_Z_LIM = (-5.0, 5.0)  # unit-Gaussian background modes z (b + w z stays positive)
_P_LIM = (-1.0, None)  # multiplicative nuisance params entering as (1 + p)


def _fill_zero(experiment, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "Replace missing NP templates by zeros of the right shape."
    N, U = _NBINS[experiment], _NU[experiment]
    z1, z2 = np.zeros(N), np.zeros((N, U))
    return (
        z1 if s_NP_1 is None else s_NP_1,
        z1 if s_NP_2 is None else s_NP_2,
        z2 if R_NP_1 is None else R_NP_1,
        z2 if R_NP_2 is None else R_NP_2,
    )


def np_templates(experiment, model_1=None, model_2=None):
    """NP templates (s_NP_1, s_NP_2, R_NP_1, R_NP_2) of two Models for one experiment ("BelleII" or "BaBar").

    A missing model (None) or one with no signal in this experiment gives zero templates.
    """

    def one(model):
        if model is None or not np.any(model.C):
            return np.zeros(_NBINS[experiment]), np.zeros((_NBINS[experiment], _NU[experiment]))
        s0, J = signal_template(experiment, model)
        return s0, _relative(s0, J)

    (s1, R1), (s2, R2) = one(model_1), one(model_2)
    return s1, s2, R1, R2


class FitResult(NamedTuple):
    """Result of min_logLL; also unpacks as (value, error, minuit, delta_chi2).

    value, error: the SM signal strength mu_SM (no NP model), or the coupling multiplier(s)
                  of the NP model(s) (a pair for two models)
    minuit:       the Minuit object of the fit (couplings in normalised units, see _normalise)
    delta_chi2:   -2 Delta ln L relative to the SM (mu_SM = 1, no new physics)
    """

    value: object
    error: object
    minuit: object
    delta_chi2: float


class _ScanGrids(NamedTuple):
    BelleII: np.ndarray
    BaBar: np.ndarray
    LEP: np.ndarray
    combined: np.ndarray


class ScanResult(_ScanGrids):
    """Delta chi^2 grids of profile_logLL_scan, indexed [x2, x1]; also unpacks as a 4-tuple.

    Besides the four grids (BelleII, BaBar, LEP, combined) it records the scan itself:
    x1, x2: the coupling multipliers of the grid; models: (model_1, model_2).
    """

    def __new__(cls, BelleII, BaBar, LEP, combined, x1=None, x2=None, models=(None, None)):
        self = super().__new__(cls, BelleII, BaBar, LEP, combined)
        self.x1, self.x2, self.models = x1, x2, models
        return self

    def __getnewargs__(self):  # pickling keeps the grids and models
        return (*self, self.x1, self.x2, self.models)


def _limits(kind):
    """Parameter limits for a likelihood of the given kind ("BelleII", "BaBar", "Combined").

    mu_SM >= 0; x1, x2 enter squared and are free; the input nuisances u are free (the
    log-normal response keeps the prediction positive); the background modes z are limited to
    +-5; multiplicative (1 + p) nuisance params are limited to p >= -1.
    """
    N, Nb = Nbins, Nbins_BaBar
    free = (None, None)
    belle = [free] * NU_BelleII + [_Z_LIM] * (3 * N) + [_P_LIM] * 3
    babar = [free] * NU_BaBar + [_P_LIM] * (Nb + 3)
    head = [(0.0, None), (None, None), (None, None)]
    return head + {"BelleII": belle, "BaBar": babar, "Combined": belle + babar}[kind]


def _fit(kind, templates, x_init, fixed, extra=None, errors=False):
    """Run one Migrad fit and return the Minuit object.

    templates are the NP template arguments of the likelihood (see np_templates; for
    "Combined" the Belle II ones followed by the BaBar ones). extra = (f, grad) is an optional
    term added to the likelihood (e.g. LEP), with its gradient. The analytic gradient is
    passed to Migrad, which runs with strategy 0: the minimum is found with far fewer calls
    because no numerical error matrix is built along the way. Set errors = True to run Hesse
    afterwards when parameter uncertainties are needed. A free NP coupling whose templates are all zero
    does not affect the likelihood and is fixed, to keep the Hesse matrix non-singular.
    """
    logLL, grad = _LOGLL[kind], _GRAD[kind]
    templates = tuple(np.ascontiguousarray(t, dtype=float) for t in templates)
    if extra is None:
        func = lambda x: logLL(x, *templates)
        gfunc = lambda x: grad(x, *templates)
    else:
        f_extra, g_extra = extra
        func = lambda x: logLL(x, *templates) + f_extra(x)
        gfunc = lambda x: grad(x, *templates) + g_extra(x)

    s_NP = templates[0::4] + templates[1::4]  # (s_NP_1 of each experiment, s_NP_2 of each)
    n_exp = len(templates) // 4
    flat = [
        f"x{k}"
        for k in (1, 2)
        if f"x{k}" not in fixed and not any(np.any(s) for s in s_NP[(k - 1) * n_exp : k * n_exp])
    ]

    m = Minuit(func, np.asarray(x_init, dtype=float), grad=gfunc)
    m.errordef = 1.0
    m.limits = _limits(kind)
    m.fixed[list(fixed) + flat] = True
    m.strategy = 0
    m.migrad(ncall=100000)
    if errors:
        m.hesse()
    return m


def _x_init(kind, head=(1.0, 0.0, 0.0)):
    "Initial parameter vector: the given (mu_SM, x1, x2) and all nuisance params at 0."
    x = np.zeros(_NPAR[kind])
    x[:3] = head
    return x


def _normalise(tpl, target=100.0):
    """Rescale NP templates to `target` events at unit coupling, for a well-scaled fit.

    The coupling then enters as x_fit = x_true * sqrt(scale); Delta chi^2 is unaffected.
    Without this, templates that vanish in some limit (e.g. ~ m_X^2 as m_X -> 0) would need
    astronomically large couplings that Migrad cannot reach. Returns (templates, scales).
    """
    s1, s2, R1, R2 = tpl
    scales = []
    out_s, out_J = [], []
    for s, R in ((s1, R1), (s2, R2)):
        total = float(np.sum(s))
        scale = total / target if total > 0 else 1.0
        scales.append(scale)
        out_s.append(s / scale)
        out_J.append(R)  # R = J/s is scale-free
    return (out_s[0], out_s[1], out_J[0], out_J[1]), scales


def min_logLL(logLL, model_1=None, model_2=None, *legacy_args, **legacy_kwargs):
    """Minimise the log likelihood w.r.t. the SM signal strength or NP couplings and the nuisance parameters.

    logLL:     logLL_BelleII or logLL_BaBar, which selects the experiment
    model_1/2: NP Models whose couplings are scaled by the fitted multipliers x1, x2
               (signal strengths x1^2, x2^2). With no model, the SM signal strength mu_SM is
               fitted; with one or two models, mu_SM = 1 and x1 (and x2) are fitted.

    Returns a FitResult (value, error, minuit, delta_chi2): mu_SM, or the coupling
    multiplier(s) of the model(s), with delta_chi2 relative to the SM (mu_SM = 1, no NP). The
    NP templates are normalised internally (see _normalise), so the couplings inside the
    Minuit object are in normalised units; use value and error instead.

    (The old call min_logLL(logLL, Nrec, Nbins, mX, cNP_1, cNP_2, Lambda, is_ident, NP = ...)
    with coupling arrays still works.)
    """
    if legacy_args or legacy_kwargs or callable(model_1):
        vals = from_legacy(
            "min_logLL",
            (model_1, model_2) + legacy_args,
            legacy_kwargs,
            ("Nrec", "Nbins", "mX", "cNP_1", "cNP_2", "Lambda", "is_ident"),
            {"mX": 1, "cNP_1": None, "cNP_2": None, "Lambda": 1e4, "is_ident": False, "NP": 0},
        )
        NP = vals["NP"]
        if NP not in (0, 1, 2):
            raise ValueError("NP must be 0, 1 or 2")
        model_1 = vals["cNP_1"] if NP >= 1 else None
        model_2 = vals["cNP_2"] if NP == 2 else None
        if NP >= 1 and model_1 is None:  # all-zero couplings: nothing to fit, but keep the old behaviour
            model_1 = Model("phi", vals["mX"])
        if NP == 2 and model_2 is None:
            model_2 = Model("phi", vals["mX"])
    if model_1 is None and model_2 is not None:
        raise ValueError("Pass a single NP model as model_1")
    NP = (model_1 is not None) + (model_2 is not None)
    kind = _kind_of(logLL)

    tpl = np_templates(kind, model_1, model_2)
    tpl, scales = _normalise(tpl)
    unscale = 1 / np.sqrt(scales[0]), 1 / np.sqrt(scales[1])

    # Minimise w.r.t. nuisance params., keeping signal strength fixed to SM value (mu = 1)
    profile_min_SM = _fit(kind, tpl, _x_init(kind), ["x0", "x1", "x2"]).fval

    if NP == 0:
        # Minimise w.r.t. signal strength and nuisance params.
        m = _fit(kind, tpl, _x_init(kind), ["x1", "x2"], errors=True)
        return FitResult(m.values[0], m.errors[0], m, m.fval - profile_min_SM)

    if NP == 1:
        # Minimise Log Likelihood of SM + NP (1 coupling)
        m = _fit(kind, tpl, _x_init(kind), ["x0", "x2"], errors=True)
        return FitResult(m.values[1] * unscale[0], m.errors[1] * unscale[0], m, m.fval - profile_min_SM)

    # Minimise Log Likelihood of SM + NP (2 couplings)
    m = _fit(kind, tpl, _x_init(kind), ["x0"], errors=True)
    return FitResult(
        (m.values[1] * unscale[0], m.values[2] * unscale[1]),
        (m.errors[1] * unscale[0], m.errors[2] * unscale[1]),
        m,
        m.fval - profile_min_SM,
    )


def profile_logLL_SM(logLL, mu):
    "Profile log likelihood w.r.t. the signal strength and nuisance parameters."
    kind = _kind_of(logLL)
    tpl = np_templates(kind)

    # Minimise w.r.t. nuisance params. for fixed signal strength
    profile_min = _fit(kind, tpl, _x_init(kind, (mu, 0.0, 0.0)), ["x0", "x1", "x2"]).fval

    # Minimise w.r.t. signal strength and nuisance params. (i.e. global minimum)
    global_min = _fit(kind, tpl, _x_init(kind), ["x1", "x2"]).fval

    return profile_min - global_min


def _lep_term(Br_LEP):
    "LEP log likelihood and its gradient as functions of the parameter vector x (uses x[1], x[2])."
    B1, B2 = float(Br_LEP[0]), float(Br_LEP[1])
    return (lambda x: logLL_LEP(np.array([x[1], x[2], B1, B2])), lambda x: grad_LEP(np.asarray(x, dtype=float), B1, B2))


def _profile_point(c1, c2, sig, templates, global_mins, Br_LEP):
    """Profile log-likelihood at ONE (cNP_1, cNP_2) grid point. This runs inside a worker."""
    head = (1.0, c1, c2)
    fix = ["x0", "x1", "x2"]
    res = np.zeros(4)  # [BelleII, BaBar, LEP, Combined]

    # Belle II
    if sig[0] == 1:
        res[0] = _fit("BelleII", templates["BelleII"], _x_init("BelleII", head), fix).fval - global_mins[0]

    # BaBar
    if sig[1] == 1:
        res[1] = _fit("BaBar", templates["BaBar"], _x_init("BaBar", head), fix).fval - global_mins[1]

    # LEP (needed for both the LEP-only and the combined result)
    if sig[2] == 1 or sig[3] == 1:
        lep = logLL_LEP(np.concatenate(([c1, c2], [Br_LEP[0], Br_LEP[1]])))
        if sig[2] == 1:
            res[2] = lep - global_mins[2]

    # Combined
    if sig[3] == 1:
        res[3] = _fit("Combined", templates["Combined"], _x_init("Combined", head), fix).fval + lep - global_mins[3]

    return res


def profile_logLL_scan(
    model_1, model_2, lims_1, lims_2, *legacy_args, pts=(100, 100), sig=(1, 1, 1, 1), n_jobs=-1, **legacy_kwargs
):
    """Profile log likelihood on a grid of the coupling multipliers of two NP Models (parallel over the grid).

    The couplings of model_1 and model_2 are scaled by x1 in lims_1 and x2 in lims_2 (pts
    points each); the two models may be different channels or masses. sig selects which of
    [Belle II, BaBar, LEP, combined] to compute. Returns a ScanResult of the four Delta chi^2
    grids (BelleII, BaBar, LEP, combined), indexed [x2, x1]. (The old call profile_logLL_scan(mX, cNP_1, cNP_2, lims_1, lims_2, pts, Lambda,
    is_ident, sig, n_jobs) with coupling arrays still works.)
    """
    if not isinstance(model_1, Model) or legacy_args or legacy_kwargs:
        vals = from_legacy(
            "profile_logLL_scan",
            (model_1, model_2, lims_1, lims_2) + legacy_args,
            legacy_kwargs,
            ("mX", "cNP_1", "cNP_2", "lims_1", "lims_2", "pts", "Lambda", "is_ident", "sig", "n_jobs"),
            {"pts": pts, "Lambda": 1.0, "is_ident": False, "sig": sig, "n_jobs": n_jobs},
        )
        model_1, model_2 = vals["cNP_1"], vals["cNP_2"]
        lims_1, lims_2, pts, sig, n_jobs = vals["lims_1"], vals["lims_2"], vals["pts"], vals["sig"], vals["n_jobs"]
    cNP_1_lims, cNP_2_lims = lims_1, lims_2

    # ---- Signal templates (computed once, shared by every grid point) ----
    tpl_BelleII = np_templates("BelleII", model_1, model_2)
    tpl_BaBar = np_templates("BaBar", model_1, model_2)
    templates = {"BelleII": tpl_BelleII, "BaBar": tpl_BaBar, "Combined": tpl_BelleII + tpl_BaBar}

    Br_LEP = (B_Bs(model_1), B_Bs(model_2))

    # ---- Global minima (only 3 fits, done serially) ----
    global_mins = [0.0, 0.0, 0.0, 0.0]  # [BelleII, BaBar, LEP, Combined]
    if sig[0] == 1:
        global_mins[0] = _fit("BelleII", tpl_BelleII, _x_init("BelleII"), ["x0"]).fval
    if sig[1] == 1:
        global_mins[1] = _fit("BaBar", tpl_BaBar, _x_init("BaBar"), ["x0"]).fval
    if sig[2] == 1:
        global_mins[2] = 0
    if sig[3] == 1:
        # The profiled combined values include the LEP term, so the global minimum must too
        global_mins[3] = _fit(
            "Combined", templates["Combined"], _x_init("Combined"), ["x0"], extra=_lep_term(Br_LEP)
        ).fval

    # ---- Profiled scan, parallel over the (cNP_1, cNP_2) grid ----
    cNP_1_vals = np.linspace(cNP_1_lims[0], cNP_1_lims[1], pts[0])
    cNP_2_vals = np.linspace(cNP_2_lims[0], cNP_2_lims[1], pts[1])
    n1, n2 = len(cNP_1_vals), len(cNP_2_vals)

    jobs = (
        delayed(_profile_point)(c1, c2, sig, templates, global_mins, Br_LEP) for c1 in cNP_1_vals for c2 in cNP_2_vals
    )

    gen = Parallel(n_jobs=n_jobs, return_as="generator")(jobs)
    results = np.array(list(tqdm(gen, total=n1 * n2, desc=f"{model_1.channel.name}, mX = {model_1.mX:.3g}")))

    # results is ordered (i over cNP_1, j over cNP_2) -> reshape, then transpose to [j, i] like the original
    results = results.reshape(n1, n2, 4)
    LL_vals_BelleII = results[:, :, 0].T
    LL_vals_BaBar = results[:, :, 1].T
    LL_vals_LEP = results[:, :, 2].T
    LL_vals_Combined = results[:, :, 3].T

    return ScanResult(
        LL_vals_BelleII, LL_vals_BaBar, LL_vals_LEP, LL_vals_Combined, cNP_1_vals, cNP_2_vals, (model_1, model_2)
    )
