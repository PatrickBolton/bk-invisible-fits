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
    b_BaBar_K0_sigma,
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
from lights.models import B_Bs, check_model
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


def _relative(s, J):
    "Relative response R = J / s of a template (zero where the template is zero)."
    s = np.asarray(s, dtype=float)
    safe = np.where(s > 0, s, 1.0)
    return np.ascontiguousarray(np.where(s[:, None] > 0, J / safe[:, None], 0.0))


# Relative responses of the SM templates to the input nuisance params.
R_SM_BelleII = _relative(s_SM_BelleII, J_SM_BelleII)
R_SM_BaBar = _relative(s_SM_BaBar, J_SM_BaBar)


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
#
# Each experiment has one _model_* function that evaluates its signal and background
# components; the expected events, the gradient and components_* are all built from it.


@nb.njit(**kwd)
def _signals(s_SM, R_SM, s_NP_1, s_NP_2, R_NP_1, R_NP_2, u):
    "Signal templates (SM, NP 1, NP 2) at the input nuisance params u."
    return s_SM * np.exp(R_SM @ u), s_NP_1 * np.exp(R_NP_1 @ u), s_NP_2 * np.exp(R_NP_2 @ u)


@nb.njit(**kwd)
def _mix(c, rows):
    "c[0] rows[0] + c[1] rows[1] + c[2] rows[2], for three coefficients and three arrays."
    return c[0] * rows[0] + c[1] * rows[1] + c[2] * rows[2]


@nb.njit(**kwd)
def _exp_BelleII(mu, sig, bkg, mu_bkg):
    "Expected Belle II events from the components of _model_BelleII."
    return _mix(mu, sig) + _mix((1 + mu_bkg[0], 1 + mu_bkg[1], 1 + mu_bkg[2]), bkg)


@nb.njit(**kwd)
def _poisson(exp, obs):
    "-2 ln of the Poisson likelihood of obs given exp, summed over bins (constant terms dropped)."
    return 2 * np.sum(exp - obs * np.log(exp))


@nb.njit(**kwd)
def _chain(x, d_mu, d_p):
    "Gradient w.r.t. x from the one w.r.t. the signal strengths (mu_SM, x1^2, x2^2) and the nuisance params."
    out = np.empty(3 + len(d_p))
    out[0], out[1], out[2] = d_mu[0], 2 * x[1] * d_mu[1], 2 * x[2] * d_mu[2]
    out[3:] = d_p
    return out


# Belle II ITA
# Nuisance block p (length NPAR_BelleII), with U = NU_BelleII:
#   p[0:U]            u          input nuisance params. (B -> K form factors, efficiencies)
#   p[U:U+N]          z_cont     background MC modes, shift w_cont * z_cont (uncorrelated)
#   p[U+N:U+2N]       z_charged
#   p[U+2N:U+3N]      z_neutral
#   p[U+3N:U+3N+3]    mu_cont, mu_charged, mu_neutral   background normalisation params.
NPAR_BelleII = NU_BelleII + 3 * Nbins + 3
BKG_NAMES_BelleII = ("cont", "charged", "neutral")

# Relative uncertainty of each Belle II background normalisation (mu_cont, mu_charged, mu_neutral)
BKG_NORM_SIGMA = 0.02

_w_BelleII = (w_cont, w_charged, w_neutral)  # MC uncertainties, in the order of BKG_NAMES_BelleII


@nb.njit(**kwd)
def _model_BelleII(p, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    """Belle II components at the nuisance block p.

    Returns the signals (SM, NP 1, NP 2) at unit strength; the backgrounds (cont, charged,
    neutral) before normalisation, b + w z; and their normalisation params mu_bkg.
    The expected events are sum_i mu_i sig_i + sum_k (1 + mu_bkg_k) bkg_k.
    """
    N, U = Nbins, NU_BelleII
    sig = _signals(s_SM_BelleII, R_SM_BelleII, s_NP_1, s_NP_2, R_NP_1, R_NP_2, p[0:U])
    bkg = (
        b_cont + w_cont * p[U : U + N],
        b_charged + w_charged * p[U + N : U + 2 * N],
        b_neutral + w_neutral * p[U + 2 * N : U + 3 * N],
    )
    return sig, bkg, (p[U + 3 * N], p[U + 3 * N + 1], p[U + 3 * N + 2])


@nb.njit(**kwd)
def _nll_BelleII(mu, p, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "-2 ln L of Belle II for signal strengths mu = (mu_SM, mu_NP_1, mu_NP_2) and nuisance block p."
    U = NU_BelleII
    sig, bkg, mu_bkg = _model_BelleII(p, s_NP_1, s_NP_2, R_NP_1, R_NP_2)
    exp = _exp_BelleII(mu, sig, bkg, mu_bkg)
    u, z = p[0:U], p[U : NPAR_BelleII - 3]
    return _poisson(exp, obs) + u @ u + z @ z + (mu_bkg[0] ** 2 + mu_bkg[1] ** 2 + mu_bkg[2] ** 2) / BKG_NORM_SIGMA**2


@nb.njit(**kwd)
def _nll_grad_BelleII(mu, p, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    """Gradient of _nll_BelleII w.r.t. mu and p.

    With f = 2 sum(e - o ln e) + penalties, df/de = 2 (1 - o/e); the chain rule through e
    gives the rest. Checked against finite differences in scripts/check_gradients.py.
    """
    N, U = Nbins, NU_BelleII
    sig, bkg, mu_bkg = _model_BelleII(p, s_NP_1, s_NP_2, R_NP_1, R_NP_2)
    g = 2 * (1 - obs / _exp_BelleII(mu, sig, bkg, mu_bkg))

    d_p = np.empty(NPAR_BelleII)
    d_p[0:U] = (
        R_SM_BelleII.T @ (g * mu[0] * sig[0]) + R_NP_1.T @ (g * mu[1] * sig[1]) + R_NP_2.T @ (g * mu[2] * sig[2])
    ) + 2 * p[0:U]
    for k in range(3):
        d_p[U + k * N : U + (k + 1) * N] = g * (1 + mu_bkg[k]) * _w_BelleII[k] + 2 * p[U + k * N : U + (k + 1) * N]
        d_p[U + 3 * N + k] = g @ bkg[k] + 2 * mu_bkg[k] / BKG_NORM_SIGMA**2
    return np.array((sig[0] @ g, sig[1] @ g, sig[2] @ g)), d_p


# BaBar
# Nuisance block p (length NPAR_BaBar), with U = NU_BaBar:
#   p[0:U]          u           input nuisance params. (B -> K* form factors, efficiencies)
#   p[U:U+Nb]       th_bkg      statistical background (MC) nuisance params.
#   p[U+Nb]         rec         reconstruction nuisance param. (sigma sigrecB0Ks0)
#   p[U+Nb+1]       NiB         peaking background nuisance param. (sigma NiBs)
#   p[U+Nb+2]       Nnorm       background normalisation nuisance param. (sigma NinormB0Ks0)
NPAR_BaBar = NU_BaBar + Nbins_BaBar + 3

# Gaussian weight of the relative background shifts th_bkg: (b / sigma_b)^2
_mc_weight_BaBar = (b_BaBar_K0 / b_BaBar_K0_sigma) ** 2


@nb.njit(**kwd)
def _model_BaBar(p, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    """BaBar components at the nuisance block p.

    Returns the signals (SM, NP 1, NP 2) at unit strength; the background
    shape (1 + rec)(1 + NiB) b_peak + b_comb before the shifts (1 + Nnorm)(1 + th_bkg); and
    the nuisance params th_bkg, rec, NiB, Nnorm. The expected events are
    (1 + rec) sum_i mu_i sig_i + (1 + Nnorm)(1 + th_bkg) bkg.
    """
    N, U = Nbins_BaBar, NU_BaBar
    sig = _signals(s_SM_BaBar, R_SM_BaBar, s_NP_1, s_NP_2, R_NP_1, R_NP_2, p[0:U])
    th, rec, NiB, Nnorm = p[U : U + N], p[U + N], p[U + N + 1], p[U + N + 2]
    bkg = (1 + rec) * (1 + NiB) * b_BaBar_peak_K0 + b_BaBar_comb_K0
    return sig, bkg, th, rec, NiB, Nnorm


@nb.njit(**kwd)
def _nll_BaBar(mu, p, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "-2 ln L of BaBar for signal strengths mu = (mu_SM, mu_NP_1, mu_NP_2) and nuisance block p."
    sig, bkg, th, rec, NiB, Nnorm = _model_BaBar(p, s_NP_1, s_NP_2, R_NP_1, R_NP_2)
    exp = (1 + rec) * _mix(mu, sig) + (1 + Nnorm) * (1 + th) * bkg
    u = p[0:NU_BaBar]
    penalty = u @ u + th @ (_mc_weight_BaBar * th)
    penalty += (rec / sigrecB0Ks0) ** 2 + (NiB / NiBs) ** 2 + (Nnorm / NinormB0Ks0) ** 2
    return _poisson(exp, obs_BaBar_K0) + penalty


@nb.njit(**kwd)
def _nll_grad_BaBar(mu, p, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "Gradient of _nll_BaBar w.r.t. mu and p (see _nll_grad_BelleII)."
    N, U = Nbins_BaBar, NU_BaBar
    sig, bkg, th, rec, NiB, Nnorm = _model_BaBar(p, s_NP_1, s_NP_2, R_NP_1, R_NP_2)
    signal = _mix(mu, sig)
    shift = (1 + Nnorm) * (1 + th)
    g = 2 * (1 - obs_BaBar_K0 / ((1 + rec) * signal + shift * bkg))

    d_p = np.empty(NPAR_BaBar)
    d_p[0:U] = (1 + rec) * (
        R_SM_BaBar.T @ (g * mu[0] * sig[0]) + R_NP_1.T @ (g * mu[1] * sig[1]) + R_NP_2.T @ (g * mu[2] * sig[2])
    ) + 2 * p[0:U]
    d_p[U : U + N] = g * (1 + Nnorm) * bkg + 2 * _mc_weight_BaBar * th
    d_p[U + N] = g @ signal + g @ (shift * (1 + NiB) * b_BaBar_peak_K0) + 2 * rec / sigrecB0Ks0**2
    d_p[U + N + 1] = g @ (shift * (1 + rec) * b_BaBar_peak_K0) + 2 * NiB / NiBs**2
    d_p[U + N + 2] = g @ ((1 + th) * bkg) + 2 * Nnorm / NinormB0Ks0**2
    return (1 + rec) * np.array((sig[0] @ g, sig[1] @ g, sig[2] @ g)), d_p


# Public likelihoods and gradients of the parameter vector x
@nb.njit(**kwd)
def _strengths(x):
    "Signal strengths (mu_SM, mu_NP_1, mu_NP_2) = (x0, x1^2, x2^2)."
    return x[0], x[1] ** 2, x[2] ** 2


@nb.njit(**kwd)
def logLL_BelleII(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "-2 times the log likelihood (minus constant factors) for the Belle II inclusive tag."
    return _nll_BelleII(_strengths(x), x[3:], s_NP_1, s_NP_2, R_NP_1, R_NP_2)


@nb.njit(**kwd)
def logLL_BaBar(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "-2 times the log likelihood (minus constant factors) for BaBar."
    return _nll_BaBar(_strengths(x), x[3:], s_NP_1, s_NP_2, R_NP_1, R_NP_2)


# Combined: x[0:3] as above, then the Belle II nuisance block, then the BaBar nuisance block.
# The Belle II and BaBar inputs (B -> K vs B -> K* form factors, separate efficiencies) are
# independent, so the two blocks share no parameters apart from mu_SM and the NP couplings.
# Templates: the four Belle II ones, then the four BaBar ones.
_PB = 3 + NPAR_BelleII  # start of the BaBar block in the combined x


@nb.njit(**kwd)
def logLL_Combined(x, s1_B, s2_B, R1_B, R2_B, s1_b, s2_b, R1_b, R2_b):
    "-2 times the log likelihood (minus constant factors) for combined experiments."
    mu = _strengths(x)
    return _nll_BelleII(mu, x[3:_PB], s1_B, s2_B, R1_B, R2_B) + _nll_BaBar(mu, x[_PB:], s1_b, s2_b, R1_b, R2_b)


@nb.njit(**kwd)
def grad_BelleII(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "Gradient of logLL_BelleII w.r.t. x."
    return _chain(x, *_nll_grad_BelleII(_strengths(x), x[3:], s_NP_1, s_NP_2, R_NP_1, R_NP_2))


@nb.njit(**kwd)
def grad_BaBar(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "Gradient of logLL_BaBar w.r.t. x."
    return _chain(x, *_nll_grad_BaBar(_strengths(x), x[3:], s_NP_1, s_NP_2, R_NP_1, R_NP_2))


@nb.njit(**kwd)
def grad_Combined(x, s1_B, s2_B, R1_B, R2_B, s1_b, s2_b, R1_b, R2_b):
    "Gradient of logLL_Combined w.r.t. x."
    mu = _strengths(x)
    d_mu_B, d_p_B = _nll_grad_BelleII(mu, x[3:_PB], s1_B, s2_B, R1_B, R2_B)
    d_mu_b, d_p_b = _nll_grad_BaBar(mu, x[_PB:], s1_b, s2_b, R1_b, R2_b)
    return _chain(x, d_mu_B + d_mu_b, np.concatenate((d_p_B, d_p_b)))


@nb.njit(**kwd)
def Nevents_BelleII(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "Expected number of Belle II ITA events: SM plus NP B+ -> K+ invisible signal and backgrounds."
    sig, bkg, mu_bkg = _model_BelleII(x[3:], s_NP_1, s_NP_2, R_NP_1, R_NP_2)
    return _exp_BelleII(_strengths(x), sig, bkg, mu_bkg)


@nb.njit(**kwd)
def Nevents_BaBar(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "Expected number of BaBar events: SM plus NP B0 -> K*0 invisible signal and backgrounds."
    sig, bkg, th, rec, _, Nnorm = _model_BaBar(x[3:], s_NP_1, s_NP_2, R_NP_1, R_NP_2)
    return (1 + rec) * _mix(_strengths(x), sig) + (1 + Nnorm) * (1 + th) * bkg


# LEP (Bs -> inv): Gaussian in the number of events N_norm * B(Bs -> inv)
LEP_N_norm = 6607.14
LEP_mu = 0.525241
LEP_sigma = 2.47728


@nb.njit(**kwd)
def logLL_LEP(x, Br_LEP_1, Br_LEP_2):
    """-2 times the log likelihood (minus constant factors) for LEP.

    x is a parameter vector as above (only the couplings x[1], x[2] enter); Br_LEP_1/2 are
    the Bs -> invisible branching fractions of the two NP models at unit coupling.
    """
    Br = x[1] ** 2 * Br_LEP_1 + x[2] ** 2 * Br_LEP_2
    return (LEP_N_norm * Br - LEP_mu) ** 2 / LEP_sigma**2


@nb.njit(**kwd)
def grad_LEP(x, Br_LEP_1, Br_LEP_2):
    "Gradient of logLL_LEP w.r.t. x."
    Br = x[1] ** 2 * Br_LEP_1 + x[2] ** 2 * Br_LEP_2
    c = 2 * (LEP_N_norm * Br - LEP_mu) / LEP_sigma**2 * LEP_N_norm
    out = np.zeros(len(x))
    out[1] = c * 2 * x[1] * Br_LEP_1
    out[2] = c * 2 * x[2] * Br_LEP_2
    return out


# Fitted components (for plotting)
def components_BelleII(x, s_NP_1=None, s_NP_2=None, R_NP_1=None, R_NP_2=None):
    "Fitted Belle II signal and background components in each bin, from a parameter vector x."
    x = np.asarray(x, dtype=float)
    sig, bkg, mu_bkg = _model_BelleII(x[3:], *_fill_zero("BelleII", s_NP_1, s_NP_2, R_NP_1, R_NP_2))
    mu = _strengths(x)
    out = {"SM": mu[0] * sig[0], "NP_1": mu[1] * sig[1], "NP_2": mu[2] * sig[2]}
    out.update({name: (1 + mu_bkg[k]) * bkg[k] for k, name in enumerate(BKG_NAMES_BelleII)})
    return out


def components_BaBar(x, s_NP_1=None, s_NP_2=None, R_NP_1=None, R_NP_2=None):
    "Fitted BaBar signal and background components in each bin, from a parameter vector x."
    x = np.asarray(x, dtype=float)
    sig, _, th, rec, NiB, Nnorm = _model_BaBar(x[3:], *_fill_zero("BaBar", s_NP_1, s_NP_2, R_NP_1, R_NP_2))
    mu = _strengths(x)
    shift = (1 + Nnorm) * (1 + th)
    return {
        "SM": (1 + rec) * mu[0] * sig[0],
        "NP_1": (1 + rec) * mu[1] * sig[1],
        "NP_2": (1 + rec) * mu[2] * sig[2],
        "peak": shift * (1 + rec) * (1 + NiB) * b_BaBar_peak_K0,
        "comb": shift * b_BaBar_comb_K0,
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


def min_logLL(logLL, model_1=None, model_2=None):
    """Minimise the log likelihood w.r.t. the SM signal strength or NP couplings and the nuisance parameters.

    logLL:     logLL_BelleII or logLL_BaBar, which selects the experiment
    model_1/2: NP Models whose couplings are scaled by the fitted multipliers x1, x2
               (signal strengths x1^2, x2^2). With no model, the SM signal strength mu_SM is
               fitted; with one or two models, mu_SM = 1 and x1 (and x2) are fitted.

    Returns a FitResult (value, error, minuit, delta_chi2): mu_SM, or the coupling
    multiplier(s) of the model(s), with delta_chi2 relative to the SM (mu_SM = 1, no NP). The
    NP templates are normalised internally (see _normalise), so the couplings inside the
    Minuit object are in normalised units; use value and error instead.
    """
    check_model(model_1, "model_1")
    check_model(model_2, "model_2")
    if model_1 is None and model_2 is not None:
        raise ValueError("Pass a single NP model as model_1")
    NP = (model_1 is not None) + (model_2 is not None)
    kind = _kind_of(logLL)

    tpl = np_templates(kind, model_1, model_2)
    tpl, scales = _normalise(tpl)
    unscale = 1 / np.sqrt(scales[0]), 1 / np.sqrt(scales[1])

    # Minimise w.r.t. nuisance params., keeping signal strength fixed to SM value (mu = 1)
    profile_min_SM = _fit(kind, tpl, _x_init(kind), ["x0", "x1", "x2"]).fval

    # Fit mu_SM (no NP), or the NP coupling(s) with mu_SM = 1, together with the nuisance params.
    fixed = {0: ["x1", "x2"], 1: ["x0", "x2"], 2: ["x0"]}[NP]
    m = _fit(kind, tpl, _x_init(kind), fixed, errors=True)
    delta_chi2 = m.fval - profile_min_SM
    if NP == 0:
        return FitResult(m.values[0], m.errors[0], m, delta_chi2)
    value = tuple(m.values[k] * unscale[k - 1] for k in range(1, NP + 1))
    error = tuple(m.errors[k] * unscale[k - 1] for k in range(1, NP + 1))
    if NP == 1:
        value, error = value[0], error[0]
    return FitResult(value, error, m, delta_chi2)


def profile_logLL_SM(logLL, mu):
    """Profile -2 Delta ln L of the SM signal strength mu (a number or an array of them).

    At each mu the nuisance params. are fitted; the result is relative to the global minimum,
    where mu is fitted too (computed once).
    """
    kind = _kind_of(logLL)
    tpl = np_templates(kind)
    global_min = _fit(kind, tpl, _x_init(kind), ["x1", "x2"]).fval
    profile = [_fit(kind, tpl, _x_init(kind, (m, 0.0, 0.0)), ["x0", "x1", "x2"]).fval for m in np.ravel(mu)]
    return np.reshape(profile, np.shape(mu)) - global_min


def _lep_term(Br_LEP):
    "LEP log likelihood and its gradient as functions of the parameter vector x."
    B1, B2 = float(Br_LEP[0]), float(Br_LEP[1])
    return (lambda x: logLL_LEP(x, B1, B2), lambda x: grad_LEP(x, B1, B2))


def _profile_point(c1, c2, sig, templates, global_mins, Br_LEP):
    """Profile log-likelihood at ONE (x1, x2) grid point. This runs inside a worker."""
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
        lep = logLL_LEP(np.array(head), float(Br_LEP[0]), float(Br_LEP[1]))
        if sig[2] == 1:
            res[2] = lep - global_mins[2]

    # Combined
    if sig[3] == 1:
        res[3] = _fit("Combined", templates["Combined"], _x_init("Combined", head), fix).fval + lep - global_mins[3]

    return res


def profile_logLL_scan(model_1, model_2, lims_1, lims_2, *, pts=(100, 100), sig=(1, 1, 1, 1), n_jobs=-1):
    """Profile log likelihood on a grid of the coupling multipliers of two NP Models (parallel over the grid).

    The couplings of model_1 and model_2 are scaled by x1 in lims_1 and x2 in lims_2 (pts
    points each); the two models may be different channels or masses. sig selects which of
    [Belle II, BaBar, LEP, combined] to compute. Returns a ScanResult of the four Delta chi^2
    grids (BelleII, BaBar, LEP, combined), indexed [x2, x1].
    """
    check_model(model_1, "model_1", allow_none=False)
    check_model(model_2, "model_2")

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

    # ---- Profiled scan, parallel over the (x1, x2) grid ----
    x1_vals = np.linspace(lims_1[0], lims_1[1], pts[0])
    x2_vals = np.linspace(lims_2[0], lims_2[1], pts[1])
    n1, n2 = len(x1_vals), len(x2_vals)

    jobs = (delayed(_profile_point)(c1, c2, sig, templates, global_mins, Br_LEP) for c1 in x1_vals for c2 in x2_vals)

    gen = Parallel(n_jobs=n_jobs, return_as="generator")(jobs)
    results = np.array(list(tqdm(gen, total=n1 * n2, desc=f"{model_1.channel.name}, mX = {model_1.mX:.3g}")))

    # results is ordered (i over x1, j over x2) -> reshape, then transpose to [j, i]
    results = results.reshape(n1, n2, 4)
    LL_vals_BelleII = results[:, :, 0].T
    LL_vals_BaBar = results[:, :, 1].T
    LL_vals_LEP = results[:, :, 2].T
    LL_vals_Combined = results[:, :, 3].T

    return ScanResult(
        LL_vals_BelleII, LL_vals_BaBar, LL_vals_LEP, LL_vals_Combined, x1_vals, x2_vals, (model_1, model_2)
    )
