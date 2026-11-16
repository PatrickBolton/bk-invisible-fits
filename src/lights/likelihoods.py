"""Likelihoods of Belle II, BaBar and LEP, their gradients and the NP signal templates.

The likelihoods are -2 ln L as functions of the parameter vector x (layout below), compiled
with numba; lights.fits minimises them.
"""

from typing import NamedTuple

import numba as nb
import numpy as np

from lights import histfactory
from lights.load_data import BaBar, BelleII, BelleII_public
from lights.signal_rates import (
    J_SM_BaBar,
    J_SM_BelleII,
    J_SM_BelleII_public,
    NU_BaBar,
    NU_BelleII,
    NU_BelleII_public,
    s_SM_BaBar,
    s_SM_BelleII,
    s_SM_BelleII_public,
    signal_template,
)

# The data, as module-level globals: numba compiles them into the likelihoods as constants
Nbins, obs = BelleII.Nbins, BelleII.obs
b_cont, b_charged, b_neutral = BelleII.b_cont, BelleII.b_charged, BelleII.b_neutral
w_cont, w_charged, w_neutral = BelleII.w_cont, BelleII.w_charged, BelleII.w_neutral
Nbins_BaBar, obs_BaBar = BaBar.Nbins, BaBar.obs
b_peak_BaBar, b_comb_BaBar = BaBar.b_peak, BaBar.b_comb
sigma_rec, sigma_NiB, sigma_Nnorm = BaBar.sigma_rec, BaBar.sigma_NiB, BaBar.sigma_Nnorm

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
#   p[U+Nb]         rec         reconstruction nuisance param. (width sigma_rec)
#   p[U+Nb+1]       NiB         peaking background nuisance param. (width sigma_NiB)
#   p[U+Nb+2]       Nnorm       background normalisation nuisance param. (width sigma_Nnorm)
NPAR_BaBar = NU_BaBar + Nbins_BaBar + 3

# Gaussian weight of the relative background shifts th_bkg: (b / sigma_b)^2
_mc_weight_BaBar = (BaBar.b_tot / BaBar.b_sigma) ** 2


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
    bkg = (1 + rec) * (1 + NiB) * b_peak_BaBar + b_comb_BaBar
    return sig, bkg, th, rec, NiB, Nnorm


@nb.njit(**kwd)
def _nll_BaBar(mu, p, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "-2 ln L of BaBar for signal strengths mu = (mu_SM, mu_NP_1, mu_NP_2) and nuisance block p."
    sig, bkg, th, rec, NiB, Nnorm = _model_BaBar(p, s_NP_1, s_NP_2, R_NP_1, R_NP_2)
    exp = (1 + rec) * _mix(mu, sig) + (1 + Nnorm) * (1 + th) * bkg
    u = p[0:NU_BaBar]
    penalty = u @ u + th @ (_mc_weight_BaBar * th)
    penalty += (rec / sigma_rec) ** 2 + (NiB / sigma_NiB) ** 2 + (Nnorm / sigma_Nnorm) ** 2
    return _poisson(exp, obs_BaBar) + penalty


@nb.njit(**kwd)
def _nll_grad_BaBar(mu, p, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "Gradient of _nll_BaBar w.r.t. mu and p (see _nll_grad_BelleII)."
    N, U = Nbins_BaBar, NU_BaBar
    sig, bkg, th, rec, NiB, Nnorm = _model_BaBar(p, s_NP_1, s_NP_2, R_NP_1, R_NP_2)
    signal = _mix(mu, sig)
    shift = (1 + Nnorm) * (1 + th)
    g = 2 * (1 - obs_BaBar / ((1 + rec) * signal + shift * bkg))

    d_p = np.empty(NPAR_BaBar)
    d_p[0:U] = (1 + rec) * (
        R_SM_BaBar.T @ (g * mu[0] * sig[0]) + R_NP_1.T @ (g * mu[1] * sig[1]) + R_NP_2.T @ (g * mu[2] * sig[2])
    ) + 2 * p[0:U]
    d_p[U : U + N] = g * (1 + Nnorm) * bkg + 2 * _mc_weight_BaBar * th
    d_p[U + N] = g @ signal + g @ (shift * (1 + NiB) * b_peak_BaBar) + 2 * rec / sigma_rec**2
    d_p[U + N + 1] = g @ (shift * (1 + rec) * b_peak_BaBar) + 2 * NiB / sigma_NiB**2
    d_p[U + N + 2] = g @ ((1 + th) * bkg) + 2 * Nnorm / sigma_Nnorm**2
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


# Belle II public likelihood (ITA)
# Belle II's released HistFactory model (arXiv:2507.12393), evaluated from arrays
# (lights.histfactory): 24 bins (regions A and B, on- and off-resonance), 7 background
# samples per on-resonance channel (5 off-resonance), with pyhf's interpolation of the
# histosys (code4p) and normsys (code4) modifiers and per-bin staterror factors gamma. The
# signal is ours: sum_i mu_i s_i exp(R_i u), with the form factor nuisances u replacing Belle
# II's signal form factor modifiers (corr_ff_c1-3, dropped), and Belle II's detector
# systematics acting on it as relative shifts.
#
# Nuisance block p (length NPAR_BelleII_public), with U = NU_BelleII_public:
#   p[0:U]              u       input nuisance params. (B -> K form factors)
#   p[U:U+H]            alpha   histosys params. (detector, background shapes), N(0, 1)
#   p[U+H:U+H+M]        alpha   normsys params. (background normalisations), N(0, 1)
#   p[U+H+M:]           z       staterror params., gamma = 1 + sigma z, N(0, 1)
HF_BelleII_public = histfactory.build(BelleII_public.workspace, drop=("corr_ff_c1", "corr_ff_c2", "corr_ff_c3"))
_hf = HF_BelleII_public
Nbins_BelleII_public = len(_hf.obs)
NH_pub, NM_pub, NG_pub = len(_hf.histosys_names), len(_hf.normsys_names), len(_hf.staterror_sigma)
NPAR_BelleII_public = NU_BelleII_public + NH_pub + NM_pub + NG_pub
obs_pub, bkg_nom_pub = _hf.obs, _hf.bkg_nominal
bkg_hs_up_pub, bkg_hs_dn_pub, sig_hs_up_pub, sig_hs_dn_pub = _hf.bkg_hs_up, _hf.bkg_hs_dn, _hf.sig_hs_up, _hf.sig_hs_dn
bkg_ns_up_pub, bkg_ns_dn_pub, bkg_ns_coef_pub = _hf.bkg_ns_up, _hf.bkg_ns_dn, _hf.bkg_ns_coef
stat_sigma_pub, bkg_stat_pub, sig_stat_pub = _hf.staterror_sigma, _hf.bkg_stat, _hf.sig_stat
# Which histosys and normsys parameters act on each background sample (padded with -1)
_hs_on = np.any(bkg_hs_up_pub != 0, axis=2) | np.any(bkg_hs_dn_pub != 0, axis=2)
_ns_on = (bkg_ns_up_pub != 1) | (bkg_ns_dn_pub != 1)
bkg_hs_idx_pub = np.full((len(bkg_nom_pub), NH_pub), -1, dtype=np.int64)
bkg_ns_idx_pub = np.full((len(bkg_nom_pub), NM_pub), -1, dtype=np.int64)
for _s in range(len(bkg_nom_pub)):
    bkg_hs_idx_pub[_s, : _hs_on[_s].sum()] = np.flatnonzero(_hs_on[_s])
    bkg_ns_idx_pub[_s, : _ns_on[_s].sum()] = np.flatnonzero(_ns_on[_s])
# Bins of each background sample (its channel)
bkg_bins_pub = np.array([[b0, b0 + n] for name, _ in _hf.bkg_names for (c, b0, n) in _hf.channel_bins if c == name])
R_SM_BelleII_public = _relative(s_SM_BelleII_public, J_SM_BelleII_public)


@nb.njit(**kwd)
def _code4p(a, up, dn):
    "pyhf code4p histosys interpolation: additive shift and its derivative w.r.t. alpha."
    if a > 1:
        return a * up, up
    if a < -1:
        return a * dn, dn
    S, A, a2 = 0.5 * (up + dn), 0.0625 * (up - dn), a * a
    return a * S + a2 * (a2 * (3 * a2 - 10) + 15) * A, S + a * (a2 * (18 * a2 - 40) + 30) * A


@nb.njit(**kwd)
def _code4(a, up, dn, coef):
    "pyhf code4 normsys interpolation: multiplicative factor and its derivative w.r.t. alpha."
    if a >= 1:
        f = up**a
        return f, f * np.log(up)
    if a <= -1:
        f = dn ** (-a)
        return f, -f * np.log(dn)
    f, df, ak = 1.0, 0.0, 1.0
    for i in range(6):
        df += (i + 1) * coef[i] * ak
        ak *= a
        f += coef[i] * ak
    return f, df


@nb.njit(**kwd)
def _nll_BelleII_public(mu, p, s_NP_1, s_NP_2, R_NP_1, R_NP_2, want_grad):
    """-2 ln L of the public Belle II likelihood (ITA) and, if want_grad, its gradient w.r.t. mu and p."""
    U, H, M = NU_BelleII_public, NH_pub, NM_pub
    N = Nbins_BelleII_public
    u, ah, am, z = p[0:U], p[U : U + H], p[U + H : U + H + M], p[U + H + M :]
    g = stat_sigma_pub * z  # gamma - 1
    sig = _signals(s_SM_BelleII_public, R_SM_BelleII_public, s_NP_1, s_NP_2, R_NP_1, R_NP_2, u)
    T = _mix(mu, sig)

    # Signal: T (1 + sum_h shift_h) gamma
    rel, gs = np.ones(N), np.ones(N)
    for b in range(N):
        if sig_stat_pub[b] >= 0:
            gs[b] = 1 + g[sig_stat_pub[b]]
        for h in range(H):
            if sig_hs_up_pub[h, b] != 0 or sig_hs_dn_pub[h, b] != 0:
                rel[b] += _code4p(ah[h], sig_hs_up_pub[h, b], sig_hs_dn_pub[h, b])[0]
    lam = T * rel * gs

    # Backgrounds: (nominal + sum_h shift_h) prod_m factor_m gamma
    S = len(bkg_nom_pub)
    F = np.ones(S)
    for s in range(S):
        for k in range(M):
            m = bkg_ns_idx_pub[s, k]
            if m < 0:
                break
            F[s] *= _code4(am[m], bkg_ns_up_pub[s, m], bkg_ns_dn_pub[s, m], bkg_ns_coef_pub[s, m])[0]
        for b in range(bkg_bins_pub[s, 0], bkg_bins_pub[s, 1]):
            A = bkg_nom_pub[s, b]
            for k in range(H):
                h = bkg_hs_idx_pub[s, k]
                if h < 0:
                    break
                A += _code4p(ah[h], bkg_hs_up_pub[s, h, b], bkg_hs_dn_pub[s, h, b])[0]
            gb = 1 + g[bkg_stat_pub[s, b]] if bkg_stat_pub[s, b] >= 0 else 1.0
            lam[b] += A * F[s] * gb

    f = _poisson(lam, obs_pub) + u @ u + ah @ ah + am @ am + z @ z
    if not want_grad:
        return f, np.zeros(3), np.zeros(0)

    # Gradient: df/dlam = 2 (1 - n / lam), then the chain rule through each term above
    G = 2 * (1 - obs_pub / lam)
    d_p = np.zeros(len(p))
    d_u, d_ah, d_am, d_g = d_p[0:U], d_p[U : U + H], d_p[U + H : U + H + M], d_p[U + H + M :]
    w = G * rel * gs  # d f / d T
    d_mu = np.array((sig[0] @ w, sig[1] @ w, sig[2] @ w))
    d_u += (
        R_SM_BelleII_public.T @ (w * mu[0] * sig[0]) + R_NP_1.T @ (w * mu[1] * sig[1]) + R_NP_2.T @ (w * mu[2] * sig[2])
    )
    d_u += 2 * u
    for b in range(N):
        if T[b] == 0:
            continue
        for h in range(H):
            if sig_hs_up_pub[h, b] != 0 or sig_hs_dn_pub[h, b] != 0:
                d_ah[h] += G[b] * T[b] * gs[b] * _code4p(ah[h], sig_hs_up_pub[h, b], sig_hs_dn_pub[h, b])[1]
        if sig_stat_pub[b] >= 0:
            d_g[sig_stat_pub[b]] += G[b] * T[b] * rel[b]
    for s in range(S):
        dlogF = np.zeros(M)
        for k in range(M):
            m = bkg_ns_idx_pub[s, k]
            if m < 0:
                break
            val, der = _code4(am[m], bkg_ns_up_pub[s, m], bkg_ns_dn_pub[s, m], bkg_ns_coef_pub[s, m])
            dlogF[m] = der / val
        for b in range(bkg_bins_pub[s, 0], bkg_bins_pub[s, 1]):
            A = bkg_nom_pub[s, b]
            gb = 1 + g[bkg_stat_pub[s, b]] if bkg_stat_pub[s, b] >= 0 else 1.0
            for k in range(H):
                h = bkg_hs_idx_pub[s, k]
                if h < 0:
                    break
                val, der = _code4p(ah[h], bkg_hs_up_pub[s, h, b], bkg_hs_dn_pub[s, h, b])
                A += val
                d_ah[h] += G[b] * der * F[s] * gb
            for k in range(M):
                m = bkg_ns_idx_pub[s, k]
                if m < 0:
                    break
                d_am[m] += G[b] * A * F[s] * gb * dlogF[m]
            if bkg_stat_pub[s, b] >= 0:
                d_g[bkg_stat_pub[s, b]] += G[b] * A * F[s]
    d_ah += 2 * ah
    d_am += 2 * am
    d_g *= stat_sigma_pub  # d/dz = sigma d/dgamma
    d_g += 2 * z
    return f, d_mu, d_p


@nb.njit(**kwd)
def logLL_BelleII_public(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "-2 times the log likelihood (minus constant factors) of Belle II's public likelihood (ITA)."
    return _nll_BelleII_public(_strengths(x), x[3:], s_NP_1, s_NP_2, R_NP_1, R_NP_2, False)[0]


@nb.njit(**kwd)
def grad_BelleII_public(x, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "Gradient of logLL_BelleII_public w.r.t. x."
    _, d_mu, d_p = _nll_BelleII_public(_strengths(x), x[3:], s_NP_1, s_NP_2, R_NP_1, R_NP_2, True)
    return _chain(x, d_mu, d_p)


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
        "peak": shift * (1 + rec) * (1 + NiB) * b_peak_BaBar,
        "comb": shift * b_comb_BaBar,
    }


# NP templates
_NBINS = {"BelleII": Nbins, "BaBar": Nbins_BaBar, "BelleII_public": Nbins_BelleII_public}
_NU = {"BelleII": NU_BelleII, "BaBar": NU_BaBar, "BelleII_public": NU_BelleII_public}


class Templates(NamedTuple):
    """NP signal templates of one experiment: the binned signals of the two NP models at unit
    coupling, s_NP_1 and s_NP_2 (Nbins,), and their relative responses to the input nuisance
    params., R_NP_1 and R_NP_2 (Nbins, NU). Unpacks into the likelihood arguments:
    logLL_BelleII(x, *templates)."""

    s_NP_1: np.ndarray
    s_NP_2: np.ndarray
    R_NP_1: np.ndarray
    R_NP_2: np.ndarray


def _fill_zero(experiment, s_NP_1, s_NP_2, R_NP_1, R_NP_2):
    "Templates with missing (None) entries replaced by zeros of the right shape."
    z1, z2 = np.zeros(_NBINS[experiment]), np.zeros((_NBINS[experiment], _NU[experiment]))
    return Templates(
        z1 if s_NP_1 is None else s_NP_1,
        z1 if s_NP_2 is None else s_NP_2,
        z2 if R_NP_1 is None else R_NP_1,
        z2 if R_NP_2 is None else R_NP_2,
    )


def np_templates(experiment, model_1=None, model_2=None):
    """NP Templates of two Models for one experiment ("BelleII" or "BaBar").

    A missing model (None) or one with no signal in this experiment gives zero templates.
    """

    def one(model):
        if model is None or not np.any(model.C):
            return None, None
        s0, J = signal_template(experiment, model)
        return s0, _relative(s0, J)

    (s1, R1), (s2, R2) = one(model_1), one(model_2)
    return _fill_zero(experiment, s1, s2, R1, R2)
