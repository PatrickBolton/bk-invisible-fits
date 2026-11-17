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


# Parameter layout, for n NP templates
# ------------------------------------
# x[0]          mu_SM, SM signal strength
# x[1:1+n]      NP couplings; the NP signal strengths are x[k]**2
# x[1+n:]       nuisance parameters (one block per experiment, see below)
#
# The NP templates are passed to the likelihoods as arguments, not as fit parameters: the
# binned signals s_NP (n, Nbins) at unit coupling and their relative responses
# R_NP = (ds_NP/du) / s_NP (n, Nbins, NU). The fits use n = 2 (two NP models; a missing one has
# zero templates), and n = 5 for the weak effective theory (lights.fits.profile_logLL_WET).
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
def _signals(s_SM, R_SM, s_NP, R_NP, u):
    "Signal templates at the input nuisance params u: the SM one (Nbins,) and the NP ones (n, Nbins)."
    sig_NP = np.empty_like(s_NP)
    for k in range(s_NP.shape[0]):
        sig_NP[k] = s_NP[k] * np.exp(R_NP[k] @ u)
    return s_SM * np.exp(R_SM @ u), sig_NP


@nb.njit(**kwd)
def _total(mu, sig_SM, sig_NP):
    "Total signal sum_i mu_i sig_i, with mu = (mu_SM, mu_NP_1, ..., mu_NP_n)."
    out = mu[0] * sig_SM
    for k in range(sig_NP.shape[0]):
        out = out + mu[1 + k] * sig_NP[k]
    return out


@nb.njit(**kwd)
def _d_strengths(w, sig_SM, sig_NP):
    "Derivatives of sum_b w_b (total signal)_b w.r.t. the strengths mu."
    out = np.empty(1 + sig_NP.shape[0])
    out[0] = sig_SM @ w
    for k in range(sig_NP.shape[0]):
        out[1 + k] = sig_NP[k] @ w
    return out


@nb.njit(**kwd)
def _d_inputs(w, mu, sig_SM, R_SM, sig_NP, R_NP):
    "Derivatives of sum_b w_b (total signal)_b w.r.t. the input nuisance params u."
    out = R_SM.T @ (w * mu[0] * sig_SM)
    for k in range(sig_NP.shape[0]):
        out = out + R_NP[k].T @ (w * mu[1 + k] * sig_NP[k])
    return out


@nb.njit(**kwd)
def _mix(c, rows):
    "c[0] rows[0] + c[1] rows[1] + c[2] rows[2], for three coefficients and three arrays."
    return c[0] * rows[0] + c[1] * rows[1] + c[2] * rows[2]


@nb.njit(**kwd)
def _exp_BelleII(mu, sig, bkg, mu_bkg):
    "Expected Belle II events from the components of _model_BelleII."
    return _total(mu, sig[0], sig[1]) + _mix((1 + mu_bkg[0], 1 + mu_bkg[1], 1 + mu_bkg[2]), bkg)


@nb.njit(**kwd)
def _poisson(exp, obs):
    "-2 ln of the Poisson likelihood of obs given exp, summed over bins (constant terms dropped)."
    return 2 * np.sum(exp - obs * np.log(exp))


@nb.njit(**kwd)
def _chain(x, d_mu, d_p):
    "Gradient w.r.t. x from the one w.r.t. the signal strengths (mu_SM, x_k^2) and the nuisance params."
    h = len(d_mu)
    out = np.empty(h + len(d_p))
    out[0] = d_mu[0]
    for k in range(1, h):
        out[k] = 2 * x[k] * d_mu[k]
    out[h:] = d_p
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
def _model_BelleII(p, s_NP, R_NP):
    """Belle II components at the nuisance block p.

    Returns the signals (SM, NP (n, Nbins)) at unit strength; the backgrounds (cont, charged,
    neutral) before normalisation, b + w z; and their normalisation params mu_bkg.
    The expected events are sum_i mu_i sig_i + sum_k (1 + mu_bkg_k) bkg_k.
    """
    N, U = Nbins, NU_BelleII
    sig = _signals(s_SM_BelleII, R_SM_BelleII, s_NP, R_NP, p[0:U])
    bkg = (
        b_cont + w_cont * p[U : U + N],
        b_charged + w_charged * p[U + N : U + 2 * N],
        b_neutral + w_neutral * p[U + 2 * N : U + 3 * N],
    )
    return sig, bkg, (p[U + 3 * N], p[U + 3 * N + 1], p[U + 3 * N + 2])


@nb.njit(**kwd)
def _nll_BelleII(mu, p, s_NP, R_NP):
    "-2 ln L of Belle II for signal strengths mu = (mu_SM, mu_NP_1, ..., mu_NP_n) and nuisance block p."
    U = NU_BelleII
    sig, bkg, mu_bkg = _model_BelleII(p, s_NP, R_NP)
    exp = _exp_BelleII(mu, sig, bkg, mu_bkg)
    u, z = p[0:U], p[U : NPAR_BelleII - 3]
    return _poisson(exp, obs) + u @ u + z @ z + (mu_bkg[0] ** 2 + mu_bkg[1] ** 2 + mu_bkg[2] ** 2) / BKG_NORM_SIGMA**2


@nb.njit(**kwd)
def _nll_grad_BelleII(mu, p, s_NP, R_NP):
    """Gradient of _nll_BelleII w.r.t. mu and p.

    With f = 2 sum(e - o ln e) + penalties, df/de = 2 (1 - o/e); the chain rule through e
    gives the rest. Checked against finite differences in scripts/check_gradients.py.
    """
    N, U = Nbins, NU_BelleII
    sig, bkg, mu_bkg = _model_BelleII(p, s_NP, R_NP)
    g = 2 * (1 - obs / _exp_BelleII(mu, sig, bkg, mu_bkg))

    d_p = np.empty(NPAR_BelleII)
    d_p[0:U] = _d_inputs(g, mu, sig[0], R_SM_BelleII, sig[1], R_NP) + 2 * p[0:U]
    for k in range(3):
        d_p[U + k * N : U + (k + 1) * N] = g * (1 + mu_bkg[k]) * _w_BelleII[k] + 2 * p[U + k * N : U + (k + 1) * N]
        d_p[U + 3 * N + k] = g @ bkg[k] + 2 * mu_bkg[k] / BKG_NORM_SIGMA**2
    return _d_strengths(g, sig[0], sig[1]), d_p


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
def _model_BaBar(p, s_NP, R_NP):
    """BaBar components at the nuisance block p.

    Returns the signals (SM, NP (n, Nbins)) at unit strength; the background
    shape (1 + rec)(1 + NiB) b_peak + b_comb before the shifts (1 + Nnorm)(1 + th_bkg); and
    the nuisance params th_bkg, rec, NiB, Nnorm. The expected events are
    (1 + rec) sum_i mu_i sig_i + (1 + Nnorm)(1 + th_bkg) bkg.
    """
    N, U = Nbins_BaBar, NU_BaBar
    sig = _signals(s_SM_BaBar, R_SM_BaBar, s_NP, R_NP, p[0:U])
    th, rec, NiB, Nnorm = p[U : U + N], p[U + N], p[U + N + 1], p[U + N + 2]
    bkg = (1 + rec) * (1 + NiB) * b_peak_BaBar + b_comb_BaBar
    return sig, bkg, th, rec, NiB, Nnorm


@nb.njit(**kwd)
def _nll_BaBar(mu, p, s_NP, R_NP):
    "-2 ln L of BaBar for signal strengths mu = (mu_SM, mu_NP_1, ..., mu_NP_n) and nuisance block p."
    sig, bkg, th, rec, NiB, Nnorm = _model_BaBar(p, s_NP, R_NP)
    exp = (1 + rec) * _total(mu, sig[0], sig[1]) + (1 + Nnorm) * (1 + th) * bkg
    u = p[0:NU_BaBar]
    penalty = u @ u + th @ (_mc_weight_BaBar * th)
    penalty += (rec / sigma_rec) ** 2 + (NiB / sigma_NiB) ** 2 + (Nnorm / sigma_Nnorm) ** 2
    return _poisson(exp, obs_BaBar) + penalty


@nb.njit(**kwd)
def _nll_grad_BaBar(mu, p, s_NP, R_NP):
    "Gradient of _nll_BaBar w.r.t. mu and p (see _nll_grad_BelleII)."
    N, U = Nbins_BaBar, NU_BaBar
    sig, bkg, th, rec, NiB, Nnorm = _model_BaBar(p, s_NP, R_NP)
    signal = _total(mu, sig[0], sig[1])
    shift = (1 + Nnorm) * (1 + th)
    g = 2 * (1 - obs_BaBar / ((1 + rec) * signal + shift * bkg))

    d_p = np.empty(NPAR_BaBar)
    d_p[0:U] = (1 + rec) * _d_inputs(g, mu, sig[0], R_SM_BaBar, sig[1], R_NP) + 2 * p[0:U]
    d_p[U : U + N] = g * (1 + Nnorm) * bkg + 2 * _mc_weight_BaBar * th
    d_p[U + N] = g @ signal + g @ (shift * (1 + NiB) * b_peak_BaBar) + 2 * rec / sigma_rec**2
    d_p[U + N + 1] = g @ (shift * (1 + rec) * b_peak_BaBar) + 2 * NiB / sigma_NiB**2
    d_p[U + N + 2] = g @ ((1 + th) * bkg) + 2 * Nnorm / sigma_Nnorm**2
    return (1 + rec) * _d_strengths(g, sig[0], sig[1]), d_p


# Public likelihoods and gradients of the parameter vector x
@nb.njit(**kwd)
def _strengths(x, n):
    "Signal strengths (mu_SM, mu_NP_1, ..., mu_NP_n) = (x0, x1^2, ..., xn^2)."
    mu = np.empty(1 + n)
    mu[0] = x[0]
    mu[1:] = x[1 : 1 + n] ** 2
    return mu


@nb.njit(**kwd)
def logLL_BelleII(x, s_NP, R_NP):
    "-2 times the log likelihood (minus constant factors) for the Belle II inclusive tag."
    h = 1 + s_NP.shape[0]
    return _nll_BelleII(_strengths(x, h - 1), x[h:], s_NP, R_NP)


@nb.njit(**kwd)
def logLL_BaBar(x, s_NP, R_NP):
    "-2 times the log likelihood (minus constant factors) for BaBar."
    h = 1 + s_NP.shape[0]
    return _nll_BaBar(_strengths(x, h - 1), x[h:], s_NP, R_NP)


# Combined: x[0:1+n] as above, then the Belle II nuisance block, then the BaBar nuisance block.
# The Belle II and BaBar inputs (B -> K vs B -> K* form factors, separate efficiencies) are
# independent, so the two blocks share no parameters apart from mu_SM and the NP couplings.
# Templates: the Belle II ones (s_NP, R_NP), then the BaBar ones.


@nb.njit(**kwd)
def logLL_Combined(x, s_B, R_B, s_b, R_b):
    "-2 times the log likelihood (minus constant factors) for combined experiments."
    h = 1 + s_B.shape[0]
    mu, pb = _strengths(x, h - 1), h + NPAR_BelleII  # pb: start of the BaBar block
    return _nll_BelleII(mu, x[h:pb], s_B, R_B) + _nll_BaBar(mu, x[pb:], s_b, R_b)


@nb.njit(**kwd)
def grad_BelleII(x, s_NP, R_NP):
    "Gradient of logLL_BelleII w.r.t. x."
    h = 1 + s_NP.shape[0]
    return _chain(x, *_nll_grad_BelleII(_strengths(x, h - 1), x[h:], s_NP, R_NP))


@nb.njit(**kwd)
def grad_BaBar(x, s_NP, R_NP):
    "Gradient of logLL_BaBar w.r.t. x."
    h = 1 + s_NP.shape[0]
    return _chain(x, *_nll_grad_BaBar(_strengths(x, h - 1), x[h:], s_NP, R_NP))


@nb.njit(**kwd)
def grad_Combined(x, s_B, R_B, s_b, R_b):
    "Gradient of logLL_Combined w.r.t. x."
    h = 1 + s_B.shape[0]
    mu, pb = _strengths(x, h - 1), h + NPAR_BelleII
    d_mu_B, d_p_B = _nll_grad_BelleII(mu, x[h:pb], s_B, R_B)
    d_mu_b, d_p_b = _nll_grad_BaBar(mu, x[pb:], s_b, R_b)
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
# Nuisance block p (length NPAR_BelleII_public_full), with U = NU_BelleII_public:
#   p[0:U]              u       input nuisance params. (B -> K form factors)
#   p[U:U+H]            alpha   histosys params. (detector, background shapes), N(0, 1)
#   p[U+H:U+H+M]        alpha   normsys params. (background normalisations), N(0, 1)
#   p[U+H+M:]           z       staterror params., gamma = 1 + sigma z, N(0, 1)
#
# The 156 staterror factors gamma are profiled analytically (_gamma_hat_public): within a bin,
# the Poisson term and their Gaussian constraints are minimised in closed form. The fitted
# likelihoods (logLL_BelleII_public, logLL_Combined_public) therefore take only the first
# NPAR_BelleII_public = U + H + M nuisance params.; the minimum over all params. is unchanged,
# and Minuit fits ~50 instead of ~200 params.
HF_BelleII_public = histfactory.build(BelleII_public.workspace, drop=("corr_ff_c1", "corr_ff_c2", "corr_ff_c3"))
_hf = HF_BelleII_public
Nbins_BelleII_public = len(_hf.obs)
NH_pub, NM_pub, NG_pub = len(_hf.histosys_names), len(_hf.normsys_names), len(_hf.staterror_sigma)
NPAR_BelleII_public = NU_BelleII_public + NH_pub + NM_pub  # fitted (staterror factors profiled)
NPAR_BelleII_public_full = NPAR_BelleII_public + NG_pub
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
def _nll_BelleII_public(mu, p, s_NP, R_NP, want_grad):
    """-2 ln L of the public Belle II likelihood (ITA) and, if want_grad, its gradient w.r.t. mu and p."""
    U, H, M = NU_BelleII_public, NH_pub, NM_pub
    N = Nbins_BelleII_public
    u, ah, am, z = p[0:U], p[U : U + H], p[U + H : U + H + M], p[U + H + M :]
    g = stat_sigma_pub * z  # gamma - 1
    sig_SM, sig_NP = _signals(s_SM_BelleII_public, R_SM_BelleII_public, s_NP, R_NP, u)
    T = _total(mu, sig_SM, sig_NP)

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
        return f, np.zeros(1 + s_NP.shape[0]), np.zeros(0)

    # Gradient: df/dlam = 2 (1 - n / lam), then the chain rule through each term above
    G = 2 * (1 - obs_pub / lam)
    d_p = np.zeros(len(p))
    d_u, d_ah, d_am, d_g = d_p[0:U], d_p[U : U + H], d_p[U + H : U + H + M], d_p[U + H + M :]
    w = G * rel * gs  # d f / d T
    d_mu = _d_strengths(w, sig_SM, sig_NP)
    d_u += _d_inputs(w, mu, sig_SM, R_SM_BelleII_public, sig_NP, R_NP)
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
def _gamma_hat_public(mu, q, s_NP, R_NP):
    """The staterror params. z (gamma = 1 + sigma z) minimising -2 ln L for the other params. q.

    In each bin, lambda = c + sum_s a_s gamma_s, where a_s is the yield of sample s before its
    staterror factor and c the yield of samples without one, and
        f = 2 (lambda - n ln lambda) + sum_s ((gamma_s - 1) / sigma_s)^2.
    df/dgamma_s = 0 gives gamma_s = 1 - sigma_s^2 a_s (1 - n / lambda), and then
        lambda^2 - (K - V) lambda - V n = 0,   K = c + sum_s a_s,   V = sum_s sigma_s^2 a_s^2,
    whose positive root is lambda. (Barlow-Beeston "lite", with one factor per sample.) A gamma_s
    that would be negative is held at its bound 0, and the bin is solved again without it.
    """
    U, H, M, N = NU_BelleII_public, NH_pub, NM_pub, Nbins_BelleII_public
    u, ah, am = q[0:U], q[U : U + H], q[U + H : U + H + M]
    sig_SM, sig_NP = _signals(s_SM_BelleII_public, R_SM_BelleII_public, s_NP, R_NP, u)
    T = _total(mu, sig_SM, sig_NP)
    S = len(bkg_nom_pub)

    # Yields before the staterror factors: row 0 the signal, rows 1 ... S the backgrounds
    a = np.zeros((S + 1, N))
    for b in range(N):
        r = 1.0
        for h in range(H):
            if sig_hs_up_pub[h, b] != 0 or sig_hs_dn_pub[h, b] != 0:
                r += _code4p(ah[h], sig_hs_up_pub[h, b], sig_hs_dn_pub[h, b])[0]
        a[0, b] = T[b] * r
    for s in range(S):
        F = 1.0
        for k in range(M):
            m = bkg_ns_idx_pub[s, k]
            if m < 0:
                break
            F *= _code4(am[m], bkg_ns_up_pub[s, m], bkg_ns_dn_pub[s, m], bkg_ns_coef_pub[s, m])[0]
        for b in range(bkg_bins_pub[s, 0], bkg_bins_pub[s, 1]):
            A = bkg_nom_pub[s, b]
            for k in range(H):
                h = bkg_hs_idx_pub[s, k]
                if h < 0:
                    break
                A += _code4p(ah[h], bkg_hs_up_pub[s, h, b], bkg_hs_dn_pub[s, h, b])[0]
            a[s + 1, b] = A * F

    z = np.zeros(NG_pub)
    idx, ab, sg, free = np.empty(S + 1, np.int64), np.empty(S + 1), np.empty(S + 1), np.empty(S + 1, np.bool_)
    for b in range(N):
        c, n_s = 0.0, 0
        for k in range(S + 1):
            j = sig_stat_pub[b] if k == 0 else bkg_stat_pub[k - 1, b]
            if j < 0:
                c += a[k, b]
            else:
                idx[n_s], ab[n_s], sg[n_s], free[n_s] = j, a[k, b], stat_sigma_pub[j], True
                n_s += 1
        n = obs_pub[b]
        while True:
            K, V = c, 0.0
            for i in range(n_s):
                if free[i]:
                    K += ab[i]
                    V += (sg[i] * ab[i]) ** 2
            d, root = K - V, np.sqrt((K - V) ** 2 + 4 * V * n)
            lam = 0.5 * (d + root) if d >= 0 else 2 * V * n / (root - d)  # positive root, without cancellation
            pull = 1 - n / lam if lam > 0 else 1.0
            changed = False
            for i in range(n_s):
                if free[i] and 1 - sg[i] ** 2 * ab[i] * pull < 0:
                    free[i], changed = False, True
            if not changed:
                break
        for i in range(n_s):
            gamma = 1 - sg[i] ** 2 * ab[i] * pull if free[i] else 0.0
            z[idx[i]] = (gamma - 1) / sg[i]
    return z


@nb.njit(**kwd)
def _nll_BelleII_public_profiled(mu, q, s_NP, R_NP, want_grad):
    """-2 ln L of the public likelihood with the staterror factors profiled, and its gradient w.r.t. mu and q.

    By the envelope theorem the gradient is that of the full likelihood at the profiled factors
    (their own derivatives vanish there, or they sit at their bound).
    """
    p = np.concatenate((q, _gamma_hat_public(mu, q, s_NP, R_NP)))
    f, d_mu, d_p = _nll_BelleII_public(mu, p, s_NP, R_NP, want_grad)
    return f, d_mu, d_p[: len(q)] if want_grad else d_p


@nb.njit(**kwd)
def logLL_BelleII_public(x, s_NP, R_NP):
    "-2 times the log likelihood (minus constant factors) of Belle II's public likelihood (ITA), staterrors profiled."
    h = 1 + s_NP.shape[0]
    return _nll_BelleII_public_profiled(_strengths(x, h - 1), x[h:], s_NP, R_NP, False)[0]


@nb.njit(**kwd)
def grad_BelleII_public(x, s_NP, R_NP):
    "Gradient of logLL_BelleII_public w.r.t. x."
    h = 1 + s_NP.shape[0]
    _, d_mu, d_q = _nll_BelleII_public_profiled(_strengths(x, h - 1), x[h:], s_NP, R_NP, True)
    return _chain(x, d_mu, d_q)


@nb.njit(**kwd)
def logLL_BelleII_public_full(x, s_NP, R_NP):
    """-2 ln L of Belle II's public likelihood with the staterror params. z as explicit params.

    x has NPAR_BelleII_public_full nuisance params. This is the HistFactory model itself (as
    in pyhf), used to validate the profiling; the fits use logLL_BelleII_public.
    """
    h = 1 + s_NP.shape[0]
    return _nll_BelleII_public(_strengths(x, h - 1), x[h:], s_NP, R_NP, False)[0]


# Combined with the public Belle II likelihood: x[0:1+n], then the public Belle II nuisance
# block, then the BaBar nuisance block (independent, as in logLL_Combined). Templates: the
# public Belle II ones, then the BaBar ones.


@nb.njit(**kwd)
def logLL_Combined_public(x, s_B, R_B, s_b, R_b):
    "-2 times the log likelihood (minus constant factors) of Belle II's public likelihood and BaBar."
    h = 1 + s_B.shape[0]
    mu, pb = _strengths(x, h - 1), h + NPAR_BelleII_public  # pb: start of the BaBar block
    belle = _nll_BelleII_public_profiled(mu, x[h:pb], s_B, R_B, False)[0]
    return belle + _nll_BaBar(mu, x[pb:], s_b, R_b)


@nb.njit(**kwd)
def grad_Combined_public(x, s_B, R_B, s_b, R_b):
    "Gradient of logLL_Combined_public w.r.t. x."
    h = 1 + s_B.shape[0]
    mu, pb = _strengths(x, h - 1), h + NPAR_BelleII_public
    _, d_mu_B, d_p_B = _nll_BelleII_public_profiled(mu, x[h:pb], s_B, R_B, True)
    d_mu_b, d_p_b = _nll_grad_BaBar(mu, x[pb:], s_b, R_b)
    return _chain(x, d_mu_B + d_mu_b, np.concatenate((d_p_B, d_p_b)))


# LEP (Bs -> inv): Gaussian in the number of events N_norm * B(Bs -> inv)
LEP_N_norm = 6607.14
LEP_mu = 0.525241
LEP_sigma = 2.47728


@nb.njit(**kwd)
def logLL_LEP(x, Br_LEP):
    """-2 times the log likelihood (minus constant factors) for LEP.

    x is a parameter vector as above (only the couplings x[1:1+n] enter); Br_LEP (n,) are
    the Bs -> invisible branching fractions of the NP models at unit coupling.
    """
    Br = np.sum(x[1 : 1 + len(Br_LEP)] ** 2 * Br_LEP)
    return (LEP_N_norm * Br - LEP_mu) ** 2 / LEP_sigma**2


@nb.njit(**kwd)
def grad_LEP(x, Br_LEP):
    "Gradient of logLL_LEP w.r.t. x."
    n = len(Br_LEP)
    Br = np.sum(x[1 : 1 + n] ** 2 * Br_LEP)
    c = 2 * (LEP_N_norm * Br - LEP_mu) / LEP_sigma**2 * LEP_N_norm
    out = np.zeros(len(x))
    out[1 : 1 + n] = c * 2 * x[1 : 1 + n] * Br_LEP
    return out


# Fitted components (for plotting)
def _components_args(experiment, x, s_NP, R_NP):
    "Parameter vector and templates of components_*: default zero templates for two NP models."
    x = np.asarray(x, dtype=float)
    if s_NP is None:
        s_NP, R_NP = np_templates(experiment)
    return x, s_NP, R_NP, 1 + s_NP.shape[0]


def components_BelleII(x, s_NP=None, R_NP=None):
    "Fitted Belle II signal and background components in each bin (signals SM, NP_1, ..., NP_n), from x."
    x, s_NP, R_NP, h = _components_args("BelleII", x, s_NP, R_NP)
    (sig_SM, sig_NP), bkg, mu_bkg = _model_BelleII(x[h:], s_NP, R_NP)
    mu = _strengths(x, h - 1)
    out = {"SM": mu[0] * sig_SM} | {f"NP_{k}": mu[k] * sig_NP[k - 1] for k in range(1, h)}
    out.update({name: (1 + mu_bkg[k]) * bkg[k] for k, name in enumerate(BKG_NAMES_BelleII)})
    return out


def components_BaBar(x, s_NP=None, R_NP=None):
    "Fitted BaBar signal and background components in each bin (signals SM, NP_1, ..., NP_n), from x."
    x, s_NP, R_NP, h = _components_args("BaBar", x, s_NP, R_NP)
    (sig_SM, sig_NP), _, th, rec, NiB, Nnorm = _model_BaBar(x[h:], s_NP, R_NP)
    mu = _strengths(x, h - 1)
    shift = (1 + Nnorm) * (1 + th)
    out = {"SM": (1 + rec) * mu[0] * sig_SM} | {f"NP_{k}": (1 + rec) * mu[k] * sig_NP[k - 1] for k in range(1, h)}
    return out | {"peak": shift * (1 + rec) * (1 + NiB) * b_peak_BaBar, "comb": shift * b_comb_BaBar}


# NP templates
_NBINS = {"BelleII": Nbins, "BaBar": Nbins_BaBar, "BelleII_public": Nbins_BelleII_public}
_NU = {"BelleII": NU_BelleII, "BaBar": NU_BaBar, "BelleII_public": NU_BelleII_public}


class Templates(NamedTuple):
    """NP signal templates of one experiment: the binned signals of the n NP models at unit
    coupling, s_NP (n, Nbins), and their relative responses to the input nuisance params.,
    R_NP (n, Nbins, NU). Unpacks into the likelihood arguments: logLL_BelleII(x, *templates)."""

    s_NP: np.ndarray
    R_NP: np.ndarray

    @property
    def n(self):
        "Number of NP templates."
        return self.s_NP.shape[0]


def np_templates(experiment, *models):
    """NP Templates of Models for one experiment ("BelleII", "BelleII_public" or "BaBar").

    One template per model, two at least (the fits of one or two models): a missing model
    (None) or one with no signal in this experiment gives a zero template.
    """
    models = list(models) + [None] * (2 - len(models))
    N, U = _NBINS[experiment], _NU[experiment]
    s_NP, R_NP = np.zeros((len(models), N)), np.zeros((len(models), N, U))
    for k, model in enumerate(models):
        if model is not None and np.any(model.C):
            s0, J = signal_template(experiment, model)
            s_NP[k], R_NP[k] = s0, _relative(s0, J)
    return Templates(s_NP, R_NP)
