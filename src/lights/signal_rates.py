"""Binned signal predictions for Belle II (B+ -> K+ E_miss) and BaBar (B0 -> K*0 E_miss).

The new-physics signal is described by a lights.models.Model (None: no new physics);
include_SM adds the SM B -> K(*) nu nu-bar signal.
"""

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from tqdm import tqdm

from lights.branching_fractions import dBdqsq_SM
from lights.constants import NB, mB, mK
from lights.form_factors import A_VALS_GRDV, a_vals_gen, aK_cov_GRDV, aK_vals_GRDV, aKs_cov_GRDV, aKs_vals_GRDV
from lights.load_data import (
    NB_BaBar,
    bin_cents,
    bin_cents_BaBar,
    bw,
    bw_BaBar,
    eff_BaBar_K0_cov,
    eff_BaBar_K0_vals,
    eff_bin_cents,
    eff_bin_cents_BaBar,
    eff_cov,
    eff_vals,
    eff_vals_gen,
    factorise,
    sm,
)
from lights.models import KS, B_tot, K, check_model, dBdqsq_tot


def efficiency(q2_points, eff_points, kind):
    """Signal efficiency as a function of q^2, interpolated through the given points.

    Clipped at zero: near the kinematic endpoint, where the efficiency vanishes, the spline
    interpolation would otherwise dip slightly negative.
    """
    f = interp1d(q2_points, eff_points, kind=kind, fill_value="extrapolate")
    return lambda q2: np.maximum(f(q2), 0.0)


def _is_three_body(model):
    return model is not None and not model.two_body


def _is_two_body(model):
    return model is not None and model.two_body


# Lowest q^2 of the integration grids: the three-body rates divide by q^2, so q^2 = 0 is avoided
_QSQ_MIN = 1e-12


### Belle II (ITA) ###
_QSQ_MAX_K = (mB - mK) ** 2


def _smeared_BelleII(kernel, model, include_SM, a_vals, eff_vals, ntrue):
    """NB * int dq2_true kernel(q2_true) eff(q2_true) dB/dq2_true, for the signal of model (and the SM).

    kernel(q2_true) gives the smearing to each reconstructed q^2 value or bin, shape
    (n_rec, len(q2_true)). A two-body signal is a delta function at q2_true = mX^2; three-body
    signals are integrated from the pair threshold 4 mX^2 to the kinematic endpoint.
    """
    eff = efficiency(eff_bin_cents, eff_vals, 3)
    out = 0.0

    if _is_two_body(model):
        out = out + kernel(model.mX**2)[:, 0] * B_tot(model, a_vals)[K] * eff(model.mX**2)

    if _is_three_body(model):
        qt = np.linspace(min(max(4 * model.mX**2, _QSQ_MIN), _QSQ_MAX_K - 1e-10), _QSQ_MAX_K, ntrue)
        out = out + simpson(kernel(qt) * (dBdqsq_tot(qt, model, a_vals)[K] * eff(qt)), x=qt, axis=1)

    if include_SM:
        qt = np.linspace(_QSQ_MIN, _QSQ_MAX_K, ntrue)
        out = out + simpson(kernel(qt) * (dBdqsq_SM(qt, a_vals=a_vals)[K] * eff(qt)), x=qt, axis=1)

    return NB * out


def dNdqsqrec_BelleII(qsqrec, model=None, *, include_SM=False, a_vals=A_VALS_GRDV, eff_vals=eff_vals, ntrue=1000):
    """Number of B+ -> K+ invisible events at Belle II per unit reconstructed q^2.

    qsqrec may be a scalar or an ascending 1D array (evaluated in one vectorised call).
    """
    q = np.atleast_1d(qsqrec)
    out = _smeared_BelleII(lambda qt: sm(q, qt), model, include_SM, a_vals, eff_vals, ntrue) + np.zeros(len(q))
    return out[0] if np.ndim(qsqrec) == 0 else out


# Bin-integrated smearing kernel. Swapping the order of integration,
#     N_bin = NB * int dq2_true  f(q2_true) * K_bin(q2_true),   K_bin(t) = int_bin dq2_rec sm(q2_rec, t),
# so the 2D smearing spline is integrated over each reconstructed bin once, here, on a fine
# true-q^2 grid. Each call then only interpolates K_bin. The smearing map is a clipped quadratic
# spline with knots at the bin centres, so the rec integral uses 41 points per bin.
def _bin_kernel(npts_rec=41, qt=None):
    qt = np.linspace(-1.0, 25.0, 2601) if qt is None else qt
    n = npts_rec - 1
    qrec = np.linspace(bin_cents[0] - bw, bin_cents[-1] + bw, len(bin_cents) * n + 1)
    S = sm(qrec, qt)  # (n_rec, n_true)
    Kb = np.array(
        [simpson(S[i * n : (i + 1) * n + 1], x=qrec[i * n : (i + 1) * n + 1], axis=0) for i in range(len(bin_cents))]
    )  # (Nbins, n_true)
    return qt, Kb


_K_QT, _K_BIN = _bin_kernel()


def smearing_kernel(qsqtrue):
    "K_bin(q2_true): probability density in q2_true to be reconstructed in each Belle II bin, shape (Nbins, n)."
    qsqtrue = np.atleast_1d(qsqtrue)
    return np.array([np.interp(qsqtrue, _K_QT, Kb) for Kb in _K_BIN])


def Nrec_BelleII(model=None, *, include_SM=False, a_vals=A_VALS_GRDV, eff_vals=eff_vals, ntrue=1000):
    """Number of B+ -> K+ invisible events in the reconstructed q^2 bins of Belle II.

    model: the new-physics Model (None: none); include_SM: add the SM signal.
    Integrates the true-q^2 spectrum against the bin-integrated smearing kernel (see
    smearing_kernel), which is equivalent to smearing and then integrating over each
    reconstructed bin.
    """
    check_model(model)
    return _smeared_BelleII(smearing_kernel, model, include_SM, a_vals, eff_vals, ntrue) + np.zeros(len(bin_cents))


### BaBar ###
def dNdqsqrec_BaBar(qsqrec, model=None, *, include_SM=False, a_vals=A_VALS_GRDV, eff_vals=eff_BaBar_K0_vals):
    "Number of B0 -> K*0 invisible events at BaBar per unit q^2 (three-body and SM signal; no smearing)."
    eff = efficiency(eff_bin_cents_BaBar * mB**2, eff_vals, 3)
    rate = dBdqsq_tot(qsqrec, model, a_vals)[KS]
    if include_SM:
        rate = rate + dBdqsq_SM(qsqrec, a_vals=a_vals)[KS]
    return NB_BaBar * rate * eff(qsqrec)


def Nrec_BaBar(model=None, *, include_SM=False, a_vals=A_VALS_GRDV, eff_vals=eff_BaBar_K0_vals, npts=41):
    """Number of B0 -> K*0 invisible events in the q^2 bins of BaBar.

    model: the new-physics Model (None: none); include_SM: add the SM signal.
    BaBar has no q^2 smearing, so the 3-body pair threshold (4 mX^2) and the kinematic
    endpoint (mB - mKs)^2 appear as sharp edges inside bins. npts integration points per
    bin keeps the error below ~0.5% (3 points gave up to 11%).
    """
    check_model(model)
    lo, hi = (bin_cents_BaBar - bw_BaBar) * mB**2, (bin_cents_BaBar + bw_BaBar) * mB**2  # bin edges in q^2
    Nrec_vals = np.zeros(len(bin_cents_BaBar))

    if _is_three_body(model) or include_SM:
        # npts points in each bin, all bins in one call (avoiding q^2 = 0, where the 3-body rates divide by q^2)
        qsq = np.linspace(np.maximum(lo, 1e-5 * mB**2), hi, npts, axis=1)  # (Nbins, npts)
        rate = dNdqsqrec_BaBar(qsq.ravel(), model, include_SM=include_SM, a_vals=a_vals, eff_vals=eff_vals)
        Nrec_vals += simpson(rate.reshape(qsq.shape), x=qsq, axis=1)

    if _is_two_body(model):  # delta function at q^2 = mX^2
        eff = efficiency(eff_bin_cents_BaBar * mB**2, eff_vals, 3)
        Nrec_vals += ((lo < model.mX**2) & (model.mX**2 < hi)) * NB_BaBar * B_tot(model, a_vals)[KS] * eff(model.mX**2)

    return Nrec_vals


### Theory uncertainty ###
def dNdqsq_cov(Ntoys, N_rec, model=None, *, include_SM=False, eff_vals=eff_vals, eff_cov=eff_cov, rng=None):
    """Theory covariance of the binned signal prediction (form factors and efficiencies) from toys.

    Each toy draws form factor coefficients and efficiencies, and N_rec (Nrec_BelleII or
    Nrec_BaBar) propagates them (including the q^2 smearing for Belle II) to the
    reconstructed bins. The covariance is taken directly over these predictions. No events
    are sampled: counting fluctuations are already accounted for by the Poisson term of the
    likelihood. Used to validate signal_template (scripts/validate_theory_cov.py).

    rng: a seed or numpy.random.Generator, for reproducible toys (None: fresh randomness).
    """
    rng = np.random.default_rng(rng)

    a_vals_r = a_vals_gen(Ntoys, rng)
    eff_vals_r = np.clip(eff_vals_gen(Ntoys, eff_vals, eff_cov, rng), 0, None)  # efficiencies cannot be negative

    bin_collect = np.array(
        [
            N_rec(model, include_SM=include_SM, a_vals=a_vals_r[i], eff_vals=eff_vals_r[i])
            for i in tqdm(range(Ntoys), desc="Processing")
        ]
    )

    return np.mean(bin_collect, axis=0), np.cov(bin_collect, rowvar=False)


# Shared input nuisance parameters
# The uncertain inputs of each experiment's signal prediction are the form factor
# coefficients and the efficiency points. They are written as theta = theta0 + A u with
# u ~ N(0, 1) and A A^T = cov(theta) (see load_data.factorise), so u are independent
# unit Gaussians. Every binned template s (SM or NP) responds to them as
#     s(u) = s0 + J u,    J = ds/du
# to first order. The same u enter all templates of an experiment, so the theory
# uncertainties of the SM and NP signals are correlated, as they should be.

A_aK = factorise(aK_cov_GRDV)  # B -> K form factors (Belle II)
A_aKs = factorise(aKs_cov_GRDV)  # B -> K* form factors (BaBar)
A_eff_BelleII = factorise(eff_cov)
A_eff_BaBar = factorise(eff_BaBar_K0_cov)

# Per experiment: (index of the varied form factor set in a_vals, efficiency values,
#                  form factor basis, efficiency basis)
_INPUTS = {"BelleII": (0, eff_vals, A_aK, A_eff_BelleII), "BaBar": (1, eff_BaBar_K0_vals, A_aKs, A_eff_BaBar)}
_NREC = {"BelleII": Nrec_BelleII, "BaBar": Nrec_BaBar}

NU_BelleII = A_aK.shape[1] + A_eff_BelleII.shape[1]  # number of input nuisance params.
NU_BaBar = A_aKs.shape[1] + A_eff_BaBar.shape[1]


def signal_template(experiment, model=None, *, include_SM=False, h=1.0):
    """Binned signal template s0 and its response J = ds/du to the input nuisance params.

    experiment is "BelleII" or "BaBar"; model the new-physics Model (None: none);
    include_SM adds the SM signal. J is computed by central differences with step h (in
    units of u). The prediction is linear in the efficiencies and quadratic in the form
    factor coefficients, so central differences give the exact derivative.
    Returns s0 (Nbins,) and J (Nbins, NU), with zero columns for inputs without variance.
    """
    slot, eff0, A_a, A_e = _INPUTS[experiment]
    Nrec = _NREC[experiment]
    a0 = [np.asarray(aK_vals_GRDV, dtype=float), np.asarray(aKs_vals_GRDV, dtype=float)]

    def rate(a_shift, e_shift):
        a = list(a0)
        a[slot] = a0[slot] + a_shift
        return np.asarray(Nrec(model, include_SM=include_SM, a_vals=a, eff_vals=eff0 + e_shift), dtype=float)

    s0 = rate(0.0, 0.0)
    cols = []
    for k in range(A_a.shape[1]):
        d = A_a[:, k]
        cols.append((rate(h * d, 0.0) - rate(-h * d, 0.0)) / (2 * h) if d.any() else np.zeros_like(s0))
    for k in range(A_e.shape[1]):
        d = A_e[:, k]
        cols.append((rate(0.0, h * d) - rate(0.0, -h * d)) / (2 * h) if d.any() else np.zeros_like(s0))

    return s0, np.ascontiguousarray(np.column_stack(cols))


# SM templates and their responses to the input nuisance params.
s_SM_BelleII, J_SM_BelleII = signal_template("BelleII", include_SM=True)
s_SM_BaBar, J_SM_BaBar = signal_template("BaBar", include_SM=True)
