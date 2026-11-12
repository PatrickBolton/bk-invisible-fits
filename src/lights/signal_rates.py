"""Binned signal predictions for Belle II (B+ -> K+ E_miss) and BaBar (B0 -> K*0 E_miss).

The new-physics signal is described by a lights.models.Model (None: no new physics);
include_SM adds the SM B -> K(*) nu nu-bar signal.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from tqdm import tqdm

from lights.load_data import (sm, factorise, eff_vals_gen, bin_cents, bw, eff_bin_cents, eff_vals, eff_cov,
                              NB_BaBar, bin_cents_BaBar, bw_BaBar, eff_bin_cents_BaBar,
                              eff_BaBar_K0_vals, eff_BaBar_K0_cov)
from lights.constants import NB, mB, mK
from lights.form_factors import aK_vals_GRDV, aKs_vals_GRDV, aK_cov_GRDV, aKs_cov_GRDV, A_VALS_GRDV, a_vals_gen
from lights.branching_fractions import dBdqsq_SM
from lights.models import Model, K, KS, B_tot, dBdqsq_tot, from_legacy


def efficiency(q2_points, eff_points, kind):
    """Signal efficiency as a function of q^2, interpolated through the given points.

    Clipped at zero: near the kinematic endpoint, where the efficiency vanishes, the spline
    interpolation would otherwise dip slightly negative.
    """
    f = interp1d(q2_points, eff_points, kind = kind, fill_value = "extrapolate")
    return lambda q2: np.maximum(f(q2), 0.)

def _is_three_body(model):
    return model is not None and not model.two_body

def _is_two_body(model):
    return model is not None and model.two_body

def _legacy_nrec(fname, args, kwargs):
    "Old call Nrec(SM, mX, cNP, Lambda, is_ident, a_vals, eff_vals) -> (model, include_SM, extra kwargs)."
    vals = from_legacy(fname, args, kwargs, ("SM", "mX", "cNP", "Lambda", "is_ident", "a_vals", "eff_vals"),
                       {"Lambda": 1., "is_ident": False})
    extra = {k: vals[k] for k in ("a_vals", "eff_vals", "ntrue", "npts") if k in vals}
    return vals["cNP"], bool(vals["SM"]), extra


### Belle II (ITA) ###
_QSQ_MAX_K = (mB - mK)**2

def _qsqtrue_grid(model, ntrue):
    "True-q^2 grid from the pair threshold 4 mX^2 to the kinematic endpoint."
    return np.linspace(min(4*model.mX**2, _QSQ_MAX_K - 1e-10), _QSQ_MAX_K, ntrue)

def dNdqsqrec_BelleII(qsqrec, model = None, include_SM = False, a_vals = A_VALS_GRDV, eff_vals = eff_vals, ntrue = 1000):
    """Number of B+ -> K+ invisible events at Belle II per unit reconstructed q^2.

    qsqrec may be a scalar or an ascending 1D array (evaluated in one vectorised call).
    """
    scalar = np.ndim(qsqrec) == 0
    qsqrec = np.atleast_1d(qsqrec)
    eff = efficiency(eff_bin_cents, eff_vals, 3)
    dNdqsqrec = np.zeros(len(qsqrec))

    if _is_two_body(model):     # delta function at true q^2 = mX^2
        dNdqsqrec += sm(qsqrec, model.mX**2)[:, 0] * B_tot(model, a_vals)[K] * eff(model.mX**2)

    if _is_three_body(model):   # integrate the smearing over true q^2
        qt = _qsqtrue_grid(model, ntrue)
        dNdqsqrec += simpson(sm(qsqrec, qt) * (dBdqsq_tot(qt, model, a_vals)[K] * eff(qt)), x = qt, axis = 1)

    if include_SM:
        qt = np.linspace(4*(1e-10)**2, _QSQ_MAX_K, ntrue)
        dNdqsqrec += simpson(sm(qsqrec, qt) * (dBdqsq_SM(qt, a_vals = a_vals)[K] * eff(qt)), x = qt, axis = 1)

    dNdqsqrec = NB * dNdqsqrec
    return dNdqsqrec[0] if scalar else dNdqsqrec

# Bin-integrated smearing kernel. Swapping the order of integration,
#     N_bin = NB * int dq2_true  f(q2_true) * K_bin(q2_true),   K_bin(t) = int_bin dq2_rec sm(q2_rec, t),
# so the 2D smearing spline is integrated over each reconstructed bin once, here, on a fine
# true-q^2 grid. Each call then only interpolates K_bin. The smearing map is a clipped quadratic
# spline with knots at the bin centres, so the rec integral uses 41 points per bin.
def _bin_kernel(npts_rec = 41, qt = np.linspace(-1., 25., 2601)):
    n = npts_rec - 1
    qrec = np.linspace(bin_cents[0] - bw, bin_cents[-1] + bw, len(bin_cents)*n + 1)
    S = sm(qrec, qt)                                  # (n_rec, n_true)
    Kb = np.array([simpson(S[i*n:(i + 1)*n + 1], x = qrec[i*n:(i + 1)*n + 1], axis = 0)
                   for i in range(len(bin_cents))])  # (Nbins, n_true)
    return qt, Kb

_K_QT, _K_BIN = _bin_kernel()

def smearing_kernel(qsqtrue):
    "K_bin(q2_true): probability density in q2_true to be reconstructed in each Belle II bin, shape (Nbins, n)."
    qsqtrue = np.atleast_1d(qsqtrue)
    return np.array([np.interp(qsqtrue, _K_QT, Kb) for Kb in _K_BIN])

def Nrec_BelleII(model = None, *legacy_args, include_SM = False, a_vals = A_VALS_GRDV, eff_vals = eff_vals,
                 ntrue = 1000, **legacy_kwargs):
    """Number of B+ -> K+ invisible events in the reconstructed q^2 bins of Belle II.

    model: the new-physics Model (None: none); include_SM: add the SM signal.
    Integrates the true-q^2 spectrum against the bin-integrated smearing kernel (see
    smearing_kernel), which is equivalent to smearing and then integrating over each
    reconstructed bin. (The old call Nrec_BelleII(SM, mX, cNP, Lambda, ...) still works.)
    """
    if legacy_args or legacy_kwargs or not (model is None or isinstance(model, Model)):
        model, include_SM, extra = _legacy_nrec("Nrec_BelleII", (model,) + legacy_args, legacy_kwargs)
        return Nrec_BelleII(model, include_SM = include_SM, **{"a_vals": a_vals, "eff_vals": eff_vals,
                                                                 "ntrue": ntrue, **extra})

    eff = efficiency(eff_bin_cents, eff_vals, 3)
    Nrec_vals = np.zeros(len(bin_cents))

    if _is_two_body(model):     # delta function at true q^2 = mX^2
        Nrec_vals += smearing_kernel(model.mX**2)[:, 0] * B_tot(model, a_vals)[K] * eff(model.mX**2)

    if _is_three_body(model):
        qt = _qsqtrue_grid(model, ntrue)
        Nrec_vals += simpson(smearing_kernel(qt) * (dBdqsq_tot(qt, model, a_vals)[K] * eff(qt)), x = qt, axis = 1)

    if include_SM:
        qt = np.linspace(4*(1e-10)**2, _QSQ_MAX_K, ntrue)
        Nrec_vals += simpson(smearing_kernel(qt) * (dBdqsq_SM(qt, a_vals = a_vals)[K] * eff(qt)), x = qt, axis = 1)

    return NB * Nrec_vals


### BaBar ###
def dNdqsqrec_BaBar(qsqrec, model = None, include_SM = False, a_vals = A_VALS_GRDV, eff_vals = eff_BaBar_K0_vals):
    "Number of B0 -> K*0 invisible events at BaBar per unit q^2 (three-body and SM signal; no smearing)."
    eff = efficiency(eff_bin_cents_BaBar * mB**2, eff_vals, 3)
    rate = dBdqsq_tot(qsqrec, model, a_vals)[KS]
    if include_SM:
        rate = rate + dBdqsq_SM(qsqrec, a_vals = a_vals)[KS]
    return NB_BaBar * rate * eff(qsqrec)

def Nrec_BaBar(model = None, *legacy_args, include_SM = False, a_vals = A_VALS_GRDV, eff_vals = eff_BaBar_K0_vals,
               npts = 41, **legacy_kwargs):
    """Number of B0 -> K*0 invisible events in the q^2 bins of BaBar.

    model: the new-physics Model (None: none); include_SM: add the SM signal.
    BaBar has no q^2 smearing, so the 3-body pair threshold (4 mX^2) and the kinematic
    endpoint (mB - mKs)^2 appear as sharp edges inside bins. npts integration points per
    bin keeps the error below ~0.5% (3 points gave up to 11%).
    (The old call Nrec_BaBar(SM, mX, cNP, Lambda, ...) still works.)
    """
    if legacy_args or legacy_kwargs or not (model is None or isinstance(model, Model)):
        model, include_SM, extra = _legacy_nrec("Nrec_BaBar", (model,) + legacy_args, legacy_kwargs)
        return Nrec_BaBar(model, include_SM = include_SM, **{"a_vals": a_vals, "eff_vals": eff_vals,
                                                             "npts": npts, **extra})

    eff = efficiency(eff_bin_cents_BaBar * mB**2, eff_vals, 3)

    Nrec_vals = []
    for c in bin_cents_BaBar:
        lo, hi = c - bw_BaBar, c + bw_BaBar
        N = 0.

        if _is_three_body(model) or include_SM:
            # Avoid q^2 = 0 exactly, where the 3-body rates divide by q^2
            qsq_vals = np.linspace(max(lo, 1e-5), hi, npts) * mB**2
            N += simpson(dNdqsqrec_BaBar(qsq_vals, model, include_SM, a_vals, eff_vals), x = qsq_vals)

        # Two-body decay: delta function at q^2 = mX^2
        if _is_two_body(model) and lo * mB**2 < model.mX**2 < hi * mB**2:
            N += NB_BaBar * B_tot(model, a_vals)[KS] * eff(model.mX**2)

        Nrec_vals.append(N)

    return np.array(Nrec_vals)


### Theory uncertainty ###
def dNdqsq_cov(Ntoys, N_rec, model = None, include_SM = False, eff_vals = eff_vals, eff_cov = eff_cov, rng = None):
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

    bin_collect = np.array([N_rec(model, include_SM = include_SM, a_vals = a_vals_r[i], eff_vals = eff_vals_r[i])
                            for i in tqdm(range(Ntoys), desc="Processing")])

    return np.mean(bin_collect, axis = 0), np.cov(bin_collect, rowvar = False)

# Shared input nuisance parameters
# The uncertain inputs of each experiment's signal prediction are the form factor
# coefficients and the efficiency points. They are written as theta = theta0 + A u with
# u ~ N(0, 1) and A A^T = cov(theta) (see load_data.factorise), so u are independent
# unit Gaussians. Every binned template s (SM or NP) responds to them as
#     s(u) = s0 + J u,    J = ds/du
# to first order. The same u enter all templates of an experiment, so the theory
# uncertainties of the SM and NP signals are correlated, as they should be.

A_aK = factorise(aK_cov_GRDV)                   # B -> K form factors (Belle II)
A_aKs = factorise(aKs_cov_GRDV)                 # B -> K* form factors (BaBar)
A_eff_BelleII = factorise(eff_cov)
A_eff_BaBar = factorise(eff_BaBar_K0_cov)

# Per experiment: (index of the varied form factor set in a_vals, efficiency values,
#                  form factor basis, efficiency basis)
_INPUTS = {"BelleII": (0, eff_vals, A_aK, A_eff_BelleII),
           "BaBar": (1, eff_BaBar_K0_vals, A_aKs, A_eff_BaBar)}
_NREC = {"BelleII": Nrec_BelleII, "BaBar": Nrec_BaBar}

NU_BelleII = A_aK.shape[1] + A_eff_BelleII.shape[1]   # number of input nuisance params.
NU_BaBar = A_aKs.shape[1] + A_eff_BaBar.shape[1]

def signal_template(experiment, model = None, include_SM = False, h = 1.0):
    """Binned signal template s0 and its response J = ds/du to the input nuisance params.

    experiment is "BelleII" or "BaBar"; model the new-physics Model (None: none);
    include_SM adds the SM signal. J is computed by central differences with step h (in
    units of u). The prediction is linear in the efficiencies and quadratic in the form
    factor coefficients, so central differences give the exact derivative.
    Returns s0 (Nbins,) and J (Nbins, NU), with zero columns for inputs without variance.
    """
    slot, eff0, A_a, A_e = _INPUTS[experiment]
    Nrec = _NREC[experiment]
    a0 = [np.asarray(aK_vals_GRDV, dtype = float), np.asarray(aKs_vals_GRDV, dtype = float)]

    def rate(a_shift, e_shift):
        a = list(a0)
        a[slot] = a0[slot] + a_shift
        return np.asarray(Nrec(model, include_SM = include_SM, a_vals = a, eff_vals = eff0 + e_shift), dtype = float)

    s0 = rate(0., 0.)
    cols = []
    for k in range(A_a.shape[1]):
        d = A_a[:, k]
        cols.append((rate(h*d, 0.) - rate(-h*d, 0.))/(2*h) if d.any() else np.zeros_like(s0))
    for k in range(A_e.shape[1]):
        d = A_e[:, k]
        cols.append((rate(0., h*d) - rate(0., -h*d))/(2*h) if d.any() else np.zeros_like(s0))

    return s0, np.ascontiguousarray(np.column_stack(cols))

# SM templates and their responses to the input nuisance params.
s_SM_BelleII, J_SM_BelleII = signal_template("BelleII", include_SM = True)
s_SM_BaBar, J_SM_BaBar = signal_template("BaBar", include_SM = True)
