import numpy as np

from lights.load_data import (sm, factorise, eff_vals_gen, bin_cents, bw, eff_bin_cents, eff_vals, eff_cov,
                              NB_BaBar, bin_cents_BaBar, bw_BaBar, eff_bin_cents_BaBar,
                              eff_BaBar_K0_vals, eff_BaBar_K0_cov)
from lights.constants import NB, mB, mK, mKs
from lights.form_factors import aK_vals_GRDV, aKs_vals_GRDV, aK_cov_GRDV, aKs_cov_GRDV, A_VALS_GRDV, a_vals_gen
from lights.branching_fractions import dBdqsq_SM, dBdqsq_tot, B_tot

from scipy.interpolate import interp1d
from scipy.integrate import simpson
from tqdm import tqdm

### Belle-II (ITA) ###
def efficiency(q2_points, eff_points, kind):
    """Signal efficiency as a function of q^2, interpolated through the given points.

    Clipped at zero: near the kinematic endpoint, where the efficiency vanishes, the spline
    interpolation would otherwise dip slightly negative.
    """
    f = interp1d(q2_points, eff_points, kind = kind, fill_value = "extrapolate")
    return lambda q2: np.maximum(f(q2), 0.)

def dNdqsqrec_BelleII(qsqrec, SM, mX, cNP, Lambda, is_ident = False, a_vals = A_VALS_GRDV, eff_vals = eff_vals, ntrue = 1000):
    """Number of B+ -> K+ invisible events at Belle II in the reconstructed q^2.

    qsqrec may be a scalar or an ascending 1D array (evaluated in one vectorised call).
    """

    scalar = np.ndim(qsqrec) == 0
    qsqrec = np.atleast_1d(qsqrec)

    cX, cXY = cNP[0:2], cNP[2:6]

    eff = efficiency(eff_bin_cents, eff_vals, 3)

    # Two-body decay: delta function at true q^2 = mX^2
    dNdqsqrec = sm(qsqrec, mX**2)[:, 0] * B_tot(mX, cX, Lambda, a_vals)[0] * eff(mX**2)

    # Three-body decays: integrate the smearing over true q^2
    qsqtrue_vals = np.linspace(min(4*mX**2, (mB - mK)**2 - 1e-10), (mB - mK)**2, ntrue)
    dNdqsqrec += simpson(sm(qsqrec, qsqtrue_vals)
                         * (dBdqsq_tot(qsqtrue_vals, mX, cXY, Lambda, is_ident, a_vals)[0] * eff(qsqtrue_vals)),
                         x = qsqtrue_vals, axis = 1)

    if SM:
        qsqtrue_vals_SM = np.linspace(4*(1e-10)**2, (mB - mK)**2, ntrue)
        dNdqsqrec += simpson(sm(qsqrec, qsqtrue_vals_SM)
                             * (dBdqsq_SM(qsqtrue_vals_SM, False, a_vals)[0] * eff(qsqtrue_vals_SM)),
                             x = qsqtrue_vals_SM, axis = 1)

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
    K = np.array([simpson(S[i*n:(i + 1)*n + 1], x = qrec[i*n:(i + 1)*n + 1], axis = 0)
                  for i in range(len(bin_cents))])  # (Nbins, n_true)
    return qt, K

_K_QT, _K_BIN = _bin_kernel()

def smearing_kernel(qsqtrue):
    "K_bin(q2_true): probability density in q2_true to be reconstructed in each Belle II bin, shape (Nbins, n)."
    qsqtrue = np.atleast_1d(qsqtrue)
    return np.array([np.interp(qsqtrue, _K_QT, Kb) for Kb in _K_BIN])

def Nrec_BelleII(SM, mX, cNP, Lambda, is_ident = False, a_vals = A_VALS_GRDV, eff_vals = eff_vals, ntrue = 1000):
    """Number of events in reconstructed q^2 bins for B+ -> K+ invisible at Belle II.

    Integrates the true-q^2 spectrum against the bin-integrated smearing kernel (see
    smearing_kernel), which is equivalent to smearing and then integrating over each
    reconstructed bin.
    """

    cX, cXY = cNP[0:2], cNP[2:6]

    eff = efficiency(eff_bin_cents, eff_vals, 3)

    # Two-body decay: delta function at true q^2 = mX^2
    Nrec_vals = smearing_kernel(mX**2)[:, 0] * B_tot(mX, cX, Lambda, a_vals)[0] * eff(mX**2)

    # Three-body decays
    qsqtrue_vals = np.linspace(min(4*mX**2, (mB - mK)**2 - 1e-10), (mB - mK)**2, ntrue)
    Nrec_vals += simpson(smearing_kernel(qsqtrue_vals)
                         * (dBdqsq_tot(qsqtrue_vals, mX, cXY, Lambda, is_ident, a_vals)[0] * eff(qsqtrue_vals)),
                         x = qsqtrue_vals, axis = 1)

    if SM:
        qsqtrue_vals_SM = np.linspace(4*(1e-10)**2, (mB - mK)**2, ntrue)
        Nrec_vals += simpson(smearing_kernel(qsqtrue_vals_SM)
                             * (dBdqsq_SM(qsqtrue_vals_SM, False, a_vals)[0] * eff(qsqtrue_vals_SM)),
                             x = qsqtrue_vals_SM, axis = 1)

    return NB * Nrec_vals


### BaBar ###
def dNdqsqrec_BaBar(qsqrec, SM, mX, cNP, Lambda, is_ident = False, a_vals = A_VALS_GRDV, eff_vals = eff_BaBar_K0_vals):
    "Number of events for B0 -> K*0 invisible in the reconstructed q^2."
    
    cX, cXY = cNP[0:2], cNP[2:6]
    
    eff = efficiency(eff_bin_cents_BaBar * mB**2, eff_vals, 3)
    
    if SM == True:
        dNdqsqrec = NB_BaBar * (dBdqsq_SM(qsqrec, False, a_vals)[1] * eff(qsqrec)
                                + dBdqsq_tot(qsqrec, mX, cXY, Lambda, is_ident, a_vals)[1] * eff(qsqrec))
    else:
        dNdqsqrec = NB_BaBar * (dBdqsq_tot(qsqrec, mX, cXY, Lambda, is_ident, a_vals)[1] * eff(qsqrec))                              
    
    return dNdqsqrec

def Nrec_BaBar(SM, mX, cNP, Lambda, is_ident = False, a_vals = A_VALS_GRDV, eff_vals = eff_BaBar_K0_vals, npts = 41):
    """Number of events in reconstructed q^2 bins for B0 -> K*0 invisible.

    BaBar has no q^2 smearing, so the 3-body pair threshold (4 mX^2) and the
    kinematic endpoint (mB - mKs)^2 appear as sharp edges inside bins. npts
    integration points per bin keeps the error below ~0.5% (3 points gave up to 11%).
    """

    cX = cNP[0:2]

    eff = efficiency(eff_bin_cents_BaBar * mB**2, eff_vals, 3)

    Nrec_vals = []
    for c in bin_cents_BaBar:
        lo, hi = c - bw_BaBar, c + bw_BaBar

        # Avoid q^2 = 0 exactly, where the 3-body rates divide by q^2
        qsq_vals = np.linspace(max(lo, 1e-5), hi, npts) * mB**2
        N = simpson(dNdqsqrec_BaBar(qsq_vals, SM, mX, cNP, Lambda, is_ident, a_vals, eff_vals), x = qsq_vals)

        # Two-body decay: delta function at q^2 = mX^2
        if lo * mB**2 < mX**2 < hi * mB**2:
            N += NB_BaBar * B_tot(mX, cX, Lambda, a_vals)[1] * eff(mX**2)

        Nrec_vals.append(N)

    return np.array(Nrec_vals)


# Estimate covariance for NP signal
def dNdqsq_cov(Ntoys, N_rec, SM, mX, cNP, Lambda, is_ident = False, eff_vals = eff_vals, eff_cov = eff_cov, rng = None):
    """Theory covariance of the binned signal prediction (form factors and efficiencies).

    Each toy draws form factor coefficients and efficiencies, and N_rec propagates them
    (including the q^2 smearing for Belle II) to the reconstructed bins. The covariance
    is taken directly over these predictions. No events are sampled: counting
    fluctuations are already accounted for by the Poisson term of the likelihood.

    rng: a seed or numpy.random.Generator, for reproducible toys (None: fresh randomness).
    """
    rng = np.random.default_rng(rng)

    a_vals_r = a_vals_gen(Ntoys, rng)
    eff_vals_r = np.clip(eff_vals_gen(Ntoys, eff_vals, eff_cov, rng), 0, None)  # efficiencies cannot be negative

    bin_collect = np.array([N_rec(SM, mX, cNP, Lambda, is_ident, a_vals_r[i], eff_vals_r[i])
                            for i in tqdm(range(Ntoys), desc="Processing")])

    return np.mean(bin_collect, axis = 0), np.cov(bin_collect, rowvar = False)
### Theory uncertainty: shared input nuisance parameters ###
# The uncertain inputs of each experiment's signal prediction are the form factor
# coefficients and the efficiency points. They are written as theta = theta0 + A u with
# u ~ N(0, 1) and A A^T = cov(theta) (see load_data.factorise), so u are independent
# unit Gaussians. Every binned template s (SM or NP) responds linearly to them:
#     s(u) = s0 + J u,    J = ds/du
# The same u enter all templates of an experiment, so the theory uncertainties of the SM
# and NP signals are correlated, as they should be (same form factors and efficiencies).

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

def signal_template(experiment, SM, mX, cNP, Lambda, is_ident = False, h = 1.0):
    """Binned signal template s0 and its response J = ds/du to the input nuisance params.

    experiment is "BelleII" or "BaBar". J is computed by central differences with step
    h (in units of u). The prediction is linear in the efficiencies and quadratic in the
    form factor coefficients, so central differences give the exact derivative.
    Returns s0 (Nbins,) and J (Nbins, NU), with zero columns for inputs without variance.
    """
    slot, eff0, A_a, A_e = _INPUTS[experiment]
    Nrec = _NREC[experiment]
    a0 = [np.asarray(aK_vals_GRDV, dtype = float), np.asarray(aKs_vals_GRDV, dtype = float)]

    def rate(a_shift, e_shift):
        a = list(a0)
        a[slot] = a0[slot] + a_shift
        return np.asarray(Nrec(SM, mX, cNP, Lambda, is_ident, a, eff0 + e_shift), dtype = float)

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
s_SM_BelleII, J_SM_BelleII = signal_template("BelleII", 1, 1., np.zeros((6, 14)), 1)
s_SM_BaBar, J_SM_BaBar = signal_template("BaBar", 1, 1., np.zeros((6, 14)), 1)
