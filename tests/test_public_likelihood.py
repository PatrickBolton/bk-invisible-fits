"""Belle II's public likelihood (ITA): the compiled HistFactory model against pyhf, and the reweighting."""

import copy

import numpy as np
import pyhf
import pytest
from scipy.integrate import quad

from lights import histfactory
from lights.branching_fractions import dBdqsq_SM
from lights.fits import min_logLL, profile_logLL_SM
from lights.likelihoods import (
    HF_BelleII_public,
    NH_pub,
    NM_pub,
    NPAR_BelleII_public,
    NU_BelleII_public,
    grad_BelleII_public,
    logLL_BelleII_public,
    np_templates,
    stat_sigma_pub,
)
from lights.load_data import BelleII_public, belle2_null_spectrum
from lights.models import Model
from lights.signal_rates import Nrec_BelleII_public, binned_B_BelleII_public, s_SM_BelleII_public

HF = HF_BelleII_public
FF = ["corr_ff_c1", "corr_ff_c2", "corr_ff_c3"]  # Belle II's signal form factor modifiers, replaced by ours


@pytest.fixture(scope="module")
def pyhf_model():
    ws = pyhf.Workspace(copy.deepcopy(BelleII_public.workspace)).prune(modifiers=FF)
    model = ws.model()
    return model, np.asarray(ws.data(model))


def to_pyhf(model, mu, ah, am, z):
    "pyhf parameters of our (mu, histosys, normsys, staterror z) point."
    pars = np.array(model.config.suggested_init(), dtype=float)
    stat_start = np.cumsum([0] + [model.config.param_set(n).n_parameters for n in HF.staterror_names])
    gamma = 1 + stat_sigma_pub * z
    for name in model.config.par_order:
        sl = model.config.par_slice(name)
        if name == "mu":
            pars[sl] = mu
        elif name in HF.histosys_names:
            pars[sl] = ah[HF.histosys_names.index(name)]
        elif name in HF.normsys_names:
            pars[sl] = am[HF.normsys_names.index(name)]
        else:
            k = HF.staterror_names.index(name)
            pars[sl] = gamma[stat_start[k] : stat_start[k + 1]]
    return pars


def test_likelihood_matches_pyhf(pyhf_model):
    # Belle II's own signal as the only template (strength x1^2 = mu, our SM switched off): the
    # compiled likelihood must then be pyhf's, also for |alpha| > 1 where the interpolation changes
    model, data = pyhf_model
    z, zR = np.zeros(len(HF.obs)), np.zeros((len(HF.obs), NU_BelleII_public))
    tpl = (HF.signal_nominal, z, zR, zR)
    rng = np.random.default_rng(3)
    ours, theirs = [], []
    for trial in range(10):
        mu = rng.uniform(0, 6)
        ah, am, zs = rng.normal(0, 1.5, NH_pub), rng.normal(0, 1.5, NM_pub), rng.normal(0, 1, len(stat_sigma_pub))
        x = np.r_[0.0, np.sqrt(mu), 0.0, np.zeros(NU_BelleII_public), ah, am, zs]
        ours.append(logLL_BelleII_public(x, *tpl))
        theirs.append(-2 * float(model.logpdf(to_pyhf(model, mu, ah, am, zs), data)[0]))
    ours, theirs = np.array(ours), np.array(theirs)
    # Large random shifts can make an expected yield negative: both are then NaN
    ok = np.isfinite(theirs)
    assert np.array_equal(ok, np.isfinite(ours)) and ok.sum() >= 5
    assert ours[ok] - ours[0] == pytest.approx(theirs[ok] - theirs[0], abs=1e-6)


def test_code4_polynomial_joins_the_exponentials_smoothly():
    up, dn = 1.3, 0.8
    a = histfactory.code4_coefficients(up, dn)
    poly = np.polynomial.Polynomial(np.r_[1.0, a])
    for alpha, f in ((1.0, lambda t: up**t), (-1.0, lambda t: dn ** (-t))):
        h = 1e-4
        for d in range(3):
            exact = (np.log(up) if alpha > 0 else -np.log(dn)) ** d * f(alpha)
            assert (
                poly.deriv(d)(alpha) == pytest.approx(exact, rel=1e-9) if d else poly(alpha) == pytest.approx(f(alpha))
            )
        assert abs(poly(alpha + h * np.sign(-alpha)) - f(alpha - h * np.sign(alpha))) < 1e-3


def test_gradient_matches_finite_differences():
    t = np_templates("BelleII_public", Model("phi", 2.1, CSL=1e-8), Model("psipsi", 1.0, Lambda=1e4, CVL=1.0))
    rng = np.random.default_rng(1)
    x = np.r_[1.3, 0.7, -0.4, 1.3 * rng.normal(size=NPAR_BelleII_public)]
    f = lambda y: logLL_BelleII_public(y, *t)
    h, e = 1e-5, np.eye(len(x))
    gn = np.array([(f(x + h * e[i]) - f(x - h * e[i])) / (2 * h) for i in range(len(x))])
    assert np.max(np.abs(grad_BelleII_public(x, *t) - gn) / (1 + np.abs(gn))) < 1e-4


def test_SM_fit_reproduces_belle_II(pyhf_model):
    # pyhf fit of Belle II's ITA likelihood (its own SM signal and form factor uncertainties):
    # mu = 5.42 +- 1.45 and -2 Delta ln L(mu = 0) = 12.50. Ours uses our SM shape and form factors.
    fit = min_logLL("BelleII_public")
    assert fit.value == pytest.approx(5.42, abs=0.06)
    assert fit.error == pytest.approx(1.45, abs=0.03)
    assert profile_logLL_SM("BelleII_public", 0.0) == pytest.approx(12.50, abs=0.1)


def test_reweighted_SM_matches_belle_II_signal():
    # Our SM spectrum folded through the joint number density is Belle II's signal sample up to
    # the differences between the SM predictions (B = 4.904e-6 here, 4.966e-6 for Belle II)
    sig = HF.signal_nominal > 0
    assert s_SM_BelleII_public[sig] == pytest.approx(HF.signal_nominal[sig], rel=0.015)
    assert np.all(s_SM_BelleII_public[~sig] == 0)
    assert binned_B_BelleII_public(include_SM=True).sum() == pytest.approx(
        quad(lambda q: dBdqsq_SM(q)[0], 1e-12, 22.885)[0], rel=1e-6
    )
    assert BelleII_public.null_binned.sum() == pytest.approx(quad(belle2_null_spectrum, 0, 22.885)[0], rel=1e-6)


def test_two_body_signal_is_interpolated_between_true_q2_bins():
    edges = BelleII_public.q2gen_bins[1:]
    centres = (edges[1:] + edges[:-1]) / 2
    model = Model("phi", np.sqrt(0.25 * centres[18] + 0.75 * centres[19]), CSL=1e-8)
    B = binned_B_BelleII_public(model)
    assert np.flatnonzero(B).tolist() == [18, 19]
    assert B[18:20] == pytest.approx([0.25 * model.B()[0], 0.75 * model.B()[0]])
    # The prediction is continuous in the mass: no jumps at the bin edges
    q2 = np.linspace(4.2, 4.8, 61)
    N = np.array(
        [
            Nrec_BelleII_public(Model("phi", np.sqrt(q), CSL=1e-8)) / Model("phi", np.sqrt(q), CSL=1e-8).B()[0]
            for q in q2
        ]
    )
    assert np.max(np.abs(np.diff(N, axis=0))) < 0.02 * np.max(np.abs(N))  # 0.15 without interpolation
    assert binned_B_BelleII_public(Model("phi", 4.79, CSL=1e-8)).sum() == 0  # beyond Belle II's endpoint
