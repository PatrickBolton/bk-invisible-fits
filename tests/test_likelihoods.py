"""Likelihoods, gradients and fits."""

import pickle

import numpy as np
import pytest
from joblib import Parallel, delayed

from lights.load_data import obs
from lights.branching_fractions import cdict
from lights.statistics import reduced_chisq
from lights.likelihoods import (logLL_BelleII, logLL_BaBar, logLL_Combined, logLL_LEP,
                                grad_BelleII, grad_BaBar, grad_Combined, grad_LEP,
                                Nevents_BelleII, components_BelleII, components_BaBar,
                                np_templates, min_logLL, profile_logLL_SM, profile_logLL_scan,
                                NPAR_BelleII, NPAR_BaBar, _kind_of, _fit, _x_init)

ZERO = np.zeros((6, 14))


def couplings(**kw):
    c = np.zeros((6, 14))
    for name, value in kw.items():
        c[cdict[name]] = value
    return c


C1 = couplings(CSL_s = 1e-8, CSR_s = 3e-9)
C2 = couplings(CSL_ss = 1e-3)
T_BELLE = np_templates("BelleII", 1.0, C1, C2, 1e4)
T_BABAR = np_templates("BaBar", 1.0, C1, C2, 1e4)


def finite_difference(f, x, h = 1e-6):
    e = np.eye(len(x))
    return np.array([(f(x + h * e[i]) - f(x - h * e[i])) / (2 * h) for i in range(len(x))])


@pytest.mark.parametrize("kind", ["BelleII", "BaBar", "Combined", "LEP"])
def test_gradients_match_finite_differences(kind):
    rng = np.random.default_rng(1)
    if kind == "BelleII":
        f, g, npar = (lambda y: logLL_BelleII(y, *T_BELLE)), (lambda y: grad_BelleII(y, *T_BELLE)), NPAR_BelleII
    elif kind == "BaBar":
        f, g, npar = (lambda y: logLL_BaBar(y, *T_BABAR)), (lambda y: grad_BaBar(y, *T_BABAR)), NPAR_BaBar
    elif kind == "Combined":
        f, g, npar = ((lambda y: logLL_Combined(y, *T_BELLE, *T_BABAR)),
                      (lambda y: grad_Combined(y, *T_BELLE, *T_BABAR)), NPAR_BelleII + NPAR_BaBar)
    else:
        f, g, npar = ((lambda y: logLL_LEP(np.array([y[1], y[2], 3e-5, 1e-4]))),
                      (lambda y: grad_LEP(y, 3e-5, 1e-4)), 2)
    x = np.concatenate(([1.3, 0.7, -0.4], 0.05 * rng.normal(size = npar)))
    gn = finite_difference(f, x)
    assert np.max(np.abs(g(x) - gn) / (1 + np.abs(gn))) < 1e-4


def test_components_sum_to_expected_events():
    x = _x_init("BelleII", (1.2, 0.5, 0.3))
    x[3:] = 0.1
    x[-3:] = 0.01
    assert sum(components_BelleII(x, *T_BELLE).values()) == pytest.approx(Nevents_BelleII(x, *T_BELLE))


def test_kind_lookup_survives_pickling():
    # joblib workers receive copies of the likelihood functions
    assert _kind_of(pickle.loads(pickle.dumps(logLL_BelleII))) == "BelleII"
    assert _kind_of("BaBar") == "BaBar"


def test_SM_fit_regression():
    mu, err, m, dchi2 = min_logLL(logLL_BelleII)
    assert m.valid
    assert mu == pytest.approx(5.086, abs = 0.01)
    assert err == pytest.approx(1.587, abs = 0.01)
    assert -dchi2 == pytest.approx(6.693, abs = 0.01)
    assert reduced_chisq(m, obs) == pytest.approx(1.981, abs = 0.01)
    assert profile_logLL_SM(logLL_BelleII, 0.) == pytest.approx(10.37, abs = 0.02)


def test_NP_fit_regression():
    c = couplings(CSL_s = 1e-8, CSR_s = 1e-8)
    x1, err, m, dchi2 = min_logLL(logLL_BelleII, None, None, 2.1, c, ZERO, 1., NP = 1)
    assert abs(x1) == pytest.approx(1.571, abs = 0.003)
    assert dchi2 == pytest.approx(-20.05, abs = 0.02)


def test_NP_fit_in_parallel_workers():
    c = couplings(CSL_s = 1e-8, CSR_s = 1e-8)
    fit = lambda mX: min_logLL(logLL_BelleII, None, None, mX, c, ZERO, 1., NP = 1)[3]
    res = Parallel(n_jobs = 2)(delayed(fit)(mX) for mX in (1.0, 2.1))
    assert res[1] == pytest.approx(-20.05, abs = 0.02)


def test_vanishing_template_is_still_fitted():
    # Templates ~ m_X^2 need large couplings; normalisation keeps them within reach of Migrad
    c = couplings(CTPL_FF = 1., CTPR_FF = -1.)
    d_small = min_logLL(logLL_BelleII, None, None, 1e-4, c, ZERO, 1e2, NP = 1)[3]
    d_ref = min_logLL(logLL_BelleII, None, None, 1e-2, c, ZERO, 1e2, NP = 1)[3]
    assert d_small < 0
    assert d_small == pytest.approx(d_ref, abs = 0.01)


def test_nuisances_cannot_cancel_a_large_signal():
    tpl = np_templates("BaBar", 2.1, couplings(CSL_s = 5e-9, CSR_s = 5e-9), couplings(CSL_s = 5e-9, CSR_s = -5e-9), 1.)
    m = _fit("BaBar", tpl, _x_init("BaBar", (1., 8., 30.)), ["x0", "x1", "x2"])
    nominal = 30**2 * tpl[1]
    fitted = components_BaBar(m.values, *tpl)["NP_2"]
    k = np.argmax(nominal)
    assert fitted[k] > 0.3 * nominal[k]


def test_scan_minimum_is_close_to_zero():
    c1 = couplings(CSL_s = 5e-9, CSR_s = 5e-9)
    c2 = couplings(CSL_s = 5e-9, CSR_s = -5e-9)
    res = profile_logLL_scan(2.1, c1, c2, [-8, 8], [-30, 30], pts = [15, 15], Lambda = 1., n_jobs = 2)
    for name, a in zip(["BelleII", "BaBar", "LEP", "Combined"], res):
        assert np.isfinite(a).all(), name
        assert -0.05 < a.min() < 1.0, name
