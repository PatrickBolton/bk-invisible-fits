"""Likelihoods, gradients and fits."""

import pickle

import numpy as np
import pytest
from joblib import Parallel, delayed

from lights.likelihoods import (
    Nevents_BelleII,
    NPAR_BaBar,
    NPAR_BelleII,
    _fit,
    _kind_of,
    _x_init,
    components_BaBar,
    components_BelleII,
    grad_BaBar,
    grad_BelleII,
    grad_Combined,
    grad_LEP,
    logLL_BaBar,
    logLL_BelleII,
    logLL_Combined,
    logLL_LEP,
    min_logLL,
    np_templates,
    profile_logLL_scan,
    profile_logLL_SM,
)
from lights.load_data import BelleII
from lights.models import Model
from lights.statistics import reduced_chisq

M1 = Model("phi", 1.0, CSL=1e-8, CSR=3e-9)
M2 = Model("phiphi", 1.0, Lambda=1e4, CSL=1e-3)
T_BELLE = np_templates("BelleII", M1, M2)
T_BABAR = np_templates("BaBar", M1, M2)
GS = Model("phi", 2.1, CSL=1e-8, CSR=1e-8)


def finite_difference(f, x, h=1e-6):
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
        f, g, npar = (
            (lambda y: logLL_Combined(y, *T_BELLE, *T_BABAR)),
            (lambda y: grad_Combined(y, *T_BELLE, *T_BABAR)),
            NPAR_BelleII + NPAR_BaBar,
        )
    else:
        f, g, npar = ((lambda y: logLL_LEP(y, 3e-5, 1e-4)), (lambda y: grad_LEP(y, 3e-5, 1e-4)), 2)
    x = np.concatenate(([1.3, 0.7, -0.4], 0.05 * rng.normal(size=npar)))
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
    fit = min_logLL(logLL_BelleII)
    assert fit.minuit.valid
    assert fit.value == pytest.approx(5.086, abs=0.01)
    assert fit.error == pytest.approx(1.587, abs=0.01)
    assert -fit.delta_chi2 == pytest.approx(6.693, abs=0.01)
    _, _, m, _ = fit  # also unpacks as a tuple
    assert reduced_chisq(m, BelleII.obs) == pytest.approx(1.981, abs=0.01)
    assert profile_logLL_SM(logLL_BelleII, 0.0) == pytest.approx(10.37, abs=0.02)
    profile = profile_logLL_SM(logLL_BelleII, [0.0, fit.value])  # an array of mu in one call
    assert profile.shape == (2,)
    assert profile == pytest.approx([10.37, 0.0], abs=0.02)


def test_NP_fit_regression():
    fit = min_logLL(logLL_BelleII, GS)
    assert abs(fit.value) == pytest.approx(1.571, abs=0.003)
    assert fit.delta_chi2 == pytest.approx(-20.05, abs=0.02)


def test_NP_fit_in_parallel_workers():
    fit = lambda model: min_logLL(logLL_BelleII, model).delta_chi2
    res = Parallel(n_jobs=2)(delayed(fit)(GS.with_mass(mX)) for mX in (1.0, 2.1))
    assert res[1] == pytest.approx(-20.05, abs=0.02)


def test_vanishing_template_is_still_fitted():
    # Templates ~ m_X^2 need large couplings; normalisation keeps them within reach of Migrad
    model = Model("PsiPsi", 1e-4, Lambda=1e2, CTPL=1.0, CTPR=-1.0)
    d_small = min_logLL(logLL_BelleII, model).delta_chi2
    d_ref = min_logLL(logLL_BelleII, model.with_mass(1e-2)).delta_chi2
    assert d_small < 0
    assert d_small == pytest.approx(d_ref, abs=0.01)


def test_nuisances_cannot_cancel_a_large_signal():
    tpl = np_templates("BaBar", Model("phi", 2.1, CSL=5e-9, CSR=5e-9), Model("phi", 2.1, CSL=5e-9, CSR=-5e-9))
    m = _fit("BaBar", tpl, _x_init("BaBar", (1.0, 8.0, 30.0)), ["x0", "x1", "x2"])
    nominal = 30**2 * tpl[1]
    fitted = components_BaBar(m.values, *tpl)["NP_2"]
    k = np.argmax(nominal)
    assert fitted[k] > 0.3 * nominal[k]


def test_scan_minimum_is_close_to_zero():
    gS = Model("phi", 2.1, CSL=5e-9, CSR=5e-9)
    gP = Model("phi", 2.1, CSL=5e-9, CSR=-5e-9)
    res = profile_logLL_scan(gS, gP, [-8, 8], [-30, 30], pts=[15, 15], n_jobs=2)
    assert np.array_equal(res.combined, res[3])
    for name, a in zip(res._fields, res):
        assert np.isfinite(a).all(), name
        assert -0.05 < a.min() < 1.0, name


def test_reduced_chisq_counts_free_strengths():
    m_SM = min_logLL(logLL_BelleII).minuit
    m_NP = min_logLL(logLL_BelleII, GS).minuit
    assert reduced_chisq(m_SM, BelleII.obs) == pytest.approx(reduced_chisq(m_SM, BelleII.obs, 1))
    assert reduced_chisq(m_NP, BelleII.obs) == pytest.approx(reduced_chisq(m_NP, BelleII.obs, 1))  # x0 fixed, x1 free
    m_free = _fit("BelleII", np_templates("BelleII", GS), _x_init("BelleII"), ["x2"])
    assert reduced_chisq(m_free, BelleII.obs) == pytest.approx(reduced_chisq(m_free, BelleII.obs, 2))


def test_experiment_names_and_functions_give_the_same_fit():
    assert min_logLL("BelleII", GS).delta_chi2 == pytest.approx(min_logLL(logLL_BelleII, GS).delta_chi2, abs=1e-6)
    with pytest.raises(ValueError, match="Unknown experiment"):
        min_logLL("Belle", GS)


def test_fit_leaves_the_zero_coupling_stationary_point():
    # x = 0 is stationary (the strength is x^2); BaBar prefers this model slightly, which a fit
    # started at x = 0 would not find
    psi = Model("psipsi", 0.6, Lambda=1e4, CVL=1.0, CVR=1.0)
    assert min_logLL("BaBar", psi).delta_chi2 < -1e-3


def test_NP_fit_is_never_worse_than_the_SM():
    # The SM is the x = 0 point of every NP model, so Delta chi^2 <= 0 up to the fit tolerance
    for model in (GS, Model("V", 2.0, CVL=1e-8)):
        assert min_logLL("BaBar", model).delta_chi2 < 5e-5


def test_scan_of_selected_experiments():
    psi = Model("psipsi", 0.6, Lambda=1e4, CVL=1.0, CVR=1.0)
    res = profile_logLL_scan(psi, None, (-1, 1), (0, 0), pts=(3, 1), experiments=("BaBar", "LEP"), n_jobs=1)
    assert res.BelleII is None and res.combined is None
    assert res.BaBar.shape == (1, 3)
    # LEP: relative to its own minimum. Vector currents give no Bs -> psi psi-bar, so LEP is flat at 0
    assert psi.B_Bs() == 0 and np.all(res.LEP == 0)
    pseudo = Model("psipsi", 0.6, Lambda=1e4, CSL=-1.0, CSR=1.0)  # pseudoscalar current: B_Bs > 0
    res = profile_logLL_scan(pseudo, None, (0, 4), (0, 0), pts=(41, 1), experiments=("LEP",), n_jobs=1)
    assert 0 <= res.LEP.min() < 0.01 and res.LEP.max() > 1
    with pytest.raises(ValueError, match="Unknown experiment"):
        profile_logLL_scan(psi, None, (-1, 1), (0, 0), experiments=("Belle",))
