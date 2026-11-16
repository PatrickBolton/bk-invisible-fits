"""Fits, profile likelihoods and scans."""

import numpy as np
import pytest
from joblib import Parallel, delayed

from lights.fits import _fit, _x_init, mass_scan, min_logLL, profile_logLL_scan, profile_logLL_SM
from lights.likelihoods import components_BaBar, np_templates
from lights.load_data import BelleII
from lights.models import Model
from lights.statistics import reduced_chisq

GS = Model("phi", 2.1, CSL=1e-8, CSR=1e-8)


def test_SM_fit_regression():
    fit = min_logLL("BelleII")
    assert fit.minuit.valid
    assert fit.value == pytest.approx(5.086, abs=0.01)
    assert fit.error == pytest.approx(1.587, abs=0.01)
    assert -fit.delta_chi2 == pytest.approx(6.693, abs=0.01)
    _, _, m, _ = fit  # also unpacks as a tuple
    assert reduced_chisq(m, BelleII.obs) == pytest.approx(1.981, abs=0.01)
    assert profile_logLL_SM("BelleII", 0.0) == pytest.approx(10.37, abs=0.02)
    profile = profile_logLL_SM("BelleII", [0.0, fit.value])  # an array of mu in one call
    assert profile.shape == (2,)
    assert profile == pytest.approx([10.37, 0.0], abs=0.02)


def test_NP_fit_regression():
    fit = min_logLL("BelleII", GS)
    assert abs(fit.value) == pytest.approx(1.571, abs=0.003)
    assert fit.delta_chi2 == pytest.approx(-20.05, abs=0.02)


def test_NP_fit_in_parallel_workers():
    fit = lambda model: min_logLL("BelleII", model).delta_chi2
    res = Parallel(n_jobs=2)(delayed(fit)(GS.with_mass(mX)) for mX in (1.0, 2.1))
    assert res[1] == pytest.approx(-20.05, abs=0.02)


def test_vanishing_template_is_still_fitted():
    # Templates ~ m_X^2 need large couplings; normalisation keeps them within reach of Migrad
    model = Model("PsiPsi", 1e-4, Lambda=1e2, CTPL=1.0, CTPR=-1.0)
    d_small = min_logLL("BelleII", model).delta_chi2
    d_ref = min_logLL("BelleII", model.with_mass(1e-2)).delta_chi2
    assert d_small < 0
    assert d_small == pytest.approx(d_ref, abs=0.01)


def test_nuisances_cannot_cancel_a_large_signal():
    tpl = np_templates("BaBar", Model("phi", 2.1, CSL=5e-9, CSR=5e-9), Model("phi", 2.1, CSL=5e-9, CSR=-5e-9))
    m = _fit("BaBar", tpl, _x_init("BaBar", (1.0, 8.0, 30.0)), ["x0", "x1", "x2"])
    nominal = 30**2 * tpl.s_NP_2
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
    m_SM = min_logLL("BelleII").minuit
    m_NP = min_logLL("BelleII", GS).minuit
    assert reduced_chisq(m_SM, BelleII.obs) == pytest.approx(reduced_chisq(m_SM, BelleII.obs, 1))
    assert reduced_chisq(m_NP, BelleII.obs) == pytest.approx(reduced_chisq(m_NP, BelleII.obs, 1))  # x0 fixed, x1 free
    m_free = _fit("BelleII", np_templates("BelleII", GS), _x_init("BelleII"), ["x2"])
    assert reduced_chisq(m_free, BelleII.obs) == pytest.approx(reduced_chisq(m_free, BelleII.obs, 2))


def test_unknown_experiment_is_an_error():
    for fit in (lambda e: min_logLL(e, GS), lambda e: profile_logLL_SM(e, 1.0), lambda e: mass_scan(e, GS, [1.0])):
        with pytest.raises(ValueError, match="Unknown experiment"):
            fit("Belle")


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


def test_mass_scan_matches_single_fits():
    masses = [1.0, 2.1]
    scan = mass_scan("BelleII", GS, masses, n_jobs=2)
    assert scan == pytest.approx([min_logLL("BelleII", GS.with_mass(m)).delta_chi2 for m in masses], abs=1e-6)


# Combined fits (Belle II + BaBar + LEP) and the choice of Belle II likelihood
VV = Model("VV", 0.3, Lambda=3e2, CSL=1.0, CSR=1.0)  # B -> K only and no Bs -> invisible: BaBar and LEP are blind
PSEUDO = Model("psipsi", 0.6, Lambda=1e4, CSL=-1.0, CSR=1.0)  # pseudoscalar current: Bs -> psi psi-bar > 0


@pytest.mark.parametrize("belle2", ["BelleII", "BelleII_public"])
def test_combined_fit_reduces_to_belle_II_when_BaBar_and_LEP_are_blind(belle2):
    combined = "combined" if belle2 == "BelleII" else "combined_public"
    assert min_logLL(combined, VV).delta_chi2 == pytest.approx(min_logLL(belle2, VV).delta_chi2, abs=2e-3)


@pytest.mark.parametrize("experiment", ["combined", "combined_public"])
def test_combined_fit_normalises_the_LEP_term_with_the_couplings(experiment):
    # The fitted minimum, recomputed from the raw (unnormalised) templates and LEP term at the
    # fitted coupling, must be the same: x_true^2 Br = x_fit^2 (Br / scale)
    from lights.fits import _lep_term, _templates

    assert PSEUDO.B_Bs() > 0
    fit = min_logLL(experiment, PSEUDO)
    raw = _fit(
        experiment,
        _templates(experiment, PSEUDO),
        _x_init(experiment, (1.0, fit.value, 0.0)),
        ["x0", "x1", "x2"],
        extra=_lep_term((PSEUDO.B_Bs(), 0.0)),
    )
    assert raw.fval == pytest.approx(fit.minuit.fval, abs=2e-3)


@pytest.mark.parametrize("belle2", ["lights", "public"])
def test_scan_with_either_belle_II_likelihood(belle2):
    # At the SM point (x = 0) the combined grid is -Delta chi^2 of the combined fit
    res = profile_logLL_scan(PSEUDO, None, (-2, 2), (0, 0), pts=(5, 1), belle2=belle2, n_jobs=1)
    assert res.belle2 == belle2 and res.x1[2] == 0
    for name, grid in zip(res._fields, res):
        assert grid.min() > -0.05, name
    combined = "combined" if belle2 == "lights" else "combined_public"
    assert res.combined[0, 2] == pytest.approx(-min_logLL(combined, PSEUDO).delta_chi2, abs=0.02)
    with pytest.raises(ValueError, match="belle2"):
        profile_logLL_scan(PSEUDO, None, (-1, 1), (0, 0), belle2="Belle")


def test_mass_scan_of_the_combined_fit():
    scan = mass_scan("combined_public", PSEUDO, [0.3, 0.6], n_jobs=1)
    assert scan[1] == pytest.approx(min_logLL("combined_public", PSEUDO).delta_chi2, abs=1e-3)
