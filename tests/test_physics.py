"""Branching fractions and form factors."""

import numpy as np
import pytest
from scipy.integrate import quad

from lights import form_factors as ff
from lights.branching_fractions import B_Bsphiphi, dBdqsq_BKPsiPsi, dBdqsq_SM
from lights.constants import mB, mK, mKs
from lights.models import Model


def test_SM_branching_fractions():
    BK = quad(lambda q: dBdqsq_SM(q)[0], 1e-8, (mB - mK) ** 2)[0]
    BKs = quad(lambda q: dBdqsq_SM(q)[1], 1e-8, (mB - mKs) ** 2)[0]
    assert BK == pytest.approx(4.904e-6, rel=1e-3)
    assert BKs == pytest.approx(8.917e-6, rel=1e-3)


@pytest.mark.parametrize("model", [Model("VV", 0.3, Lambda=1e2, CSL=1.0), Model("PsiPsi", 0.3, Lambda=1e2, CVL=1.0)])
def test_no_K_rate_leaks_into_Kstar_rows(model):
    # dBdqsq_BKVV and dBdqsq_BKPsiPsi only implement B -> K; their K* rows must be zero
    rates = model.dBdqsq(np.array([2.0, 5.0]))
    assert rates.shape == (4, 2)
    assert np.all(rates[0] > 0)
    assert np.all(rates[1:] == 0)


def test_Bs_to_scalar_pair_vector_currents_vanish():
    # p.(p1 - p2) = m1^2 - m2^2 = 0 for phi phi*: only the pseudoscalar coupling contributes
    assert B_Bsphiphi(0.5, [0, 0, 1.0, 1.0], 1e3) == 0
    assert B_Bsphiphi(0.5, [0, 0, -1.0, 1.0], 1e3) == 0
    ps = B_Bsphiphi(0.5, [-1.0, 1.0, 0, 0], 1e3)
    assert ps > 0
    assert B_Bsphiphi(0.5, [-1.0, 1.0, 0, 0], 1e3, True) == pytest.approx(2 * ps)


def test_small_mass_limit_has_no_cancellation():
    # Rates proportional to 1 - beta^2 = 4 m^2/q^2 must scale smoothly as m -> 0
    C = np.zeros(14)
    C[12], C[13] = 1.0, -1.0
    ratios = [dBdqsq_BKPsiPsi(5.0, m, C, 1e2)[0] / m**2 for m in (1e-10, 1e-4, 1e-2)]
    assert ratios[0] > 0
    assert ratios == pytest.approx([ratios[0]] * 3, rel=1e-3)


def test_form_factor_relations():
    assert ff.f0f(0.0) == pytest.approx(ff.fpf(0.0))
    assert ff.T2f(0.0) == pytest.approx(ff.T1f(0.0))
    assert ff.A12f(0.0) == pytest.approx((mB**2 - mKs**2) / (8 * mB * mKs) * ff.A0f(0.0))


def test_form_factors_vectorised():
    q = np.linspace(0.0, 20.0, 5)
    for f in (ff.fpf, ff.f0f, ff.fTf, ff.A0f, ff.A1f, ff.A12f, ff.Vf, ff.T1f, ff.T2f, ff.T23f):
        assert f(q) == pytest.approx([f(x) for x in q])


def test_random_toys_reproducible():
    a = ff.a_vals_gen(5, rng=3)
    b = ff.a_vals_gen(5, rng=3)
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def test_WET_vector_coupling_is_the_SM():
    # dB/dq^2 of B -> K nu nu-bar in the WET (arXiv:2507.12393 eq. 9) is the SM rate for
    # C_V = Xt / sW^2 in the normalisation of dBdqsq_SM (alpha = 1/127.952), i.e. 6.60 in the WET
    # normalisation with alpha_e(m_b) = 1/133 (EOS: 6.61)
    from lights.constants import ALPHA, ALPHA_MB, SWsq, Xt

    q = np.linspace(0.5, 22.0, 9)
    CV_SM = Xt / SWsq * ALPHA / ALPHA_MB
    assert CV_SM == pytest.approx(6.613, abs=0.01)
    wet = Model("nunu", 0.0, CV=CV_SM).dBdqsq(q)
    assert wet[0] == pytest.approx(dBdqsq_SM(q)[0], rel=1e-12)
    assert np.all(wet[1:] == 0)  # K only
    # the three couplings add incoherently
    both = Model("nunu", 0.0, CS=2.0, CT=3.0).dBdqsq(q)[0]
    assert both == pytest.approx(Model("nunu", 0.0, CS=2.0).dBdqsq(q)[0] + Model("nunu", 0.0, CT=3.0).dBdqsq(q)[0])
