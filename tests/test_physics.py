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


def wet_eq9_K(q2, CV, CS, CT):
    "B -> K nu nu-bar in the WET, eq. 9 of arXiv:2507.12393 (EOS b-to-psd-nu-nu.cc), alpha_e(m_b) normalisation."
    from lights.constants import ALPHA_MB, GF, LAMBDAt, mb, ms, tauB
    from lights.kinematics import lam

    L = lam(mB**2, mK**2, q2)
    pre = (
        3
        * tauB
        * (4 * GF / np.sqrt(2) * ALPHA_MB / (2 * np.pi) * LAMBDAt) ** 2
        * np.sqrt(L)
        * q2
        / ((4 * np.pi) ** 3 * mB**3)
    )
    return pre * (
        L / (24 * q2) * ff.fpf(q2) ** 2 * CV**2
        + (mB**2 - mK**2) ** 2 / (8 * (mb - ms) ** 2) * ff.f0f(q2) ** 2 * CS**2
        + 2 * L / (3 * (mB + mK) ** 2) * ff.fTf(q2) ** 2 * CT**2
    )


def wet_eos_Kstar(q2, CVL=0.0, CVR=0.0, CSL=0.0, CSR=0.0, CTL=0.0):
    "B -> K* nu nu-bar in the WET: EOS b-to-vec-nu-nu.cc (total and longitudinal width) times tau_B0."
    from lights.constants import ALPHA_MB, GF, LAMBDAt, mb, ms, tauB0
    from lights.kinematics import lam

    L = lam(mB**2, mKs**2, q2)
    V, A1, A12, A0 = ff.Vf(q2), ff.A1f(q2), ff.A12f(q2), ff.A0f(q2)
    T1, T2, T23 = ff.T1f(q2), ff.T2f(q2), ff.T23f(q2)
    norm = 3 * tauB0 * (4 * GF * ALPHA_MB / (2 * np.pi)) ** 2 / 2 * LAMBDAt**2 * np.sqrt(L) / (4 * np.pi * mB) ** 3
    longitudinal = 8 * mB**2 * mKs**2 / 3 * A12**2 * (CVL - CVR) ** 2
    total = (
        q2 * L / (12 * (mB + mKs) ** 2) * V**2 * (CVL + CVR) ** 2
        + longitudinal
        + q2 * (mB + mKs) ** 2 / 12 * A1**2 * (CVL - CVR) ** 2
        + q2 * L / (8 * (mb + ms) ** 2) * A0**2 * (CSR - CSL) ** 2
        + q2
        * (
            32 * mB**2 * mKs**2 / (3 * (mB + mKs) ** 2) * T23**2
            + (4 * L * T1**2 + 4 * (mB**2 - mKs**2) ** 2 * T2**2) / (3 * q2)
        )
        * CTL**2
    )
    return norm * total, norm * longitudinal


WET_CASES = [  # (CVL, CVR, CSL, CSR, CTL)
    (2.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 2.0, 0.0, 0.0, 0.0),
    (2.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 2.0, 0.0, 0.0),
    (0.0, 0.0, 2.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 2.0),
    (1.0, -0.5, 2.0, 1.0, 1.5),
]


@pytest.mark.parametrize("C", WET_CASES)
def test_WET_rates_match_belle_II_and_EOS(C):
    CVL, CVR, CSL, CSR, CTL = C
    q = np.linspace(0.5, 18.0, 8)
    rates = Model("nunu", 0.0, CVL=CVL, CVR=CVR, CSL=CSL, CSR=CSR, CTL=CTL).dBdqsq(q)
    assert rates[0] == pytest.approx(wet_eq9_K(q, CVL + CVR, CSL + CSR, CTL), rel=1e-10)
    total, longitudinal = wet_eos_Kstar(q, CVL, CVR, CSL, CSR, CTL)
    assert rates[1] == pytest.approx(total, rel=1e-10)
    if CSL == CSR == CTL == 0:  # EOS's longitudinal width has the vector currents only
        assert rates[2] == pytest.approx(longitudinal, rel=1e-10)


def test_WET_vector_coupling_is_the_SM():
    # The WET rate is the SM rate for CVL = Xt / sW^2 in the normalisation of dBdqsq_SM
    # (alpha = 1/127.952), i.e. 6.60 in the WET normalisation with alpha_e(m_b) = 1/133 (EOS: 6.61)
    from lights.constants import ALPHA, ALPHA_MB, SWsq, Xt

    q = np.linspace(0.5, 22.0, 9)
    CV_SM = Xt / SWsq * ALPHA / ALPHA_MB
    assert CV_SM == pytest.approx(6.613, abs=0.01)
    assert Model("nunu", 0.0, CVL=CV_SM).dBdqsq(q) == pytest.approx(dBdqsq_SM(q), rel=1e-12)
    # ... also with other form factors: they are passed through
    a = ff.a_vals_gen(1, rng=2)[0]
    assert Model("nunu", 0.0, CVL=CV_SM).dBdqsq(q, a) == pytest.approx(dBdqsq_SM(q, a_vals=a), rel=1e-12)
