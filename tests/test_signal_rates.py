"""Binned signal predictions and their theory uncertainties."""

import numpy as np
import pytest
from scipy.integrate import simpson

from lights.constants import mB
from lights.load_data import BaBar, BelleII
from lights.models import Model
from lights.signal_rates import (
    Nrec_BaBar,
    Nrec_BelleII,
    dNdqsq_cov,
    dNdqsqrec_BaBar,
    dNdqsqrec_BelleII,
    efficiency,
    signal_template,
)

# (model, include_SM)
CASES = [
    (None, True),  # SM
    (Model("phi", 2.1, CSL=1e-8, CSR=1e-8), False),  # 2-body
    (Model("psipsi", 1.5, Lambda=1e4, CVL=1.0, CVR=1.0), False),
]  # 3-body, threshold inside a bin


def dense_BelleII(model, include_SM, n=201):
    "Smear, then integrate each reconstructed bin with n points."
    out = []
    for lo, hi in zip(BelleII.bins[:-1], BelleII.bins[1:]):
        q = np.linspace(lo, hi, n)
        out.append(simpson(dNdqsqrec_BelleII(q, model, include_SM=include_SM), x=q))
    return np.array(out)


def dense_BaBar(model, include_SM, n=2001):
    out = []
    for lo, hi in zip(BaBar.bins[:-1], BaBar.bins[1:]):
        q = np.linspace(max(lo, 1e-5 * mB**2), hi, n)
        out.append(simpson(dNdqsqrec_BaBar(q, model, include_SM=include_SM), x=q))
    return np.array(out)


@pytest.mark.parametrize("case", CASES)
def test_Nrec_BelleII_matches_dense_integration(case):
    model, include_SM = case
    new, ref = Nrec_BelleII(model, include_SM=include_SM), dense_BelleII(model, include_SM)
    big = ref > 1e-3 * ref.max()
    assert new[big] == pytest.approx(ref[big], rel=2e-3)
    assert new.sum() == pytest.approx(ref.sum(), rel=1e-3)


@pytest.mark.parametrize("case", [CASES[0], CASES[2]])
def test_Nrec_BaBar_matches_dense_integration(case):
    model, include_SM = case
    new, ref = Nrec_BaBar(model, include_SM=include_SM), dense_BaBar(model, include_SM)
    big = ref > 1e-3 * ref.max()
    assert new[big] == pytest.approx(ref[big], rel=3e-3)


def test_two_body_signal_lands_in_its_bin():
    s = Nrec_BaBar(Model("phi", 2.1, CSL=-1e-8, CSR=1e-8))
    k = np.argmin(np.abs(BaBar.bin_cents - 2.1**2))
    assert s[k] > 0 and np.all(np.delete(s, k) == 0)


def test_massless_three_body_signal_is_finite():
    # The true-q^2 grid starts at the pair threshold 4 mX^2, which must not be q^2 = 0
    massless = Nrec_BelleII(Model("psipsi", 0.0, Lambda=1e4, CTL=1.0, CTR=-1.0))
    light = Nrec_BelleII(Model("psipsi", 1e-10, Lambda=1e4, CTL=1.0, CTR=-1.0))
    assert np.all(np.isfinite(massless)) and massless == pytest.approx(light, rel=1e-6)


def test_efficiency_is_never_negative():
    eff = efficiency(np.arange(1.0, 24.0, 2.0), np.r_[np.linspace(0.1, 0.02, 10), 0.0, 0.0], 3)
    assert np.all(eff(np.linspace(0.0, 25.0, 500)) >= 0)


def test_templates_are_non_negative():
    s, _ = signal_template("BelleII", Model("psipsi", 2.3, Lambda=1e4, CVL=1.0))
    assert np.all(s >= 0)


@pytest.mark.parametrize("experiment", ["BelleII", "BaBar"])
def test_linear_response_reproduces_toy_covariance(experiment):
    # s + J u with u ~ N(0, 1) implies cov = J J^T; compare with full nonlinear toys
    s0, J = signal_template(experiment, include_SM=True)
    mean, C = dNdqsq_cov(experiment, 2000, include_SM=True, rng=1)
    # Bins below 1% of the peak are excluded: at the kinematic endpoint the efficiency is close to
    # zero, so the toys there are distorted by clipping in the same way (Belle II's last bin, 0.1 events).
    live = s0 > 1e-2 * s0.max()
    assert mean[live] == pytest.approx(s0[live], rel=0.02)
    assert np.sqrt(np.diag(J @ J.T))[live] == pytest.approx(np.sqrt(np.diag(C))[live], rel=0.05)
