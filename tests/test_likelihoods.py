"""Likelihoods, gradients and NP templates."""

import numpy as np
import pytest

from lights.likelihoods import (
    Nevents_BelleII,
    NPAR_BaBar,
    NPAR_BelleII,
    Templates,
    components_BelleII,
    grad_BaBar,
    grad_BelleII,
    grad_Combined,
    grad_LEP,
    logLL_BaBar,
    logLL_BelleII,
    logLL_Combined,
    logLL_LEP,
    np_templates,
)
from lights.models import Model

M1 = Model("phi", 1.0, CSL=1e-8, CSR=3e-9)
M2 = Model("phiphi", 1.0, Lambda=1e4, CSL=1e-3)
T_BELLE = np_templates("BelleII", M1, M2)
T_BABAR = np_templates("BaBar", M1, M2)


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
    x = np.r_[1.2, 0.5, 0.3, np.full(NPAR_BelleII, 0.1)]
    x[-3:] = 0.01
    assert sum(components_BelleII(x, *T_BELLE).values()) == pytest.approx(Nevents_BelleII(x, *T_BELLE))


def test_templates_unpack_into_the_likelihood():
    assert isinstance(T_BELLE, Templates) and T_BELLE.s_NP_1.shape == (20,)
    empty = np_templates("BaBar")  # no models: zero templates of the right shape
    assert not np.any(empty.s_NP_1) and empty.R_NP_2.shape == T_BABAR.R_NP_2.shape
    x = np.r_[1.0, 0.5, 0.3, np.zeros(NPAR_BelleII)]
    assert logLL_BelleII(x, *T_BELLE) == logLL_BelleII(
        x, T_BELLE.s_NP_1, T_BELLE.s_NP_2, T_BELLE.R_NP_1, T_BELLE.R_NP_2
    )
