"""The Model class and the channel registry."""

import pickle

import numpy as np
import pytest

from lights.fits import min_logLL
from lights.likelihoods import logLL_BelleII
from lights.models import B_Bs, B_tot, Model, dBdqsq_tot
from lights.signal_rates import Nrec_BelleII


def test_unknown_channel_or_coupling_is_an_error():
    with pytest.raises(ValueError, match="Unknown channel"):
        Model("scalar", 1.0)
    with pytest.raises(ValueError, match="has no coupling"):
        Model("phi", 1.0, CVL=1.0)  # vector coupling in the scalar channel
    with pytest.raises(ValueError, match="has no coupling"):
        Model("psipsi", 1.0, CSL_ff=1.0)  # old coupling-array style name


def test_models_are_immutable_and_copyable():
    m = Model("psipsi", 0.6, Lambda=1e4, CVL=0.5, CVR=0.5)
    with pytest.raises(AttributeError):
        m.mX = 1.0
    with pytest.raises(ValueError):
        m.C[0] = 2.0
    assert m.with_mass(0.8).mX == 0.8 and m.with_mass(0.8).couplings == m.couplings
    assert m.scaled(3.0).couplings == {"CVL": 1.5, "CVR": 1.5}
    assert m.couplings == {"CVL": 0.5, "CVR": 0.5}
    assert repr(m) == "Model('psipsi', mX = 0.6, Lambda = 10000, CVL = 0.5, CVR = 0.5)"


def test_models_pickle():
    m = Model("PsiPsi", 0.3, Lambda=1e2, is_ident=True, CVL=1.0, CTPL=0.4)
    m2 = pickle.loads(pickle.dumps(m))
    assert repr(m2) == repr(m) and np.array_equal(m2.C, m.C)


def test_rates_follow_the_channel_kind():
    two = Model("phi", 2.1, CSL=1e-8)
    three = Model("phiphi", 0.5, Lambda=1e4, CSL=1e-3)
    q = np.linspace(1.0, 20.0, 5)
    assert np.any(two.B() > 0) and np.all(two.dBdqsq(q) == 0)
    assert np.all(three.B() == 0) and np.any(three.dBdqsq(q) > 0)
    assert np.all(B_tot(None) == 0) and np.all(dBdqsq_tot(q, None) == 0) and B_Bs(None) == 0
    assert B_Bs(two) == 0 and B_Bs(three) > 0


def test_coupling_arrays_are_rejected_with_a_hint():
    with pytest.raises(TypeError, match="coupling arrays are no longer supported"):
        Nrec_BelleII(np.zeros((6, 14)))
    with pytest.raises(TypeError, match="model_1 must be"):
        min_logLL(logLL_BelleII, np.zeros((6, 14)))
