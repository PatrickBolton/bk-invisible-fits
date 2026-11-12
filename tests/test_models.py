"""The Model class, the channel registry and the old coupling-array interface."""

import pickle
import warnings

import numpy as np
import pytest

from lights.models import Model, CHANNELS, cdict, B_tot, dBdqsq_tot, B_Bs
from lights.signal_rates import Nrec_BelleII, Nrec_BaBar
from lights.likelihoods import min_logLL, logLL_BelleII


def test_unknown_channel_or_coupling_is_an_error():
    with pytest.raises(ValueError, match = "Unknown channel"):
        Model("scalar", 1.)
    with pytest.raises(ValueError, match = "has no coupling"):
        Model("phi", 1., CVL = 1.)          # vector coupling in the scalar channel
    with pytest.raises(ValueError, match = "has no coupling"):
        Model("psipsi", 1., CSL_ff = 1.)    # old cdict-style name


def test_models_are_immutable_and_copyable():
    m = Model("psipsi", 0.6, Lambda = 1e4, CVL = 0.5, CVR = 0.5)
    with pytest.raises(AttributeError):
        m.mX = 1.
    with pytest.raises(ValueError):
        m.C[0] = 2.
    assert m.with_mass(0.8).mX == 0.8 and m.with_mass(0.8).couplings == m.couplings
    assert m.scaled(3.).couplings == {"CVL": 1.5, "CVR": 1.5}
    assert m.couplings == {"CVL": 0.5, "CVR": 0.5}
    assert repr(m) == "Model('psipsi', mX = 0.6, Lambda = 10000, CVL = 0.5, CVR = 0.5)"


def test_models_pickle():
    m = Model("PsiPsi", 0.3, Lambda = 1e2, is_ident = True, CVL = 1., CTPL = 0.4)
    m2 = pickle.loads(pickle.dumps(m))
    assert repr(m2) == repr(m) and np.array_equal(m2.C, m.C)


def test_rates_follow_the_channel_kind():
    two = Model("phi", 2.1, CSL = 1e-8)
    three = Model("phiphi", 0.5, Lambda = 1e4, CSL = 1e-3)
    q = np.linspace(1., 20., 5)
    assert np.any(two.B() > 0) and np.all(two.dBdqsq(q) == 0)
    assert np.all(three.B() == 0) and np.any(three.dBdqsq(q) > 0)
    assert np.all(B_tot(None) == 0) and np.all(dBdqsq_tot(q, None) == 0) and B_Bs(None) == 0
    assert B_Bs(two) == 0 and B_Bs(three) > 0


def test_cdict_layout_matches_the_registry():
    # Old 6 x 14 layout: one row per channel, couplings in the order of the rate function
    assert len(cdict) == sum(len(ch.couplings) for ch in CHANNELS.values())
    assert cdict["CSL_s"] == (0, 0) and cdict["CTR_v"] == (1, 3) and cdict["CVR_ss"] == (2, 3)
    assert cdict["CTR_ff"] == (3, 9) and cdict["CSR_vv"] == (4, 1) and cdict["CTPR_FF"] == (5, 13)


def test_coupling_arrays_round_trip():
    m = Model("psipsi", 0.6, Lambda = 1e4, CVLp = 0.5, CTR = -0.2)
    m2 = Model.from_cnp(m.to_cnp(), 0.6, 1e4)
    assert repr(m2) == repr(m)
    assert Model.from_cnp(np.zeros((6, 14)), 1.) is None
    mixed = np.zeros((6, 14)); mixed[cdict["CSL_s"]] = 1.; mixed[cdict["CSL_ss"]] = 1.
    with pytest.raises(ValueError, match = "more than one channel"):
        Model.from_cnp(mixed, 1.)


def test_old_calls_give_the_same_results():
    m = Model("phi", 2.1, CSL = 5e-9, CSR = 5e-9)
    c = m.to_cnp()
    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter("always")
        old_B, old_b = Nrec_BelleII(0, 2.1, c, 1.), Nrec_BaBar(0, 2.1, c, 1.)
        old_fit = min_logLL(logLL_BelleII, Nrec_BelleII, 20, 2.1, c, np.zeros((6, 14)), 1., NP = 1)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert np.array_equal(old_B, Nrec_BelleII(m)) and np.array_equal(old_b, Nrec_BaBar(m))
    assert old_fit.delta_chi2 == pytest.approx(min_logLL(logLL_BelleII, m).delta_chi2, abs = 1e-6)
