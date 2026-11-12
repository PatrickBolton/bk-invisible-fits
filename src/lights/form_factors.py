import numpy as np

from .constants import mB, mK, mKs
from .load_data import load, read_csv_to_array
from .paths import FORM_FACTOR_DIR as DATA_DIR

# Pole masses for the B -> P and B -> V form factors
Mfp = 5.4154
Mf0 = 5.711
MfT = 5.4154

MA0 = 5.366
MA1 = 5.829
MA12 = 5.829
MV = 5.415
MT1 = 5.415
MT2 = 5.829
MT23 = 5.829

# Coefficients for f+, f0 and fT form factors (B -> P)
aK_vals_GRDV = load(DATA_DIR / "GRDV" / "aK_vals_GRDV.csv")
aK_cov_GRDV = np.array(read_csv_to_array(DATA_DIR / "GRDV" / "aK_cov_GRDV.csv"))

# Coefficients for A0, A1, A12, V, T1, T2, T23 (B -> V)
aKs_vals_GRDV = load(DATA_DIR / "GRDV" / "aKs_vals_GRDV.csv")
aKs_cov_GRDV = np.array(read_csv_to_array(DATA_DIR / "GRDV" / "aKs_cov_GRDV.csv"))

# Default form factor coefficients, (B -> K, B -> K*). A tuple, so it is safe as a default argument.
A_VALS_GRDV = (aK_vals_GRDV, aKs_vals_GRDV)


def a_vals_gen(Ntoys, rng=None):
    """Generates random values for form factor a values.

    rng: a seed or numpy.random.Generator, for reproducible toys (None: fresh randomness).
    """
    rng = np.random.default_rng(rng)
    aK_vals_r = rng.multivariate_normal(aK_vals_GRDV, aK_cov_GRDV, size=Ntoys)
    aKs_vals_r = rng.multivariate_normal(aKs_vals_GRDV, aKs_cov_GRDV, size=Ntoys)
    a_vals_r = [list(pair) for pair in zip(aK_vals_r, aKs_vals_r)]

    return a_vals_r


def z(qsq, mK):
    "Conformal variable z(q^2) for B -> K(*), with t+ = (mB + mK)^2 and t0 = (mB + mK)(sqrt(mB) - sqrt(mK))^2."
    tp = (mB + mK) ** 2
    t0 = (mB + mK) * (np.sqrt(mB) - np.sqrt(mK)) ** 2
    a, b = np.sqrt(tp - qsq), np.sqrt(tp - t0)
    return (a - b) / (a + b)


_z0 = {mK: z(0.0, mK), mKs: z(0.0, mKs)}  # z(0) for each final-state meson, computed once


def _bsz(qsq, M_pole, a0, a1, a2, m_meson):
    """Pole times second-order z-expansion: (a0 + a1 dz + a2 dz^2) / (1 - q^2/M_pole^2),
    with dz = z(q^2) - z(0)."""
    dz = z(qsq, m_meson) - _z0[m_meson]
    return (a0 + a1 * dz + a2 * dz**2) / (1 - qsq / M_pole**2)


# B -> P form factors
def fpf(qsq, aK_vals=aK_vals_GRDV):
    return _bsz(qsq, Mfp, aK_vals[0], aK_vals[1], aK_vals[2], mK)


def f0f(qsq, aK_vals=aK_vals_GRDV):
    # f0(0) = f+(0): shares the constant coefficient with f+
    return _bsz(qsq, Mf0, aK_vals[0], aK_vals[3], aK_vals[4], mK)


def fTf(qsq, aK_vals=aK_vals_GRDV):
    return _bsz(qsq, MfT, aK_vals[5], aK_vals[6], aK_vals[7], mK)


# B -> V form factors
def A0f(qsq, aKs_vals=aKs_vals_GRDV):
    return _bsz(qsq, MA0, aKs_vals[0], aKs_vals[1], aKs_vals[2], mKs)


def A1f(qsq, aKs_vals=aKs_vals_GRDV):
    return _bsz(qsq, MA1, aKs_vals[3], aKs_vals[4], aKs_vals[5], mKs)


def A12f(qsq, aKs_vals=aKs_vals_GRDV):
    # A12(0) = (mB^2 - mKs^2)/(8 mB mKs) A0(0)
    aA12_0 = (mB**2 - mKs**2) / (8 * mB * mKs) * aKs_vals[0]
    return _bsz(qsq, MA12, aA12_0, aKs_vals[6], aKs_vals[7], mKs)


def Vf(qsq, aKs_vals=aKs_vals_GRDV):
    return _bsz(qsq, MV, aKs_vals[8], aKs_vals[9], aKs_vals[10], mKs)


def T1f(qsq, aKs_vals=aKs_vals_GRDV):
    return _bsz(qsq, MT1, aKs_vals[11], aKs_vals[12], aKs_vals[13], mKs)


def T2f(qsq, aKs_vals=aKs_vals_GRDV):
    # T2(0) = T1(0)
    return _bsz(qsq, MT2, aKs_vals[11], aKs_vals[14], aKs_vals[15], mKs)


def T23f(qsq, aKs_vals=aKs_vals_GRDV):
    return _bsz(qsq, MT23, aKs_vals[16], aKs_vals[17], aKs_vals[18], mKs)
