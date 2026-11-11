from .constants import mB, mK, mKs
from .load_data import load, read_csv_to_array

import numpy as np
import numpy.random as npr

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

def a_vals_gen(Ntoys):
    """Generates random values for form factor a values."""
    aK_vals_r = npr.multivariate_normal(aK_vals_GRDV, aK_cov_GRDV, size = Ntoys)
    aKs_vals_r = npr.multivariate_normal(aKs_vals_GRDV, aKs_cov_GRDV, size = Ntoys)
    a_vals_r = [list(pair) for pair in zip(aK_vals_r, aKs_vals_r)]
    
    return a_vals_r

def z(qsq, mK):
    tp = (mB + mK)**2
    t0 = (mB + mK) * (np.sqrt(mB) - np.sqrt(mK))**2
    return (np.sqrt(tp - qsq) - np.sqrt(tp - t0))/(np.sqrt(tp - qsq) + np.sqrt(tp - t0))

# B -> P form factors
def fpf(qsq, aK_vals = aK_vals_GRDV):
    afp_0, afp_1, afp_2 = aK_vals[0:3]
    Pfp = 1 - qsq/Mfp**2
    return 1/Pfp * (afp_0 + afp_1 * (z(qsq, mK) - z(0, mK)) + afp_2 * (z(qsq, mK) - z(0, mK))**2)

def f0f(qsq, aK_vals = aK_vals_GRDV):
    af0_0 = aK_vals[0]
    af0_1, af0_2 = aK_vals[3:5]
    Pf0 = 1 - qsq/Mf0**2
    return 1/Pf0 * (af0_0 + af0_1 * (z(qsq, mK) - z(0, mK)) + af0_2 * (z(qsq, mK) - z(0, mK))**2)

def fTf(qsq, aK_vals = aK_vals_GRDV):
    afT_0, afT_1, afT_2 = aK_vals[5:8]
    PfT = 1 - qsq/MfT**2
    return 1/PfT * (afT_0 + afT_1 * (z(qsq, mK) - z(0, mK)) + afT_2 * (z(qsq, mK) - z(0, mK))**2)

# B -> V form factors
def A0f(qsq, aKs_vals = aKs_vals_GRDV):
    aA0_0, aA0_1, aA0_2 = aKs_vals[0:3]
    PA0 = 1 - qsq/MA0**2
    return 1/PA0 * (aA0_0 + aA0_1 * (z(qsq, mKs) - z(0, mKs)) + aA0_2 * (z(qsq, mKs) - z(0, mKs))**2)

def A1f(qsq, aKs_vals = aKs_vals_GRDV):
    aA1_0, aA1_1, aA1_2 = aKs_vals[3:6]
    PA1 = 1 - qsq/MA1**2
    return 1/PA1 * (aA1_0 + aA1_1 * (z(qsq, mKs) - z(0, mKs)) + aA1_2 * (z(qsq, mKs) - z(0, mKs))**2)

def A12f(qsq, aKs_vals = aKs_vals_GRDV):
    aA12_0 = (mB**2 - mKs**2)/(8*mB*mKs) * aKs_vals[0]
    aA12_1, aA12_2 = aKs_vals[6:8]
    PA12 = 1 - qsq/MA12**2
    return 1/PA12 * (aA12_0 + aA12_1 * (z(qsq, mKs) - z(0, mKs)) + aA12_2 * (z(qsq, mKs) - z(0, mKs))**2)

def Vf(qsq, aKs_vals = aKs_vals_GRDV):
    aV_0, aV_1, aV_2 = aKs_vals[8:11]
    PV = 1 - qsq/MV**2
    return 1/PV * (aV_0 + aV_1 * (z(qsq, mKs) - z(0, mKs)) + aV_2 * (z(qsq, mKs) - z(0, mKs))**2)

def T1f(qsq, aKs_vals = aKs_vals_GRDV):
    aT1_0, aT1_1, aT1_2 = aKs_vals[11:14]
    PT1 = 1 - qsq/MT1**2
    return 1/PT1 * (aT1_0 + aT1_1 * (z(qsq, mKs) - z(0, mKs)) + aT1_2 * (z(qsq, mKs) - z(0, mKs))**2)

def T2f(qsq, aKs_vals = aKs_vals_GRDV):
    aT2_0 = aKs_vals[11]
    aT2_1, aT2_2 = aKs_vals[14:16]
    PT2 = 1 - qsq/MT2**2
    return 1/PT2 * (aT2_0 + aT2_1 * (z(qsq, mKs) - z(0, mKs)) + aT2_2 * (z(qsq, mKs) - z(0, mKs))**2)

def T23f(qsq, aKs_vals = aKs_vals_GRDV):
    aT23_0, aT23_1, aT23_2 = aKs_vals[16:19]
    PT23 = 1 - qsq/MT23**2
    return 1/PT23 * (aT23_0 + aT23_1 * (z(qsq, mKs) - z(0, mKs)) + aT23_2 * (z(qsq, mKs) - z(0, mKs))**2)
