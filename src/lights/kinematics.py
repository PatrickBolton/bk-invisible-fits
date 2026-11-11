import numpy as np

from .constants import mB

def lam(a, b, c):
    "Källén function."
    return (a - b - c)**2 - 4*b*c

def pKvec(mBsq, qsq, mKsq, lims):
    "Three-momentum of outgoing K(*)+."
    return np.sqrt(lam(mBsq, qsq, mKsq) * lims)/(2*mB)

def bX(mX, qsq):
    "Beta factor of outgoing invisible state X."
    return np.sqrt((1 - 4*mX**2/qsq) * np.heaviside(qsq - 4*mX**2, 1))

def lims_2body(mB, mK, mX):
    "Lower and upper limits on mX for two-body decays."
    return np.heaviside(mX, 1) * np.heaviside((mB - mK) - mX, 1)

def lims_3body(mB, mK, mX, qsq):
    "Lower and upper limits on mX and q^2 for three-body decays."
    return (np.heaviside(mX, 1) * np.heaviside((mB - mK)/2 - mX, 1)
            * np.heaviside(qsq - 4*mX**2, 1) * np.heaviside((mB - mK)**2 - qsq, 1))