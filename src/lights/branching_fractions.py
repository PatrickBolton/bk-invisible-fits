"""Branching fractions of B -> K(*) + invisible and Bs -> invisible.

The operators and their Wilson coefficients are defined in docs/operators.md. Each rate
function takes the coefficients Ci of one channel in a fixed order (see lights.models.CHANNELS
for the names), the scale Lambda of the effective operators and, for pairs, is_ident
(identical, i.e. real or Majorana, particles).

The B -> K(*) functions return four rows: K, K*, K* longitudinal and K* transverse (see
the row indices K, KS, KS_L, KS_T in lights.models). Two-body functions return branching
fractions, three-body functions dB/dq^2 in GeV^-2.
"""

import numpy as np

from .constants import mB, mBs0, mK, mKs, mb, ms, tauB, tauB0, tauBs, fBs, CSM
from .kinematics import lims_2body, lims_3body, pKvec, bX
from .form_factors import A_VALS_GRDV, fpf, f0f, fTf, Vf, A0f, A1f, A12f, T1f, T2f, T23f

# Rows of the returned B -> K(*) rates
K, KS, KS_L, KS_T = 0, 1, 2, 3

# Shared ingredients
DEL_K = 1 - (mK/mB)**2      # 1 - mK^2/mB^2
DEL_KS = 1 - (mKs/mB)**2    # 1 - mK*^2/mB^2

def _parity(CL, CR, factor = 1):
    """Parity-even and parity-odd combinations of left- and right-handed couplings,
    ((CR + CL)/2, (CR - CL)/2), each times factor."""
    return (CR + CL)/2 * factor, (CR - CL)/2 * factor

def _identity_factors(is_ident):
    """Factors (a, b) for a pair of identical (real or Majorana) particles.

    a = 2 for identical particles: operators symmetric under exchanging the two particles get
    an extra factor 2 in the amplitude, and the rate a symmetry factor 1/a for the identical
    final state. b = 0 for identical particles: operators antisymmetric under the exchange
    (the vector and tensor currents of a real scalar or Majorana fermion) vanish.
    For distinct particles a = b = 1.
    """
    return 1 + is_ident, 1 - is_ident

def _two_body_kinematics(mX):
    "Phase-space limits and K, K* momenta in B -> K(*) X (q^2 = mX^2): limsK, limsKs, pK, pKs."
    limsK = lims_2body(mB, mK, mX)
    limsKs = lims_2body(mB, mKs, mX)
    return limsK, limsKs, pKvec(mB**2, mX**2, mK**2, limsK), pKvec(mB**2, mX**2, mKs**2, limsKs)

def _three_body_kinematics(qsq, mX):
    """Phase-space limits, K, K* momenta and the velocity beta of X in B -> K(*) X X-bar:
    limsK, limsKs, pK, pKs, beta."""
    limsK = lims_3body(mB, mK, mX, qsq)
    limsKs = lims_3body(mB, mKs, mX, qsq)
    return (limsK, limsKs, pKvec(mB**2, qsq, mK**2, limsK), pKvec(mB**2, qsq, mKs**2, limsKs),
            bX(mX, qsq))

def _rates(BK_rate, BKs_rate = None, BKs_rate_L = None):
    """Rows (K, K*, K*_L, K*_T). K* rows are zero if not given; K*_T = K* - K*_L."""
    if BKs_rate is None:
        z = np.zeros_like(BK_rate)
        return np.array([BK_rate, z, z, z])
    return np.array([BK_rate, BKs_rate, BKs_rate_L, BKs_rate - BKs_rate_L])


# B -> K(*) X
def B_BKphi(mphi, Ci, a_vals = A_VALS_GRDV):
    """Branching fractions of B -> K(*) phi, a scalar of mass mphi. Ci = (CSL, CSR).

    phi has spin 0, so the K* is purely longitudinal.
    """
    gS, gP = _parity(*Ci)
    limsK, limsKs, pK, pKs = _two_body_kinematics(mphi)

    aK_vals, aKs_vals = a_vals
    f0 = f0f(mphi**2, aK_vals)
    A0 = A0f(mphi**2, aKs_vals)

    BK_rate = (tauB * pK/(8*np.pi)
               * (abs(gS)**2 * (mB*DEL_K/(mb - ms) * f0)**2)
              ) * limsK

    BKs_rate = (tauB0 * pKs/(8*np.pi)
                * (4*abs(gP)**2 * (pKs/(mb + ms) * A0)**2)
               ) * limsKs

    return _rates(BK_rate, BKs_rate, BKs_rate)

def B_BKV(mV, Ci, Lambda = 1, a_vals = A_VALS_GRDV):
    """Branching fractions of B -> K(*) V, a vector of mass mV. Ci = (CVL, CVR, CTL, CTR)."""
    CVL, CVR, CTL, CTR = Ci
    hV, hA = _parity(CVL, CVR)
    hT, hTt = _parity(CTL, CTR)
    limsK, limsKs, pK, pKs = _two_body_kinematics(mV)

    aK_vals, aKs_vals = a_vals
    q2 = mV**2
    fp, fT = fpf(q2, aK_vals), fTf(q2, aK_vals)
    V, A1, A12 = Vf(q2, aKs_vals), A1f(q2, aKs_vals), A12f(q2, aKs_vals)
    T1, T2, T23 = T1f(q2, aKs_vals), T2f(q2, aKs_vals), T23f(q2, aKs_vals)

    BK_rate = (tauB * pK/(2*np.pi)
               * (abs(hV)**2 * (pK/mV)**2 * fp**2
                  + 4*abs(hT)**2 * (pK/Lambda)**2 * (mV/(mB + mK) * fT)**2
                  + 4*np.real(hV*np.conjugate(hT)) * pK**2/((mB + mK)*Lambda) * fp * fT)
              ) * limsK

    def Ks_rate(A1, V, T1, T2):
        "K* rate for the given transverse form factors (A1, V, T1, T2); zero them for the longitudinal part."
        return (tauB0 * pKs/(2*np.pi)
                * (2*abs(hV)**2 * (pKs/(mB + mKs) * V)**2
                   + abs(hA)**2 * (1/2*((mB + mKs)/mB * A1)**2 + 16*(mKs/mV * A12)**2)
                   + 8*abs(hT)**2 * (pKs/Lambda * T1)**2
                   + 2*abs(hTt)**2 * (mV/Lambda)**2 * ((mB*DEL_KS/mV * T2)**2 + 8 * (mKs/(mB + mKs) * T23)**2)
                   - 8*np.real(hV*np.conjugate(hT)) * pKs**2/((mB + mKs)*Lambda) * V*T1
                   - 2*np.real(hA*np.conjugate(hTt)) * (mB + mKs)/Lambda * (DEL_KS * A1*T2 + 16*(mKs/(mB + mKs))**2 * A12*T23))
               ) * limsKs

    # longitudinal K*: the transverse form factors set to zero
    return _rates(BK_rate, Ks_rate(A1, V, T1, T2), Ks_rate(0, 0, 0, 0))


# B -> K(*) X X-bar
def dBdqsq_SM(qsq, is_ident = False, a_vals = A_VALS_GRDV):
    """SM dB/dq^2 for B -> K(*) nu nu-bar: three massless neutrino flavours with the
    left-handed current, (CVL, CVLp) = (CSM/2, -CSM/2) in the psi psi-bar basis."""
    return 3 * dBdqsq_BKpsipsi(qsq, 0., [CSM/2, 0, -CSM/2, 0, 0, 0, 0, 0, 0, 0], 1, is_ident, a_vals)

def dBdqsq_BKphiphi(qsq, mphi, Ci, Lambda = 1, is_ident = False, a_vals = A_VALS_GRDV):
    """dB/dq^2 of B -> K(*) phi phi-bar, a pair of scalars of mass mphi. Ci = (CSL, CSR, CVL, CVR)."""
    a, b = _identity_factors(is_ident)
    CSL, CSR, CVL, CVR = Ci
    gSS, gPS = _parity(CSL, CSR, a)
    gVV, gAV = _parity(CVL, CVR, b)
    limsK, limsKs, pK, pKs, bphi = _three_body_kinematics(qsq, mphi)

    aK_vals, aKs_vals = a_vals
    fp, f0 = fpf(qsq, aK_vals), f0f(qsq, aK_vals)
    V, A0, A1, A12 = Vf(qsq, aKs_vals), A0f(qsq, aKs_vals), A1f(qsq, aKs_vals), A12f(qsq, aKs_vals)

    BK_rate = (tauB * bphi/(96*np.pi**3) * pK/Lambda**2 * 1/a
               * (3/4*abs(gSS)**2 * (mB*DEL_K/(mb - ms) * f0)**2
                    + abs(gVV)**2 * (pK/Lambda)**2 * bphi**2 * fp**2)
              ) * limsK

    def Ks_rate(V, A1):
        "K* rate for the given transverse form factors (V, A1); zero them for the longitudinal part."
        return (tauB0 * bphi/(96*np.pi**3) * pKs/Lambda**2 * 1/a
                * (3*abs(gPS)**2 * (pKs/(mb + ms) * A0)**2
                   + 2*abs(gVV)**2 * qsq/Lambda**2 * bphi**2 * (pKs/(mB + mKs) * V)**2
                   + abs(gAV)**2 * qsq/Lambda**2 * bphi**2 * (1/2*((mB + mKs)/mB * A1)**2 + 16*(mKs*A12)**2/qsq))
               ) * limsKs

    # longitudinal K*: the transverse form factors set to zero
    return _rates(BK_rate, Ks_rate(V, A1), Ks_rate(0, 0))

def dBdqsq_BKpsipsi(qsq, mpsi, Ci, Lambda = 1, is_ident = False, a_vals = A_VALS_GRDV):
    """dB/dq^2 of B -> K(*) psi psi-bar, a pair of fermions of mass mpsi.

    Ci = (CVL, CVR, CVLp, CVRp, CSL, CSR, CSLp, CSRp, CTL, CTR); the primed couplings have
    a gamma_5 in the fermion current.
    """
    a, b = _identity_factors(is_ident)
    CVL, CVR, CVLp, CVRp, CSL, CSR, CSLp, CSRp, CTL, CTR = Ci
    fVV, fAV = _parity(CVL, CVR, b)
    fVA, fAA = _parity(CVLp, CVRp, a)
    fSS, fPS = _parity(CSL, CSR, a)
    fSP, fPP = _parity(CSLp, CSRp, a)
    fTT, fTtT = _parity(CTL, CTR, b)
    limsK, limsKs, pK, pKs, bpsi = _three_body_kinematics(qsq, mpsi)

    b2 = bpsi**2
    omb2 = 4*mpsi**2/qsq   # = 1 - b2, without the cancellation as mpsi -> 0
    bp2 = (3 - b2)/2
    bpp2 = (3 - 2*b2)

    aK_vals, aKs_vals = a_vals
    fp, f0, fT = fpf(qsq, aK_vals), f0f(qsq, aK_vals), fTf(qsq, aK_vals)
    V, A0, A1, A12 = Vf(qsq, aKs_vals), A0f(qsq, aKs_vals), A1f(qsq, aKs_vals), A12f(qsq, aKs_vals)
    T1, T2, T23 = T1f(qsq, aKs_vals), T2f(qsq, aKs_vals), T23f(qsq, aKs_vals)

    BK_rate = (tauB * bpsi/(24*np.pi**3) * pK*qsq/Lambda**4 * 1/a
               * (((abs(fVV)**2 * bp2 + abs(fVA)**2 * b2) * fp
                   + 12*np.real(fVV * np.conjugate(fTT)) * mpsi/(mB + mK) * fT) * pK**2/qsq * fp
                  + 3/8*(abs(fVA)**2 * omb2 * (mb - ms)**2/qsq
                         + abs(fSS)**2 * b2 + abs(fSP)**2
                         + 4*np.real(fVA * np.conjugate(fSP)) * mpsi*(mb - ms)/qsq) * (mB*DEL_K/(mb - ms) * f0)**2
                  + 2*(abs(fTT)**2 * bpp2 + abs(fTtT)**2 * b2) * (pK/(mB + mK) * fT)**2)
              ) * limsK

    def Ks_rate(A1, V, T1, T2):
        "K* rate for the given transverse form factors (A1, V, T1, T2); zero them for the longitudinal part."
        return (tauB0 * bpsi/(24*np.pi**3) * pKs*qsq/Lambda**4 * 1/a
                * (2*(abs(fVV)**2 * bp2 + abs(fVA)**2 * b2) * (pKs/(mB + mKs) * V)**2
                   + (abs(fAV)**2 * bp2 + abs(fAA)**2 * b2) * (1/2*((mB + mKs)/mB * A1)**2 + 16*(mKs * A12)**2/qsq)
                   + 3/2*(abs(fAA)**2 * omb2 * (mb + ms)**2/qsq
                          + abs(fPS)**2 * b2 + abs(fPP)**2
                          - 4*np.real(fAA * np.conjugate(fPP)) * mpsi*(mb + ms)/qsq) * (pKs/(mb + ms) * A0)**2
                   + 4*((abs(fTT)**2 * bpp2 + abs(fTtT)**2 * b2) * T1
                        - 6*np.real(fVV * np.conjugate(fTT))* mpsi/(mB + mKs) * V) * pKs**2/qsq * T1
                   + (abs(fTT)**2 * b2 + abs(fTtT)**2 * bpp2) * ((mB*DEL_KS*T2)**2/qsq + 8*(mKs/(mB + mKs) * T23)**2)
                   - 6*np.real(fAV * np.conjugate(fTtT)) * mpsi*(mB + mKs)/qsq * (DEL_KS * A1*T2
                                                                                  + 16*(mKs/(mB + mKs))**2 * A12*T23))
               ) * limsKs

    # longitudinal K*: the transverse form factors set to zero
    return _rates(BK_rate, Ks_rate(A1, V, T1, T2), Ks_rate(0, 0, 0, 0))

def dBdqsq_BKVV(qsq, mV, Ci, Lambda = 1, is_ident = False, a_vals = A_VALS_GRDV):
    """dB/dq^2 of B -> K V V-bar, a pair of vectors of mass mV. Ci = (CSL, CSR).

    Only the K rate is implemented; the K* rows are zero.
    """
    a, _ = _identity_factors(is_ident)
    hS, _ = _parity(*Ci, a)
    limsK, _, pK, _, bV = _three_body_kinematics(qsq, mV)
    b2 = bV**2
    b4 = bV**4

    aK_vals, _ = a_vals
    f0 = f0f(qsq, aK_vals)

    BK_rate = (tauB * bV/(512*np.pi**3) * pK*qsq**2/Lambda**6 * 1/a
               * (abs(hS)**2 * (3 - 2*b2 + 3*b4)/4 * (mB*DEL_K/(mb - ms) * f0)**2)
              ) * limsK

    return _rates(BK_rate)

def dBdqsq_BKPsiPsi(qsq, mPsi, Ci, Lambda = 1, is_ident = False, a_vals = A_VALS_GRDV):
    """dB/dq^2 of B -> K Psi Psi-bar, a pair of spin-3/2 fermions of mass mPsi.

    Ci = (CVL, CVR, CVLp, CVRp, CSL, CSR, CSLp, CSRp, CTL, CTR, CTSL, CTSR, CTPL, CTPR).
    Only the K rate is implemented; the K* rows are zero.
    """
    a, b = _identity_factors(is_ident)
    CVL, CVR, CVLp, CVRp, CSL, CSR, CSLp, CSRp, CTL, CTR, CTSL, CTSR, CTPL, CTPR = Ci
    FVV, FAV = _parity(CVL, CVR, b)
    FVA, FAA = _parity(CVLp, CVRp, a)
    FSS, FPS = _parity(CSL, CSR, a)
    FSP, FPP = _parity(CSLp, CSRp, a)
    FTT, FTtT = _parity(CTL, CTR, b)
    FTS, FTtS = _parity(CTSL, CTSR, a)
    FTP, FTtP = _parity(CTPL, CTPR, a)
    limsK, _, pK, _, bPsi = _three_body_kinematics(qsq, mPsi)

    b2 = bPsi**2
    omb2 = 4*mPsi**2/qsq   # = 1 - b2, without the cancellation as mPsi -> 0
    b4 = bPsi**4
    b6 = bPsi**6
    J = (9 - 6*b2 + 5*b4)/8

    aK_vals, _ = a_vals
    fp, f0, fT = fpf(qsq, aK_vals), f0f(qsq, aK_vals), fTf(qsq, aK_vals)

    BK_rate = (tauB * 1/9 * bPsi/(24*np.pi**3) * pK*qsq**3/Lambda**8 * 1/a
               * (((abs(FVV)**2 * (15 - 15*b2 + 25*b4 - 9*b6)/16 + abs(FVA)**2 * b2 * (5 - 2*b2 + 5*b4)/8) * fp
                   + 2*np.real(FVV * (6*np.conjugate(FTT) * (15 - 10*b2 + 19*b4)/24
                                      + 2*np.conjugate(FTS) * b4
                                      + np.conjugate(FTtP) * (15 - 10*b2 + 3*b4)/8)) * mPsi/(mB + mK) * fT) *  pK**2/qsq * fp
                  + 3/8*(abs(FVA)**2 * omb2 * J * (mb - ms)**2/qsq
                         + abs(FSS)**2 * b2 * (5 - 6*b2 + 9*b4)/8 + abs(FSP)**2 * J
                         + 4*np.real(FVA*np.conjugate(FSP)) * J * mPsi*(mb - ms)/qsq) * (mB*DEL_K/(mb - ms) * f0)**2
                  + 2*(abs(FTT)**2 * (15 - 20*b2 + 23*b4 - 10*b6)/8
                       + 1/4*abs(FTS)**2 * b4 * (9 - 5*b2)/4
                       + 1/4*abs(FTP)**2 * b2 * (7 - 3*b2)/4
                       + abs(FTtT)**2 * b2 * J
                       + 3/16*abs(FTtS)**2 * b2 * omb2 * (11 - 5*b2)/6
                       + 5/16*abs(FTtP)**2 * omb2 * (3 - b2)/2
                       + np.real(FTT*(np.conjugate(FTS) * b4 * (9 - 5*b2)/4
                                      + 5/4*np.conjugate(FTtP) * omb2 * (3 - b2)/2)
                                 + FTtT*(np.conjugate(FTP) * b2 * (3 + b2)/4
                                         - 1/4*np.conjugate(FTtS) * b2 * omb2 * (-3 + 5*b2)/2)
                                 - 1/2*FTP*np.conjugate(FTtS) * b2 * omb2)) * (pK/(mB + mK) * fT)**2)
                 ) * limsK

    return _rates(BK_rate)


# Bs -> X X-bar
def B_Bsphiphi(mphi, Ci, Lambda = 1, is_ident = False):
    """B(Bs -> phi phi-bar), scalars of mass mphi. Ci = (CSL, CSR, CVL, CVR).

    Only the pseudoscalar coupling contributes: the scalar quark current has no Bs -> vacuum
    matrix element, and the vector currents give p.(p1 - p2) = 0 for equal masses.
    """
    a, _ = _identity_factors(is_ident)
    CSL, CSR, _, _ = Ci
    _, gPS = _parity(CSL, CSR, a)
    lims = lims_2body(mBs0, mphi, mphi)
    bphi = bX(mphi, mBs0**2)

    Bs_rate = (tauBs * bphi/(16*np.pi) * fBs**2*mBs0/Lambda**2 * 1/a
               * (abs(gPS)**2 * (mBs0/(mb + ms))**2 )
              ) * lims

    return Bs_rate

def B_Bspsipsi(mpsi, Ci, Lambda = 1, is_ident = False):
    """B(Bs -> psi psi-bar), fermions of mass mpsi. Ci as for dBdqsq_BKpsipsi.

    Only the axial-vector (fAA) and pseudoscalar (fPS, fPP) couplings contribute.
    """
    a, _ = _identity_factors(is_ident)
    _, _, CVLp, CVRp, CSL, CSR, CSLp, CSRp, _, _ = Ci
    _, fAA = _parity(CVLp, CVRp, a)
    _, fPS = _parity(CSL, CSR, a)
    _, fPP = _parity(CSLp, CSRp, a)
    lims = lims_2body(mBs0, mpsi, mpsi)
    bpsi = bX(mpsi, mBs0**2)
    b2 = bpsi**2
    omb2 = 4*mpsi**2/mBs0**2   # = 1 - b2, without the cancellation as mpsi -> 0

    Bs_rate = (tauBs * bpsi/(8*np.pi) * fBs**2*mBs0**3/Lambda**4 * (mBs0/(mb + ms))**2 * 1/a
               * (abs(fAA)**2 * omb2 * ((mb + ms)/mBs0)**2
                  + abs(fPS)**2 * b2 + abs(fPP)**2
                  - 4*np.real(fAA * np.conjugate(fPP)) * mpsi*(mb + ms)/mBs0**2)
              ) * lims

    return Bs_rate


# Channel totals and the coupling layout now live in lights.models (CHANNELS, Model,
# B_tot, dBdqsq_tot, B_Bs). The old names below are kept for compatibility.
def __getattr__(name):
    if name in ("cdict", "B_tot", "dBdqsq_tot", "B_Bstot"):
        import warnings
        from . import models
        warnings.warn(f"lights.branching_fractions.{name} is deprecated; use lights.models "
                      "(Model, CHANNELS, B_tot, dBdqsq_tot, B_Bs)", DeprecationWarning, stacklevel = 2)
        if name == "cdict":
            return models.cdict
        def legacy(*args, **kwargs):
            if name == "B_tot":         # B_tot(mX, cX, Lambda, a_vals), cX = rows 0-1
                mX, cX, Lambda, a_vals = args
                cNP = np.zeros((6, 14)); cNP[0:2] = cX
                return models.B_tot(models.Model.from_cnp(cNP, mX, Lambda), a_vals)
            if name == "dBdqsq_tot":    # dBdqsq_tot(qsq, mX, cXY, Lambda, is_ident, a_vals), cXY = rows 2-5
                vals = dict(zip(("qsq", "mX", "cXY", "Lambda", "is_ident", "a_vals"), args), **kwargs)
                cNP = np.zeros((6, 14)); cNP[2:6] = vals["cXY"]
                m = models.Model.from_cnp(cNP, vals["mX"], vals["Lambda"], vals.get("is_ident", False))
                return models.dBdqsq_tot(vals["qsq"], m, vals.get("a_vals", A_VALS_GRDV))
            vals = dict(zip(("mX", "cNP", "Lambda", "is_ident"), args), **kwargs)   # B_Bstot
            return models.B_Bs(models.Model.from_cnp(vals["cNP"], vals["mX"], vals["Lambda"],
                                                     vals.get("is_ident", False)))
        return legacy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
