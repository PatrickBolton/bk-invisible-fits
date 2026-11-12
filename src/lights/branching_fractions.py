import numpy as np

from .constants import mB, mBs0, mK, mKs, mb, ms, tauB, tauB0, tauBs, fBs, CSM
from .kinematics import lims_2body, lims_3body, pKvec, bX
from .form_factors import aK_vals_GRDV, aKs_vals_GRDV, A_VALS_GRDV, fpf, f0f, fTf, Vf, A0f, A1f, A12f, T1f, T2f, T23f

# B -> K(*)X branching fractions
def B_BKphi(mphi, Ci, a_vals = A_VALS_GRDV):
    """Branching fraction for B -> K(*) phi."""
    
    CSL, CSR = Ci
    gS = (CSR + CSL)/2
    gP = (CSR - CSL)/2
    
    limsK = lims_2body(mB, mK, mphi)
    limsKs = lims_2body(mB, mKs, mphi)
    
    pK = pKvec(mB**2, mphi**2, mK**2, limsK)
    pKs = pKvec(mB**2, mphi**2, mKs**2, limsKs)

    aK_vals, aKs_vals = a_vals
    
    f0 = f0f(mphi**2, aK_vals)
    A0 = A0f(mphi**2, aKs_vals)
    
    delK = 1 - (mK/mB)**2
    
    BK_rate = (tauB * pK/(8*np.pi) 
               * (abs(gS)**2 * (mB*delK/(mb - ms) * f0)**2)
              ) * limsK
    
    BKs_rate = (tauB0 * pKs/(8*np.pi) 
                * (4*abs(gP)**2 * (pKs/(mb + ms) * A0)**2)
               ) * limsKs
    
    BKs_rate_L = BKs_rate

    BKs_rate_T = BKs_rate - BKs_rate_L
    
    return np.array([BK_rate, BKs_rate, BKs_rate_L, BKs_rate_T])

def B_BKV(mV, Ci, Lambda = 1, a_vals = A_VALS_GRDV):
    """Branching fraction for B -> K(*) V."""

    CVL, CVR, CTL, CTR = Ci
    hV = (CVR + CVL)/2
    hA = (CVR - CVL)/2
    hT = (CTR + CTL)/2
    hTt = (CTR - CTL)/2
    
    limsK = lims_2body(mB, mK, mV)
    limsKs = lims_2body(mB, mKs, mV)
    
    pK = pKvec(mB**2, mV**2, mK**2, limsK)
    pKs = pKvec(mB**2, mV**2, mKs**2, limsKs)

    aK_vals, aKs_vals = a_vals
    
    fp = fpf(mV**2, aK_vals)
    fT = fTf(mV**2, aK_vals)
    V = Vf(mV**2, aKs_vals)
    A1 = A1f(mV**2, aKs_vals)
    A12 = A12f(mV**2, aKs_vals)
    T1 = T1f(mV**2, aKs_vals)
    T2 = T2f(mV**2, aKs_vals)
    T23 = T23f(mV**2, aKs_vals)
    
    delKs = 1 - (mKs/mB)**2
    
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
                   + 2*abs(hTt)**2 * (mV/Lambda)**2 * ((mB*delKs/mV * T2)**2 + 8 * (mKs/(mB + mKs) * T23)**2)
                   - 8*np.real(hV*np.conjugate(hT)) * pKs**2/((mB + mKs)*Lambda) * V*T1 
                   - 2*np.real(hA*np.conjugate(hTt)) * (mB + mKs)/Lambda * (delKs * A1*T2 + 16*(mKs/(mB + mKs))**2 * A12*T23))
               ) * limsKs

    BKs_rate = Ks_rate(A1, V, T1, T2)
    BKs_rate_L = Ks_rate(0, 0, 0, 0)   # longitudinal K*: transverse form factors set to zero
    
    BKs_rate_T = BKs_rate - BKs_rate_L
    
    return np.array([BK_rate, BKs_rate, BKs_rate_L, BKs_rate_T])

# B -> K(*)XX branching fractions
def dBdqsq_SM(qsq, is_ident = False, a_vals = A_VALS_GRDV):
    """SM differential branching fraction for B -> K(*) v v in q^2."""
    return 3 * dBdqsq_BKpsipsi(qsq, 0., [CSM/2, 0, -CSM/2, 0, 0, 0, 0, 0, 0, 0], 1, is_ident, a_vals)

def dBdqsq_BKphiphi(qsq, mphi, Ci, Lambda = 1, is_ident = False, a_vals = A_VALS_GRDV):
    """Differential branching fraction for B -> K(*) phi phi in q^2."""
   
    a = 1 + is_ident
    b = 1 - is_ident
    
    CSL, CSR, CVL, CVR = Ci
    gSS = (CSR + CSL)/2 * a
    gPS = (CSR - CSL)/2 * a
    gVV = (CVR + CVL)/2 * b
    gAV = (CVR - CVL)/2 * b
    
    limsK = lims_3body(mB, mK, mphi, qsq)
    limsKs = lims_3body(mB, mKs, mphi, qsq)
    
    pK = pKvec(mB**2, qsq, mK**2, limsK)
    pKs = pKvec(mB**2, qsq, mKs**2, limsKs)
    bphi = bX(mphi, qsq)

    aK_vals, aKs_vals = a_vals

    fp = fpf(qsq, aK_vals)
    f0 = f0f(qsq, aK_vals)
    V = Vf(qsq, aKs_vals)
    A0 = A0f(qsq, aKs_vals)
    A1 = A1f(qsq, aKs_vals)
    A12 = A12f(qsq, aKs_vals)
    
    delK = 1 - (mK/mB)**2

    BK_rate = (tauB * bphi/(96*np.pi**3) * pK/Lambda**2 * 1/a
               * (3/4*abs(gSS)**2 * (mB*delK/(mb - ms) * f0)**2
                    + abs(gVV)**2 * (pK/Lambda)**2 * bphi**2 * fp**2)
              ) * limsK

    def Ks_rate(V, A1):
        "K* rate for the given transverse form factors (V, A1); zero them for the longitudinal part."
        return (tauB0 * bphi/(96*np.pi**3) * pKs/Lambda**2 * 1/a
                * (3*abs(gPS)**2 * (pKs/(mb + ms) * A0)**2
                   + 2*abs(gVV)**2 * qsq/Lambda**2 * bphi**2 * (pKs/(mB + mKs) * V)**2
                   + abs(gAV)**2 * qsq/Lambda**2 * bphi**2 * (1/2*((mB + mKs)/mB * A1)**2 + 16*(mKs*A12)**2/qsq))
               ) * limsKs

    BKs_rate = Ks_rate(V, A1)
    BKs_rate_L = Ks_rate(0, 0)   # longitudinal K*: transverse form factors set to zero
    
    BKs_rate_T = BKs_rate - BKs_rate_L
    
    return np.array([BK_rate, BKs_rate, BKs_rate_L, BKs_rate_T])

def dBdqsq_BKpsipsi(qsq, mpsi, Ci, Lambda = 1, is_ident = False, a_vals = A_VALS_GRDV):
    """Differential branching fraction for B -> K(*) psi psi in q^2."""

    a = 1 + is_ident
    b = 1 - is_ident

    CVL, CVR, CVLp, CVRp, CSL, CSR, CSLp, CSRp, CTL, CTR = Ci
    fVV = (CVR + CVL)/2 * b
    fAV = (CVR - CVL)/2 * b
    fVA = (CVRp + CVLp)/2 * a
    fAA = (CVRp - CVLp)/2 * a
    fSS = (CSR + CSL)/2 * a
    fPS = (CSR - CSL)/2 * a
    fSP = (CSRp + CSLp)/2 * a
    fPP = (CSRp - CSLp)/2 * a
    fTT = (CTR + CTL)/2 * b
    fTtT = (CTR - CTL)/2 * b

    limsK = lims_3body(mB, mK, mpsi, qsq)
    limsKs = lims_3body(mB, mKs, mpsi, qsq)

    pK = pKvec(mB**2, qsq, mK**2, limsK)
    pKs = pKvec(mB**2, qsq, mKs**2, limsKs)
    
    bpsi = bX(mpsi, qsq)
    b2 = bpsi**2
    omb2 = 4*mpsi**2/qsq   # = 1 - b2, without the cancellation as mpsi -> 0
    bp2 = (3 - b2)/2
    bpp2 = (3 - 2*b2)

    aK_vals, aKs_vals = a_vals
    
    fp = fpf(qsq, aK_vals)
    f0 = f0f(qsq, aK_vals)
    fT = fTf(qsq, aK_vals)
    V = Vf(qsq, aKs_vals)
    A0 = A0f(qsq, aKs_vals)
    A1 = A1f(qsq, aKs_vals)
    A12 = A12f(qsq, aKs_vals)
    T1 = T1f(qsq, aKs_vals)
    T2 = T2f(qsq, aKs_vals)
    T23 = T23f(qsq, aKs_vals)
    
    delK = 1 - (mK/mB)**2
    delKs = 1 - (mKs/mB)**2
    
    BK_rate = (tauB * bpsi/(24*np.pi**3) * pK*qsq/Lambda**4 * 1/a
               * (((abs(fVV)**2 * bp2 + abs(fVA)**2 * b2) * fp 
                   + 12*np.real(fVV * np.conjugate(fTT)) * mpsi/(mB + mK) * fT) * pK**2/qsq * fp
                  + 3/8*(abs(fVA)**2 * omb2 * (mb - ms)**2/qsq 
                         + abs(fSS)**2 * b2 + abs(fSP)**2 
                         + 4*np.real(fVA * np.conjugate(fSP)) * mpsi*(mb - ms)/qsq) * (mB*delK/(mb - ms) * f0)**2
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
                   + (abs(fTT)**2 * b2 + abs(fTtT)**2 * bpp2) * ((mB*delKs*T2)**2/qsq + 8*(mKs/(mB + mKs) * T23)**2)
                   - 6*np.real(fAV * np.conjugate(fTtT)) * mpsi*(mB + mKs)/qsq * (delKs * A1*T2
                                                                                  + 16*(mKs/(mB + mKs))**2 * A12*T23))
               ) * limsKs

    BKs_rate = Ks_rate(A1, V, T1, T2)
    BKs_rate_L = Ks_rate(0, 0, 0, 0)   # longitudinal K*: transverse form factors set to zero
    
    BKs_rate_T = BKs_rate - BKs_rate_L
    
    return np.array([BK_rate, BKs_rate, BKs_rate_L, BKs_rate_T])

def dBdqsq_BKVV(qsq, mV, Ci, Lambda = 1, is_ident = False, a_vals = A_VALS_GRDV):
    """Differential branching fraction for B -> K(*) V V in q^2."""

    a = 1 + is_ident
    b = 1 - is_ident
    
    CSL, CSR = Ci
    hS = (CSR + CSL)/2 * a
    hP = (CSR - CSL)/2 * a
    
    limsK = lims_3body(mB, mK, mV, qsq)
    limsKs = lims_3body(mB, mKs, mV, qsq)
    
    pK = pKvec(mB**2, qsq, mK**2, limsK)
    pKs = pKvec(mB**2, qsq, mKs**2, limsKs)
    
    bV = bX(mV, qsq)
    b2 = bV**2
    b4 = bV**4

    aK_vals, aKs_vals = a_vals
    
    f0 = f0f(qsq, aK_vals)
    A0 = A0f(qsq, aKs_vals)
    
    delK = 1 - (mK/mB)**2
    
    BK_rate = (tauB * bV/(512*np.pi**3) * pK*qsq**2/Lambda**6 * 1/a
               * (abs(hS)**2 * (3 - 2*b2 + 3*b4)/4 * (mB*delK/(mb - ms) * f0)**2)
              ) * limsK
    
    z = np.zeros_like(BK_rate)
    return np.array([BK_rate, z, z, z])

def dBdqsq_BKPsiPsi(qsq, mPsi, Ci, Lambda = 1, is_ident = False, a_vals = A_VALS_GRDV):
    """Differential branching fraction for B -> K(*) Psi Psi in q^2."""

    a = 1 + is_ident
    b = 1 - is_ident
    
    CVL, CVR, CVLp, CVRp, CSL, CSR, CSLp, CSRp, CTL, CTR, CTSL, CTSR, CTPL, CTPR = Ci
    FVV = (CVR + CVL)/2 * b
    FAV = (CVR - CVL)/2 * b
    FVA = (CVRp + CVLp)/2 * a
    FAA = (CVRp - CVLp)/2 * a
    FSS = (CSR + CSL)/2 * a
    FPS = (CSR - CSL)/2 * a
    FSP = (CSRp + CSLp)/2 * a
    FPP = (CSRp - CSLp)/2 * a
    FTT = (CTR + CTL)/2 * b
    FTtT = (CTR - CTL)/2 * b
    FTS = (CTSR + CTSL)/2 * a
    FTtS = (CTSR - CTSL)/2 * a
    FTP = (CTPR + CTPL)/2 * a
    FTtP = (CTPR - CTPL)/2 * a

    limsK = lims_3body(mB, mK, mPsi, qsq)
    limsKs = lims_3body(mB, mKs, mPsi, qsq)
    
    pK = pKvec(mB**2, qsq, mK**2, limsK)
    pKs = pKvec(mB**2, qsq, mKs**2, limsKs)
    
    bPsi = bX(mPsi, qsq)
    b2 = bPsi**2
    omb2 = 4*mPsi**2/qsq   # = 1 - b2, without the cancellation as mPsi -> 0
    b4 = bPsi**4
    b6 = bPsi**6
    J = (9 - 6*b2 + 5*b4)/8

    aK_vals, aKs_vals = a_vals

    fp = fpf(qsq, aK_vals)
    f0 = f0f(qsq, aK_vals)
    fT = fTf(qsq, aK_vals)
    
    delK = 1 - (mK/mB)**2
    
    BK_rate = (tauB * 1/9 * bPsi/(24*np.pi**3) * pK*qsq**3/Lambda**8 * 1/a
               * (((abs(FVV)**2 * (15 - 15*b2 + 25*b4 - 9*b6)/16 + abs(FVA)**2 * b2 * (5 - 2*b2 + 5*b4)/8) * fp 
                   + 2*np.real(FVV * (6*np.conjugate(FTT) * (15 - 10*b2 + 19*b4)/24
                                      + 2*np.conjugate(FTS) * b4
                                      + np.conjugate(FTtP) * (15 - 10*b2 + 3*b4)/8)) * mPsi/(mB + mK) * fT) *  pK**2/qsq * fp
                  + 3/8*(abs(FVA)**2 * omb2 * J * (mb - ms)**2/qsq
                         + abs(FSS)**2 * b2 * (5 - 6*b2 + 9*b4)/8 + abs(FSP)**2 * J 
                         + 4*np.real(FVA*np.conjugate(FSP)) * J * mPsi*(mb - ms)/qsq) * (mB*delK/(mb - ms) * f0)**2
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

    z = np.zeros_like(BK_rate)
    return np.array([BK_rate, z, z, z])

# Bs -> XX branching fractions
def B_Bsphiphi(mphi, Ci, Lambda = 1, is_ident = False):
    """Branching fraction for Bs -> phi phi."""
    
    CSL, CSR, _, _ = Ci
    
    a = 1 + is_ident
    b = 1 - is_ident
        
    gSS = (CSR + CSL)/2 * a
    gPS = (CSR - CSL)/2 * a
    
    lims = lims_2body(mBs0, mphi, mphi)
    
    bphi = bX(mphi, mBs0**2)
    
    Bs_rate = (tauBs * bphi/(16*np.pi) * fBs**2*mBs0/Lambda**2 * 1/a
               * (abs(gPS)**2 * (mBs0/(mb + ms))**2 )
              ) * lims
    
    return Bs_rate

def B_Bspsipsi(mpsi, Ci, Lambda = 1, is_ident = False):
    """Branching fraction for Bs -> psi psi."""

    CVL, CVR, CVLp, CVRp, CSL, CSR, CSLp, CSRp, CTL, CTR = Ci

    a = 1 + is_ident
    b = 1 - is_ident
    
    fVV = (CVR + CVL)/2 * b
    fAV = (CVR - CVL)/2 * b
    fVA = (CVRp + CVLp)/2 * a
    fAA = (CVRp - CVLp)/2 * a
    fSS = (CSR + CSL)/2 * a
    fPS = (CSR - CSL)/2 * a
    fSP = (CSRp + CSLp)/2 * a
    fPP = (CSRp - CSLp)/2 * a
    fTT = (CTL + CTR)/2 * b
    fTtT = (CTR - CTL)/2 * b
    
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
