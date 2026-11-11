"""Physical constants used throughout lights. Masses in GeV."""

import numpy as np

Lumi = 362.
SigmaUps4S = 387./362. * 1e6
NB = Lumi * SigmaUps4S # Number of B mesons

# Meson masses
mB = 5.27934 # B+ meson mass
mBs0 = 5.36689 # Bs meson mass
mK = 0.493677 # K+ meson mass
mKs = 0.89166 # K*(892)+ meson mass

mb = 4.87 # b quark mass
ms = 0.07966 # s quark mass

# Meson lifetimes
tauB = (1.638e-12)/(6.582e-16)/(1e-9) # B+- meson lifetime
tauB0 = (1.519e-12)/(6.582e-16)/(1e-9)  # B0 meson lifetime
tauBs = (1.509e-12)/(6.582e-16)/(1e-9)  # Bs meson lifetime
fBs = 0.2303 # Bs meson decay constant (GeV)

# SM neutrino coupling
GF = 1.1663787e-5 # GeV^-2
ALPHA = 1/127.952 # Fine structure constant
LAMBDAt = 0.04185 # Vtb Vts^* in SM
Xt = 1.469 # NLO QCD and two-loop electroweak corrections
SWsq = 0.23121 # Sine of weak mixing angle squared
CSM = 4*GF*LAMBDAt/np.sqrt(2) * Xt/SWsq * ALPHA/(2*np.pi)
