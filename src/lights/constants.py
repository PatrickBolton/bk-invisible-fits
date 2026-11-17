"""Physical constants used throughout lights. Masses in GeV."""

import numpy as np

Lumi = 362.0
SigmaUps4S = 387.0 / 362.0 * 1e6
# Number of BB-bar pairs at Belle II. The number of signal-side B+ mesons would be
# 2 * f(+-) * NB; that factor is assumed to be absorbed in the signal efficiencies.
NB = Lumi * SigmaUps4S

# Meson masses
mB = 5.27934  # B+ meson mass
mBs0 = 5.36689  # Bs meson mass
mK = 0.493677  # K+ meson mass
mKs = 0.89555  # K*(892)0 meson mass (the K* rates are for B0 -> K*0, with tauB0)

# MS-bar quark masses at mu_b ~ 4.2 GeV, the scale of the Wilson coefficients (as in EOS); they enter the
# scalar and pseudoscalar currents through the equations of motion
mb = 4.18  # b quark mass, m_b(m_b) (PDG)
ms = 0.07966  # s quark mass

# Meson lifetimes
tauB = (1.638e-12) / (6.582e-16) / (1e-9)  # B+- meson lifetime
tauB0 = (1.519e-12) / (6.582e-16) / (1e-9)  # B0 meson lifetime
tauBs = (1.509e-12) / (6.582e-16) / (1e-9)  # Bs meson lifetime
fBs = 0.2303  # Bs meson decay constant (GeV)

# SM neutrino coupling
GF = 1.1663787e-5  # GeV^-2
ALPHA = 1 / 127.952  # Fine structure constant
ALPHA_MB = 1 / 133.0  # alpha_e(m_b): the normalisation of the WET coefficients of B -> K nu nu-bar (EOS)
LAMBDAt = 0.04185  # Vtb Vts^* in SM
Xt = 1.469  # NLO QCD and two-loop electroweak corrections
SWsq = 0.23121  # Sine of weak mixing angle squared
CSM = 4 * GF * LAMBDAt / np.sqrt(2) * Xt / SWsq * ALPHA / (2 * np.pi)
