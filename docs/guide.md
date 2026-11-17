# bk-invisible-fits: user guide

How to use bk-invisible-fits (the Python package `lights`), how its fits work, and what is in the package and notebooks. See the
[README](../README.md) for an overview, the references and the installation.

## Usage

A model is one channel with its mass, its couplings (Wilson coefficients) and the scale
Lambda of the effective operators. Couplings not given are zero.

```python
import numpy as np
from lights.models import Model, CHANNELS
from lights.fits import (best_fit_model, fit_WET, fit_WET_vector, mass_scan, min_logLL,
                         profile_logLL_scan, profile_logLL_WET)

gS = Model("phi", mX = 2.1, CSL = 5e-9, CSR = 5e-9)                   # B -> K phi
fV = Model("psipsi", mX = 0.6, Lambda = 1e4, CVL = 1., CVR = 1.)       # B -> K psi psi-bar
fV_Majorana = Model("psipsi", mX = 0.6, Lambda = 1e4, is_ident = True, CVL = 1., CVR = 1.)

list(CHANNELS)          # "phi", "V", "phiphi", "psipsi", "VV", "PsiPsi", "nunu" (WET)
fV.dBdqsq(5.0)          # dB/dq^2 at q^2 = 5 GeV^2, rows K, K*, K*_L, K*_T
gS.B()                  # branching fractions of a two-body channel
fV.B_Bs()               # B(Bs -> psi psi-bar)

# SM fit: signal strength mu_SM, its error and -2 Delta ln L relative to mu_SM = 1
fit = min_logLL("BelleII")
fit.value, fit.error, fit.delta_chi2

# NP fit: the couplings of the model are scaled by a fitted multiplier
fit = min_logLL("BelleII", gS)
best, fit = best_fit_model("BelleII", gS)    # the model with its couplings at their best fit
best.couplings, best.dBdqsq(5.0)

# The NP fit as a function of the mass (in parallel)
delta_chi2 = mass_scan("BelleII", gS, [0.5, 1.0, 2.1])

# Experiments: "BelleII", "BelleII_public", "BaBar", and the combined fits "combined" and
# "combined_public" (Belle II, BaBar and LEP, with our own or the public Belle II likelihood)
fit = min_logLL("combined_public", gS)

# Likelihood on a grid of the multipliers of two models (in parallel): a slice, with mu_SM = 1
# and no other couplings (only the nuisance params. are fitted)
gP = Model("phi", mX = 2.1, CSL = 5e-9, CSR = -5e-9)
scan = profile_logLL_scan(gS, gP, (-8, 8), (-30, 30), pts = (100, 100))
scan.BelleII, scan.BaBar, scan.LEP, scan.combined     # Delta chi^2 grids, indexed [x2, x1]
scan_public = profile_logLL_scan(gS, gP, (-8, 8), (-30, 30), belle2 = "public")

# WET: the vector couplings interfere with the SM and are fitted with the SM included in them
from lights.constants import CVL_SM
wet = fit_WET_vector("combined", CVL = CVL_SM)    # C_VR, with C_VL at its SM value
wet.CVR, wet.errors, wet.delta_chi2, wet.model    # wet.model: the total rate, SM included
fit = fit_WET("combined")                         # all five: C_VL +- C_VR, C_SL +- C_SR, C_TL
grid = profile_logLL_WET("combined", ("V+", "T"), np.linspace(0, 24, 25), np.linspace(0, 10, 11))
```

`lights.plotting.plot_contours` draws the contours of a scan. Model objects are immutable;
`with_mass()` and `scaled()` return modified copies.

## Method

**Likelihoods.** Each experiment's likelihood is a binned Poisson likelihood with
Gaussian-constrained nuisance parameters for the backgrounds. The form factor coefficients
and efficiencies enter as shared nuisance parameters u ~ N(0, 1). Every signal template (SM
and NP) responds to them log-normally, so the theory uncertainties of the SM and NP signals
are correlated. The combined fit adds the Belle II, BaBar and LEP likelihoods. The
likelihoods and their analytic gradients are compiled with numba, and minimised with iminuit;
parameter errors come from the Hessian of the analytic gradient.

**Belle II's public likelihood.** The experiment `"BelleII_public"` (its inclusive-tag part)
and the combined fit `"combined_public"` use Belle II's released HistFactory model, evaluated
by a compiled reimplementation that agrees with pyhf. NP signals are obtained by kinematic
reweighting: Belle II's joint number density in true $q^2$ times the model's branching fraction in
each $q^2$ bin. Our own Belle II likelihood (`"BelleII"`; `belle2 = "lights"` in the scans) and the public one differ mainly in their high-$q^2$
backgrounds, so they prefer different NP shapes: the public likelihood favours scalar and
tensor couplings more than ours does.

**Interference with the SM.** NP rates add to the SM, so the fits of invisible particles and
of the scalar and tensor WET couplings assume no interference with it, which holds for these.
The vector WET couplings $C_{\rm VL}$ and $C_{\rm VR}$ do interfere. Their fits (`fit_WET_vector`, `fit_WET`)
include the SM in the couplings, using the fact that the rate is a sum of positive terms in
($C_{\rm VL} + C_{\rm VR})^2$, $(C_{\rm VL} − C_{\rm VR})^2$, $(C_{\rm SL} + C_{\rm SR})^2$, $(C_{\rm SL} − C_{\rm SR})$^2 and $C_{\rm TL}^2$. The other fits reject
these couplings. The WET couplings are normalised as in arXiv:2507.12393, with
$α_e(m_b) = 1/133$, so that the SM is $C_{\rm VL} = 6.60$.

**Slices and profiles.** `profile_logLL_scan` fixes mu_SM = 1 and has no couplings besides
the two scanned, so its grids are slices for a scenario with just those couplings.
`profile_logLL_strengths` and `profile_logLL_WET` fit the remaining strengths or couplings at
each grid point (profiles), so their grids also reflect what those do.

## Package layout

| Module | Contents |
| --- | --- |
| `lights.models` | `Model` and the channel registry `CHANNELS` |
| `lights.branching_fractions` | $B \to K^{(*)} X(\bar{X})$ and Bs → X X̄ rates for each channel, and the WET rate |
| `lights.form_factors` | $B \to K$ and $B \to K^*$ form factors (GRDV fit) |
| `lights.load_data` | the experimental inputs, grouped as `BelleII`, `BaBar` and `BelleII_public` |
| `lights.signal_rates` | binned signal predictions and their response to the theory inputs |
| `lights.likelihoods` | likelihoods and gradients (compiled with numba), NP signal templates |
| `lights.fits` | fits, profile likelihoods, mass scans, coupling scans and the WET fits |
| `lights.histfactory` | Belle II's public HistFactory (pyhf) likelihood as arrays for the compiled likelihood |
| `lights.statistics` | reduced chi-squared of a fit |
| `lights.plotting` | figure style and contour plots |

## Notebooks

| Notebook | Contents |
| --- | --- |
| `01_BR_plots` | branching fraction spectra of each channel and of the WET couplings |
| `02_SM_fit` | the inputs; the SM fit to Belle II and the profile likelihood of mu_SM, compared with Belle II's published result and public likelihood |
| `03_NP_mass` | Belle II fits as a function of the new particle's mass, with both Belle II likelihoods |
| `04_NP_coupling` | coupling scans of each scenario (Belle II, BaBar, LEP and combined); the WET fit of Belle II's Fig. 3 (arXiv:2507.12393), and of all five WET couplings with BaBar |
| `05_fit_predict` | normalised spectra of each scenario; SM + NP spectra at the best-fit couplings (invisible particles and WET), with coupling and form factor uncertainties |

The notebooks take from seconds (01, 02, 05) to a few minutes (03, 04) to run, and save
their figures to `plots/`.

## Scripts

- `scripts/check_gradients.py` checks the analytic likelihood gradients against finite differences.
- `scripts/validate_theory_cov.py` compares the linear theory-uncertainty model with full toys.

Run them with `uv run python scripts/<name>.py`.
