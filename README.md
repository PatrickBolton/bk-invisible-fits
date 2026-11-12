# lights

Fits of new-physics models with light invisible particles to the Belle II B+ → K+ E_miss
(inclusive tag) and BaBar B0 → K*0 E_miss binned data, together with the LEP constraint on
Bs → invisible.

The new physics is a light scalar, vector, fermion or spin-3/2 particle produced singly
(B → K X) or in pairs (B → K X X̄) through the effective operators in
[docs/operators.md](docs/operators.md). The code computes the branching fractions and the
binned signal predictions of each experiment, including the Belle II q² smearing and the
form factor and efficiency uncertainties, and fits them with binned Poisson likelihoods.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.14:

```sh
uv sync            # creates .venv with the package (editable) and the dev tools
uv run pytest      # runs the tests
```

The experimental inputs and form factors are in `data/`. To use data stored elsewhere, set
the environment variable `LIGHTS_DATA_DIR` to that directory.

## Usage

A model is one channel with its mass, its couplings (Wilson coefficients) and the scale
Lambda of the effective operators. Couplings not given are zero.

```python
from lights.models import Model, CHANNELS
from lights.fits import mass_scan, min_logLL, profile_logLL_scan

gS = Model("phi", mX = 2.1, CSL = 5e-9, CSR = 5e-9)                   # B -> K phi
fV = Model("psipsi", mX = 0.6, Lambda = 1e4, CVL = 1., CVR = 1.)       # B -> K psi psi-bar
fV_Majorana = Model("psipsi", mX = 0.6, Lambda = 1e4, is_ident = True, CVL = 1., CVR = 1.)

list(CHANNELS)          # "phi", "V", "phiphi", "psipsi", "VV", "PsiPsi"
fV.dBdqsq(5.0)          # dB/dq^2 at q^2 = 5 GeV^2, rows K, K*, K*_L, K*_T
gS.B()                  # branching fractions of a two-body channel
fV.B_Bs()               # B(Bs -> psi psi-bar)

# SM fit: signal strength mu_SM, its error and -2 Delta ln L relative to mu_SM = 1
fit = min_logLL("BelleII")
fit.value, fit.error, fit.delta_chi2

# NP fit: the couplings of the model are scaled by a fitted multiplier
fit = min_logLL("BelleII", gS)

# The NP fit as a function of the mass (in parallel)
delta_chi2 = mass_scan("BelleII", gS, [0.5, 1.0, 2.1])

# Profile likelihood on a grid of the multipliers of two models (in parallel)
gP = Model("phi", mX = 2.1, CSL = 5e-9, CSR = -5e-9)
scan = profile_logLL_scan(gS, gP, (-8, 8), (-30, 30), pts = (100, 100))
scan.BelleII, scan.BaBar, scan.LEP, scan.combined     # Delta chi^2 grids, indexed [x2, x1]
```

`lights.plotting.plot_contours` draws the contours of a scan. Model objects are immutable;
`with_mass()` and `scaled()` return modified copies.

## Package layout

| Module | Contents |
| --- | --- |
| `lights.models` | `Model` and the channel registry `CHANNELS` |
| `lights.branching_fractions` | B → K(*) X(X̄) and Bs → X X̄ rates for each channel |
| `lights.form_factors` | B → K and B → K* form factors (GRDV fit) |
| `lights.load_data` | the experimental inputs, grouped as `BelleII` and `BaBar` |
| `lights.signal_rates` | binned signal predictions and their response to the theory inputs |
| `lights.likelihoods` | likelihoods and gradients (compiled with numba), NP signal templates |
| `lights.fits` | fits, profile likelihoods, mass scans and coupling scans |
| `lights.statistics` | reduced chi-squared of a fit |
| `lights.plotting` | figure style and contour plots |

### Likelihoods

Each experiment's likelihood is a binned Poisson likelihood with Gaussian-constrained
nuisance parameters for the backgrounds. The form factor coefficients and efficiencies enter
as shared nuisance parameters u ~ N(0, 1), to which every signal template (SM and NP)
responds log-normally, so the theory uncertainties of the SM and NP signals are correlated.
The combined fit adds the Belle II, BaBar and LEP likelihoods. Fits use iminuit with analytic
gradients (`lights.fits`); the likelihoods are compiled with numba.

## Notebooks

| Notebook | Contents |
| --- | --- |
| `01_BR_plots` | branching fraction spectra of each channel |
| `02_SM_fit` | inputs, SM fit to Belle II and the profile likelihood of mu_SM |
| `03_NP_mass` | Belle II fits as a function of the new particle's mass |
| `04_NP_coupling` | coupling scans: Belle II, BaBar, LEP and combined contours |
| `05_fit_predict` | normalised SM and NP spectra with form factor uncertainties |

Figures are saved to `plots/`.

## Scripts

- `scripts/check_gradients.py` checks the analytic likelihood gradients against finite differences.
- `scripts/validate_theory_cov.py` compares the linear theory-uncertainty model with full toys.

Run them with `uv run python scripts/<name>.py`.
