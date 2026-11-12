"""Fits of the lights likelihoods: signal strength and coupling fits, profile likelihoods and scans.

Experiments are named "BelleII", "BaBar" and, for scans, "LEP" and "combined" (Belle II,
BaBar and LEP together). New physics is given as lights.models.Model objects, whose
couplings are scaled by the fitted multipliers x1, x2 (signal strengths x1^2, x2^2).

    min_logLL("BelleII")                       SM fit: signal strength mu_SM
    min_logLL("BelleII", model)                NP fit: coupling multiplier of model
    profile_logLL_SM("BelleII", mu)            profile likelihood of mu_SM
    mass_scan("BelleII", model, masses)        NP fit as a function of the mass
    profile_logLL_scan(model_1, model_2, ...)  Delta chi^2 grids of two coupling multipliers
"""

from typing import NamedTuple

import numpy as np
from iminuit import Minuit
from joblib import Parallel, delayed
from tqdm import tqdm

from lights.likelihoods import (
    LEP_mu,
    LEP_N_norm,
    LEP_sigma,
    NPAR_BaBar,
    NPAR_BelleII,
    Templates,
    grad_BaBar,
    grad_BelleII,
    grad_Combined,
    grad_LEP,
    logLL_BaBar,
    logLL_BelleII,
    logLL_Combined,
    logLL_LEP,
    np_templates,
)
from lights.load_data import BaBar, BelleII
from lights.models import B_Bs, check_model
from lights.signal_rates import NU_BaBar, NU_BelleII

# Likelihood, gradient and number of parameters of each fit: "combined" is Belle II + BaBar
# (the LEP term is added separately, see _lep_term)
_LOGLL = {"BelleII": logLL_BelleII, "BaBar": logLL_BaBar, "combined": logLL_Combined}
_GRAD = {"BelleII": grad_BelleII, "BaBar": grad_BaBar, "combined": grad_Combined}
_NPAR = {"BelleII": 3 + NPAR_BelleII, "BaBar": 3 + NPAR_BaBar, "combined": 3 + NPAR_BelleII + NPAR_BaBar}
_KIND_BY_NAME = {"logLL_BelleII": "BelleII", "logLL_BaBar": "BaBar"}


def _kind_of(experiment):
    """Experiment name ("BelleII" or "BaBar") of an experiment name or likelihood function.

    Functions are looked up by name, not object identity: joblib workers receive a pickled
    copy of the function, which is a different object from the module-level one.
    """
    if isinstance(experiment, str):
        name = experiment
    else:
        name = _KIND_BY_NAME.get(getattr(experiment, "__name__", None))
    if name not in ("BelleII", "BaBar"):
        raise ValueError(f'Unknown experiment {experiment!r}; expected "BelleII" or "BaBar"')
    return name


class FitResult(NamedTuple):
    """Result of min_logLL; also unpacks as (value, error, minuit, delta_chi2).

    value, error: the SM signal strength mu_SM (no NP model), or the coupling multiplier(s)
                  of the NP model(s) (a pair for two models)
    minuit:       the Minuit object of the fit (couplings in normalised units, see _normalise)
    delta_chi2:   -2 Delta ln L relative to the SM (mu_SM = 1, no new physics)
    """

    value: object
    error: object
    minuit: object
    delta_chi2: float


class _ScanGrids(NamedTuple):
    BelleII: np.ndarray
    BaBar: np.ndarray
    LEP: np.ndarray
    combined: np.ndarray


class ScanResult(_ScanGrids):
    """Delta chi^2 grids of profile_logLL_scan, indexed [x2, x1] (None if not computed); also unpacks as a 4-tuple.

    Besides the four grids (BelleII, BaBar, LEP, combined) it records the scan itself:
    x1, x2: the coupling multipliers of the grid; models: (model_1, model_2).
    """

    def __new__(cls, BelleII, BaBar, LEP, combined, x1=None, x2=None, models=(None, None)):
        self = super().__new__(cls, BelleII, BaBar, LEP, combined)
        self.x1, self.x2, self.models = x1, x2, models
        return self

    def __getnewargs__(self):  # pickling keeps the grids and models
        return (*self, self.x1, self.x2, self.models)


# Fitting
_Z_LIM = (-5.0, 5.0)  # unit-Gaussian background modes z (b + w z stays positive)
_P_LIM = (-1.0, None)  # multiplicative nuisance params entering as (1 + p)

N_START_EVENTS = 10.0  # expected NP signal events at the start of a fit with free couplings


def _limits(kind):
    """Parameter limits for a fit of the given kind ("BelleII", "BaBar", "combined").

    mu_SM >= 0; x1, x2 enter squared and are free; the input nuisances u are free (the
    log-normal response keeps the prediction positive); the background modes z are limited to
    +-5; multiplicative (1 + p) nuisance params are limited to p >= -1.
    """
    free = (None, None)
    belle = [free] * NU_BelleII + [_Z_LIM] * (3 * BelleII.Nbins) + [_P_LIM] * 3
    babar = [free] * NU_BaBar + [_P_LIM] * (BaBar.Nbins + 3)
    head = [(0.0, None), (None, None), (None, None)]
    return head + {"BelleII": belle, "BaBar": babar, "combined": belle + babar}[kind]


def _x_init(kind, head=(1.0, 0.0, 0.0)):
    "Initial parameter vector: the given (mu_SM, x1, x2) and all nuisance params at 0."
    x = np.zeros(_NPAR[kind])
    x[:3] = head
    return x


def _fit(kind, templates, x_init, fixed, extra=None, errors=False):
    """Run one Migrad fit and return the Minuit object.

    kind:      "BelleII", "BaBar" or "combined"
    templates: the NP Templates of the experiment; for "combined" a pair (Belle II, BaBar)
    fixed:     names of the fixed parameters among x0 (mu_SM), x1, x2
    extra:     optional (f, grad) added to the likelihood (e.g. LEP, see _lep_term)

    The analytic gradient is passed to Migrad, which runs with strategy 0: the minimum is
    found with far fewer calls because no numerical error matrix is built along the way. Set
    errors = True to run Hesse afterwards when parameter uncertainties are needed. A free NP
    coupling whose templates are all zero does not affect the likelihood and is fixed, to
    keep the Hesse matrix non-singular.
    """
    parts = (templates,) if isinstance(templates, Templates) else tuple(templates)
    args = tuple(np.ascontiguousarray(a, dtype=float) for t in parts for a in t)
    logLL, grad = _LOGLL[kind], _GRAD[kind]
    if extra is None:
        func = lambda x: logLL(x, *args)
        gfunc = lambda x: grad(x, *args)
    else:
        f_extra, g_extra = extra
        func = lambda x: logLL(x, *args) + f_extra(x)
        gfunc = lambda x: grad(x, *args) + g_extra(x)

    # Expected NP events at unit coupling, summed over the experiments, for x1 and x2
    n_events = [sum(float(np.sum(t.s_NP_1)) for t in parts), sum(float(np.sum(t.s_NP_2)) for t in parts)]
    flat = [f"x{k}" for k in (1, 2) if f"x{k}" not in fixed and n_events[k - 1] == 0]

    # The NP strengths are x1^2 and x2^2, so x = 0 is a stationary point where Migrad can stop
    # without moving. A free coupling starting at 0 starts at N_START_EVENTS expected events instead.
    x_init = np.array(x_init, dtype=float)
    for k in (1, 2):
        if f"x{k}" not in fixed and f"x{k}" not in flat and x_init[k] == 0:
            x_init[k] = np.sqrt(N_START_EVENTS / n_events[k - 1])

    m = Minuit(func, x_init, grad=gfunc)
    m.errordef = 1.0
    m.limits = _limits(kind)
    m.fixed[list(fixed) + flat] = True
    m.strategy = 0
    m.tol = 0.01  # EDM < 2e-5: Delta chi^2 accurate to ~1e-5 (the default 0.1 left up to ~1e-4)
    m.migrad(ncall=100000)
    if errors:
        m.hesse()
    return m


def _normalise(tpl, target=100.0):
    """Rescale NP Templates to `target` events at unit coupling, for a well-scaled fit.

    The coupling then enters as x_fit = x_true * sqrt(scale); Delta chi^2 is unaffected.
    Without this, templates that vanish in some limit (e.g. ~ m_X^2 as m_X -> 0) would need
    astronomically large couplings that Migrad cannot reach. The responses R = J/s are
    scale-free. Returns (templates, (scale_1, scale_2)).
    """
    scales = tuple(float(np.sum(s)) / target if np.sum(s) > 0 else 1.0 for s in (tpl.s_NP_1, tpl.s_NP_2))
    return tpl._replace(s_NP_1=tpl.s_NP_1 / scales[0], s_NP_2=tpl.s_NP_2 / scales[1]), scales


# Fits of one experiment
def min_logLL(experiment, model_1=None, model_2=None):
    """Minimise the log likelihood w.r.t. the SM signal strength or NP couplings and the nuisance parameters.

    experiment: "BelleII" or "BaBar" (or the likelihood function logLL_BelleII / logLL_BaBar)
    model_1/2:  NP Models whose couplings are scaled by the fitted multipliers x1, x2
                (signal strengths x1^2, x2^2). With no model, the SM signal strength mu_SM is
                fitted; with one or two models, mu_SM = 1 and x1 (and x2) are fitted.

    Returns a FitResult (value, error, minuit, delta_chi2): mu_SM, or the coupling
    multiplier(s) of the model(s), with delta_chi2 relative to the SM (mu_SM = 1, no NP). The
    NP templates are normalised internally (see _normalise), so the couplings inside the
    Minuit object are in normalised units; use value and error instead.
    """
    check_model(model_1, "model_1")
    check_model(model_2, "model_2")
    if model_1 is None and model_2 is not None:
        raise ValueError("Pass a single NP model as model_1")
    NP = (model_1 is not None) + (model_2 is not None)
    kind = _kind_of(experiment)

    tpl, scales = _normalise(np_templates(kind, model_1, model_2))
    unscale = 1 / np.sqrt(scales[0]), 1 / np.sqrt(scales[1])

    # Minimise w.r.t. nuisance params., keeping signal strength fixed to SM value (mu = 1)
    profile_min_SM = _fit(kind, tpl, _x_init(kind), ["x0", "x1", "x2"]).fval

    # Fit mu_SM (no NP), or the NP coupling(s) with mu_SM = 1, together with the nuisance params.
    fixed = {0: ["x1", "x2"], 1: ["x0", "x2"], 2: ["x0"]}[NP]
    m = _fit(kind, tpl, _x_init(kind), fixed, errors=True)
    delta_chi2 = m.fval - profile_min_SM
    if NP == 0:
        return FitResult(m.values[0], m.errors[0], m, delta_chi2)
    value = tuple(m.values[k] * unscale[k - 1] for k in range(1, NP + 1))
    error = tuple(m.errors[k] * unscale[k - 1] for k in range(1, NP + 1))
    if NP == 1:
        value, error = value[0], error[0]
    return FitResult(value, error, m, delta_chi2)


def profile_logLL_SM(experiment, mu):
    """Profile -2 Delta ln L of the SM signal strength mu (a number or an array of them).

    experiment: "BelleII" or "BaBar" (or the likelihood function logLL_BelleII / logLL_BaBar).
    At each mu the nuisance params. are fitted; the result is relative to the global minimum,
    where mu is fitted too (computed once).
    """
    kind = _kind_of(experiment)
    tpl = np_templates(kind)
    global_min = _fit(kind, tpl, _x_init(kind), ["x1", "x2"]).fval
    profile = [_fit(kind, tpl, _x_init(kind, (m, 0.0, 0.0)), ["x0", "x1", "x2"]).fval for m in np.ravel(mu)]
    return np.reshape(profile, np.shape(mu)) - global_min


def _delta_chi2(experiment, model):
    "Delta chi^2 of the fit of one NP model (runs inside a joblib worker)."
    return min_logLL(experiment, model).delta_chi2


def mass_scan(experiment, model, masses, *, n_jobs=-1):
    """Delta chi^2 of the NP fit (min_logLL) as a function of the mass, in parallel over the masses.

    At each mass m the coupling multiplier of model.with_mass(m) is fitted; the result is
    relative to the SM (mu_SM = 1, no NP), as in min_logLL.
    """
    kind = _kind_of(experiment)
    check_model(model, allow_none=False)
    jobs = (delayed(_delta_chi2)(kind, model.with_mass(m)) for m in tqdm(masses, desc=repr(model)))
    return np.array(Parallel(n_jobs=n_jobs)(jobs))


# Profile likelihood scans of two coupling multipliers
SCAN_EXPERIMENTS = ("BelleII", "BaBar", "LEP", "combined")  # the grids of a ScanResult


def _lep_term(Br_LEP):
    "LEP log likelihood and its gradient as functions of the parameter vector x."
    B1, B2 = float(Br_LEP[0]), float(Br_LEP[1])
    return (lambda x: logLL_LEP(x, B1, B2), lambda x: grad_LEP(x, B1, B2))


def _lep_min(Br_LEP):
    """Minimum of logLL_LEP over the couplings: the branching fraction N_norm * Br can take any
    value >= 0 if a model has a Bs -> invisible rate, else only 0."""
    Br = max(LEP_mu / LEP_N_norm, 0.0) if max(Br_LEP) > 0 else 0.0
    return (LEP_N_norm * Br - LEP_mu) ** 2 / LEP_sigma**2


def _profile_point(c1, c2, experiments, templates, global_mins, Br_LEP):
    """Profile -2 Delta ln L of each experiment at ONE (x1, x2) grid point. This runs inside a worker."""
    head = (1.0, c1, c2)
    lep = logLL_LEP(np.array(head), float(Br_LEP[0]), float(Br_LEP[1]))
    res = []
    for name in experiments:
        if name == "LEP":
            res.append(lep - global_mins[name])
            continue
        fval = _fit(name, templates[name], _x_init(name, head), ["x0", "x1", "x2"]).fval
        res.append(fval + (lep if name == "combined" else 0.0) - global_mins[name])
    return res


def profile_logLL_scan(model_1, model_2, lims_1, lims_2, *, pts=(100, 100), experiments=SCAN_EXPERIMENTS, n_jobs=-1):
    """Profile log likelihood on a grid of the coupling multipliers of two NP Models (parallel over the grid).

    The couplings of model_1 and model_2 (None: no second model) are scaled by x1 in lims_1
    and x2 in lims_2 (pts points each); the two models may be different channels or masses.
    experiments: which of "BelleII", "BaBar", "LEP" and "combined" (all three) to compute.

    Returns a ScanResult of the Delta chi^2 grids (BelleII, BaBar, LEP, combined), indexed
    [x2, x1], each relative to its own global minimum; grids not computed are None.
    """
    check_model(model_1, "model_1", allow_none=False)
    check_model(model_2, "model_2")
    unknown = set(experiments) - set(SCAN_EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unknown experiment(s) {sorted(unknown)}; choose from {SCAN_EXPERIMENTS}")
    experiments = [name for name in SCAN_EXPERIMENTS if name in experiments]

    # ---- Signal templates (computed once, shared by every grid point) ----
    tpl_BelleII = np_templates("BelleII", model_1, model_2)
    tpl_BaBar = np_templates("BaBar", model_1, model_2)
    templates = {"BelleII": tpl_BelleII, "BaBar": tpl_BaBar, "combined": (tpl_BelleII, tpl_BaBar)}
    Br_LEP = (B_Bs(model_1), B_Bs(model_2))

    # ---- Global minima (one fit per experiment, done serially) ----
    global_mins = {}
    for name in experiments:
        if name == "LEP":
            global_mins[name] = _lep_min(Br_LEP)
            continue
        # The profiled combined values include the LEP term, so the global minimum must too
        extra = _lep_term(Br_LEP) if name == "combined" else None
        global_mins[name] = _fit(name, templates[name], _x_init(name), ["x0"], extra=extra).fval

    # ---- Profiled scan, parallel over the (x1, x2) grid ----
    x1_vals = np.linspace(lims_1[0], lims_1[1], pts[0])
    x2_vals = np.linspace(lims_2[0], lims_2[1], pts[1])
    n1, n2 = len(x1_vals), len(x2_vals)

    jobs = (
        delayed(_profile_point)(c1, c2, experiments, templates, global_mins, Br_LEP) for c1 in x1_vals for c2 in x2_vals
    )
    gen = Parallel(n_jobs=n_jobs, return_as="generator")(jobs)
    results = np.array(list(tqdm(gen, total=n1 * n2, desc=f"{model_1.channel.name}, mX = {model_1.mX:.3g}")))

    # results is ordered (i over x1, j over x2): reshape, then transpose each grid to [j, i]
    results = results.reshape(n1, n2, len(experiments))
    grids = {name: results[:, :, i].T for i, name in enumerate(experiments)}
    return ScanResult(*(grids.get(name) for name in SCAN_EXPERIMENTS), x1_vals, x2_vals, (model_1, model_2))
