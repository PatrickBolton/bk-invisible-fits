"""Fits of the lights likelihoods: signal strength and coupling fits, profile likelihoods and scans.

Experiments (EXPERIMENTS):
    "BelleII"          the lights Belle II likelihood (ITA)
    "BelleII_public"   Belle II's public likelihood (ITA, arXiv:2507.12393)
    "BaBar"            BaBar
    "combined"         lights Belle II + BaBar + LEP
    "combined_public"  public Belle II + BaBar + LEP
New physics is given as lights.models.Model objects, whose couplings are scaled by the
fitted multipliers x1, x2 (signal strengths x1^2, x2^2).

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
    NH_pub,
    NM_pub,
    NPAR_BaBar,
    NPAR_BelleII,
    NPAR_BelleII_public,
    Templates,
    grad_BaBar,
    grad_BelleII,
    grad_BelleII_public,
    grad_Combined,
    grad_Combined_public,
    grad_LEP,
    logLL_BaBar,
    logLL_BelleII,
    logLL_BelleII_public,
    logLL_Combined,
    logLL_Combined_public,
    logLL_LEP,
    np_templates,
)
from lights.load_data import BaBar, BelleII
from lights.models import B_Bs, check_model
from lights.signal_rates import NU_BaBar, NU_BelleII, NU_BelleII_public

# Each fit: the B -> K(*) likelihoods it adds (their NP templates, in this order), its compiled
# likelihood and gradient, and its number of parameters. The combined fits also include LEP,
# added separately (see _lep_term).
_PARTS = {
    "BelleII": ("BelleII",),
    "BelleII_public": ("BelleII_public",),
    "BaBar": ("BaBar",),
    "combined": ("BelleII", "BaBar"),
    "combined_public": ("BelleII_public", "BaBar"),
}
_LOGLL = {
    "BelleII": logLL_BelleII,
    "BelleII_public": logLL_BelleII_public,
    "BaBar": logLL_BaBar,
    "combined": logLL_Combined,
    "combined_public": logLL_Combined_public,
}
_GRAD = {
    "BelleII": grad_BelleII,
    "BelleII_public": grad_BelleII_public,
    "BaBar": grad_BaBar,
    "combined": grad_Combined,
    "combined_public": grad_Combined_public,
}
_NPAR = {
    "BelleII": 3 + NPAR_BelleII,
    "BelleII_public": 3 + NPAR_BelleII_public,
    "BaBar": 3 + NPAR_BaBar,
    "combined": 3 + NPAR_BelleII + NPAR_BaBar,
    "combined_public": 3 + NPAR_BelleII_public + NPAR_BaBar,
}
_WITH_LEP = ("combined", "combined_public")
EXPERIMENTS = tuple(_PARTS)  # the experiments of min_logLL, profile_logLL_SM and mass_scan


def _check_experiment(experiment):
    "Raise a ValueError unless experiment is one of EXPERIMENTS."
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; expected one of {EXPERIMENTS}")


def _templates(experiment, model_1=None, model_2=None):
    "NP Templates of a fit: one Templates, or a tuple of them for the combined fits (see _PARTS)."
    parts = tuple(np_templates(e, model_1, model_2) for e in _PARTS[experiment])
    return parts[0] if len(parts) == 1 else parts


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
    x1, x2: the coupling multipliers of the grid; models: (model_1, model_2); belle2: the Belle
    II likelihood of the BelleII and combined grids ("lights" or "public").
    """

    def __new__(cls, BelleII, BaBar, LEP, combined, x1=None, x2=None, models=(None, None), belle2="lights"):
        self = super().__new__(cls, BelleII, BaBar, LEP, combined)
        self.x1, self.x2, self.models, self.belle2 = x1, x2, models, belle2
        return self

    def __getnewargs__(self):  # pickling keeps the grids, models and likelihood choice
        return (*self, self.x1, self.x2, self.models, self.belle2)


# Fitting
_Z_LIM = (-5.0, 5.0)  # unit-Gaussian background modes z (b + w z stays positive)
_P_LIM = (-1.0, None)  # multiplicative nuisance params entering as (1 + p)

N_START_EVENTS = 10.0  # expected NP signal events at the start of a fit with free couplings


def _limits(kind):
    """Parameter limits for a fit of the given kind (one of EXPERIMENTS).

    mu_SM >= 0; x1, x2 enter squared and are free; the input nuisances u are free (the
    log-normal response keeps the prediction positive); the background modes z and the
    histosys and normsys parameters of the public likelihood are limited to +-5 (as in pyhf);
    multiplicative (1 + p) nuisance params are limited to p >= -1. (The staterror factors of
    the public likelihood are profiled analytically, within their bound gamma >= 0.)
    """
    free = (None, None)
    belle = [free] * NU_BelleII + [_Z_LIM] * (3 * BelleII.Nbins) + [_P_LIM] * 3
    babar = [free] * NU_BaBar + [_P_LIM] * (BaBar.Nbins + 3)
    public = [free] * NU_BelleII_public + [_Z_LIM] * (NH_pub + NM_pub)  # staterror factors are profiled
    head = [(0.0, None), (None, None), (None, None)]
    blocks = {"BelleII": belle, "BelleII_public": public, "BaBar": babar}
    return head + [lim for part in _PARTS[kind] for lim in blocks[part]]


def _x_init(kind, head=(1.0, 0.0, 0.0)):
    "Initial parameter vector: the given (mu_SM, x1, x2) and all nuisance params at 0."
    x = np.zeros(_NPAR[kind])
    x[:3] = head
    return x


def _fit(kind, templates, x_init, fixed, extra=None):
    """Run one Migrad fit and return the Minuit object.

    kind:      one of EXPERIMENTS
    templates: the NP Templates of the experiment; for the combined fits a pair (Belle II, BaBar)
    fixed:     names of the fixed parameters among x0 (mu_SM), x1, x2
    extra:     optional (f, grad) added to the likelihood (e.g. LEP, see _lep_term)

    The analytic gradient is passed to Migrad, which runs with strategy 0: the minimum is
    found with far fewer calls because no numerical error matrix is built along the way (see
    _errors for the parameter errors). A free NP coupling whose templates are all zero does not affect the likelihood and is fixed, to
    keep the Hessian non-singular (unless an extra term, such as LEP, may depend on it).
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
    flat = [f"x{k}" for k in (1, 2) if f"x{k}" not in fixed and n_events[k - 1] == 0 and extra is None]

    # The NP strengths are x1^2 and x2^2, so x = 0 is a stationary point where Migrad can stop
    # without moving. A free coupling starting at 0 starts at N_START_EVENTS expected events instead.
    x_init = np.array(x_init, dtype=float)
    for k in (1, 2):
        if f"x{k}" not in fixed and f"x{k}" not in flat and x_init[k] == 0:
            x_init[k] = np.sqrt(N_START_EVENTS / n_events[k - 1]) if n_events[k - 1] > 0 else 1.0

    m = Minuit(func, x_init, grad=gfunc)
    m.errordef = 1.0
    m.limits = _limits(kind)
    m.fixed[list(fixed) + flat] = True
    m.strategy = 0
    m.tol = 0.01  # EDM < 2e-5: Delta chi^2 accurate to ~1e-5 (the default 0.1 left up to ~1e-4)
    m.migrad(ncall=100000)
    return m


def _errors(kind, templates, m, extra=None, h=1e-4):
    """Parameter errors at the minimum of a fit (m, from _fit), from the Hessian of the analytic gradient.

    The Hessian is computed by central differences of the gradient (exact up to O(h^2)), and
    the covariance of -2 ln L is 2 H^-1; fixed parameters get error 0. This replaces Minuit's
    Hesse, whose numerical second derivatives depend on the step sizes left by Migrad (with the
    public Belle II likelihood it underestimated the error of mu_SM by 4.5%) and are distorted
    for parameters near a limit (BaBar's mu_SM: 4.1 instead of 7.2). Near a limit the error is
    the parabolic one, ignoring the limit.
    """
    parts = (templates,) if isinstance(templates, Templates) else tuple(templates)
    args = tuple(np.ascontiguousarray(a, dtype=float) for t in parts for a in t)
    grad = _GRAD[kind]
    g = (lambda x: grad(x, *args)) if extra is None else (lambda x: grad(x, *args) + extra[1](x))
    x = np.array(m.values)
    free = [i for i in range(len(x)) if not m.fixed[i]]
    H = np.empty((len(free), len(free)))
    for a, i in enumerate(free):
        dx = np.zeros(len(x))
        dx[i] = h
        H[a] = ((g(x + dx) - g(x - dx)) / (2 * h))[free]
    errors = np.zeros(len(x))
    errors[free] = np.sqrt(np.diag(2 * np.linalg.inv(0.5 * (H + H.T))))
    return errors


def _normalise(tpl, target=100.0):
    """Rescale NP Templates to `target` events at unit coupling, for a well-scaled fit.

    The coupling then enters as x_fit = x_true * sqrt(scale); Delta chi^2 is unaffected.
    Without this, templates that vanish in some limit (e.g. ~ m_X^2 as m_X -> 0) would need
    astronomically large couplings that Migrad cannot reach. The responses R = J/s are
    scale-free. For a tuple of Templates (a combined fit) the scale is common to all of them,
    from their summed events, since the coupling is shared. Returns (templates, (scale_1, scale_2)).
    """
    parts = (tpl,) if isinstance(tpl, Templates) else tuple(tpl)
    totals = (sum(float(np.sum(t.s_NP_1)) for t in parts), sum(float(np.sum(t.s_NP_2)) for t in parts))
    scales = tuple(total / target if total > 0 else 1.0 for total in totals)
    out = tuple(t._replace(s_NP_1=t.s_NP_1 / scales[0], s_NP_2=t.s_NP_2 / scales[1]) for t in parts)
    return (out[0] if isinstance(tpl, Templates) else out), scales


# Fits
def min_logLL(experiment, model_1=None, model_2=None, *, errors=True):
    """Minimise the log likelihood w.r.t. the SM signal strength or NP couplings and the nuisance parameters.

    experiment: one of EXPERIMENTS ("BelleII", "BelleII_public", "BaBar", or the combined fits
                "combined" and "combined_public", which include BaBar and LEP)
    model_1/2:  NP Models whose couplings are scaled by the fitted multipliers x1, x2
                (signal strengths x1^2, x2^2). With no model, the SM signal strength mu_SM is
                fitted; with one or two models, mu_SM = 1 and x1 (and x2) are fitted.

    Returns a FitResult (value, error, minuit, delta_chi2): mu_SM, or the coupling
    multiplier(s) of the model(s), with delta_chi2 relative to the SM (mu_SM = 1, no NP). The
    NP templates are normalised internally (see _normalise), so the couplings inside the
    Minuit object are in normalised units; use value and error instead.

    errors: compute the parameter errors (from the Hessian of the analytic gradient, see
    _errors; the errors stored in the Minuit object are not these). Without them (error is
    None) the fit is a little faster.
    """
    check_model(model_1, "model_1")
    check_model(model_2, "model_2")
    if model_1 is None and model_2 is not None:
        raise ValueError("Pass a single NP model as model_1")
    _check_experiment(experiment)
    NP = (model_1 is not None) + (model_2 is not None)

    tpl, scales = _normalise(_templates(experiment, model_1, model_2))
    unscale = 1 / np.sqrt(scales[0]), 1 / np.sqrt(scales[1])
    # LEP, in the normalised couplings: x_true^2 Br = x_fit^2 (Br / scale)
    extra = None
    if experiment in _WITH_LEP:
        extra = _lep_term((B_Bs(model_1) / scales[0], B_Bs(model_2) / scales[1]))

    # Minimise w.r.t. nuisance params., keeping signal strength fixed to SM value (mu = 1)
    profile_min_SM = _fit(experiment, tpl, _x_init(experiment), ["x0", "x1", "x2"], extra=extra).fval

    # Fit mu_SM (no NP), or the NP coupling(s) with mu_SM = 1, together with the nuisance params.
    fixed = {0: ["x1", "x2"], 1: ["x0", "x2"], 2: ["x0"]}[NP]
    m = _fit(experiment, tpl, _x_init(experiment), fixed, extra=extra)
    delta_chi2 = m.fval - profile_min_SM
    err = _errors(experiment, tpl, m, extra) if errors else [None] * 3
    if NP == 0:
        return FitResult(m.values[0], err[0], m, delta_chi2)
    value = tuple(m.values[k] * unscale[k - 1] for k in range(1, NP + 1))
    error = tuple(err[k] * unscale[k - 1] if errors else None for k in range(1, NP + 1))
    if NP == 1:
        value, error = value[0], error[0]
    return FitResult(value, error, m, delta_chi2)


def profile_logLL_SM(experiment, mu):
    """Profile -2 Delta ln L of the SM signal strength mu (a number or an array of them).

    experiment: one of EXPERIMENTS (LEP does not depend on mu). At each mu the nuisance
    params. are fitted; the result is relative to the global minimum, where mu is fitted too
    (computed once).
    """
    _check_experiment(experiment)
    tpl = _templates(experiment)
    global_min = _fit(experiment, tpl, _x_init(experiment), ["x1", "x2"]).fval
    profile = [_fit(experiment, tpl, _x_init(experiment, (m, 0.0, 0.0)), ["x0", "x1", "x2"]).fval for m in np.ravel(mu)]
    return np.reshape(profile, np.shape(mu)) - global_min


def _delta_chi2(experiment, model):
    "Delta chi^2 of the fit of one NP model (runs inside a joblib worker)."
    return min_logLL(experiment, model, errors=False).delta_chi2


def mass_scan(experiment, model, masses, *, n_jobs=-1):
    """Delta chi^2 of the NP fit (min_logLL) as a function of the mass, in parallel over the masses.

    experiment: one of EXPERIMENTS. At each mass m the coupling multiplier of
    model.with_mass(m) is fitted; the result is relative to the SM (mu_SM = 1, no NP), as in
    min_logLL.
    """
    _check_experiment(experiment)
    check_model(model, allow_none=False)
    jobs = (delayed(_delta_chi2)(experiment, model.with_mass(m)) for m in tqdm(masses, desc=repr(model)))
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


def _profile_point(c1, c2, fits, global_mins, Br_LEP):
    """Profile -2 Delta ln L of each grid at ONE (x1, x2) grid point. This runs inside a worker.

    fits: {grid name: (kind, templates)}; the LEP grid has kind None.
    """
    head = (1.0, c1, c2)
    lep = logLL_LEP(np.array(head), float(Br_LEP[0]), float(Br_LEP[1]))
    res = []
    for name, (kind, tpl) in fits.items():
        if kind is None:
            res.append(lep - global_mins[name])
            continue
        fval = _fit(kind, tpl, _x_init(kind, head), ["x0", "x1", "x2"]).fval
        res.append(fval + (lep if kind in _WITH_LEP else 0.0) - global_mins[name])
    return res


def profile_logLL_scan(
    model_1, model_2, lims_1, lims_2, *, pts=(100, 100), experiments=SCAN_EXPERIMENTS, belle2="lights", n_jobs=-1
):
    """Profile log likelihood on a grid of the coupling multipliers of two NP Models (parallel over the grid).

    The couplings of model_1 and model_2 (None: no second model) are scaled by x1 in lims_1
    and x2 in lims_2 (pts points each); the two models may be different channels or masses.
    experiments: which of "BelleII", "BaBar", "LEP" and "combined" (all three) to compute.
    belle2: the Belle II likelihood of the BelleII and combined grids, "lights" (the lights
    likelihood) or "public" (Belle II's public likelihood).

    Returns a ScanResult of the Delta chi^2 grids (BelleII, BaBar, LEP, combined), indexed
    [x2, x1], each relative to its own global minimum; grids not computed are None.
    """
    check_model(model_1, "model_1", allow_none=False)
    check_model(model_2, "model_2")
    unknown = set(experiments) - set(SCAN_EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unknown experiment(s) {sorted(unknown)}; choose from {SCAN_EXPERIMENTS}")
    if belle2 not in ("lights", "public"):
        raise ValueError(f'belle2 must be "lights" or "public", not {belle2!r}')
    suffix = "" if belle2 == "lights" else "_public"
    kinds = {"BelleII": "BelleII" + suffix, "BaBar": "BaBar", "LEP": None, "combined": "combined" + suffix}
    kinds = {name: kind for name, kind in kinds.items() if name in experiments}

    # ---- Signal templates (computed once per experiment, shared by every grid point) ----
    needed = {part for kind in kinds.values() if kind for part in _PARTS[kind]}
    tpl = {e: np_templates(e, model_1, model_2) for e in needed}
    fits = {}
    for name, kind in kinds.items():
        parts = tuple(tpl[e] for e in _PARTS[kind]) if kind else ()
        fits[name] = (kind, parts[0] if len(parts) == 1 else parts)
    Br_LEP = (B_Bs(model_1), B_Bs(model_2))

    # ---- Global minima (one fit per grid, done serially) ----
    global_mins = {}
    for name, (kind, t) in fits.items():
        if kind is None:
            global_mins[name] = _lep_min(Br_LEP)
            continue
        # The profiled combined values include the LEP term, so the global minimum must too
        extra = _lep_term(Br_LEP) if kind in _WITH_LEP else None
        global_mins[name] = _fit(kind, t, _x_init(kind), ["x0"], extra=extra).fval

    # ---- Profiled scan, parallel over the (x1, x2) grid ----
    x1_vals = np.linspace(lims_1[0], lims_1[1], pts[0])
    x2_vals = np.linspace(lims_2[0], lims_2[1], pts[1])
    n1, n2 = len(x1_vals), len(x2_vals)

    jobs = (delayed(_profile_point)(c1, c2, fits, global_mins, Br_LEP) for c1 in x1_vals for c2 in x2_vals)
    gen = Parallel(n_jobs=n_jobs, return_as="generator")(jobs)
    results = np.array(list(tqdm(gen, total=n1 * n2, desc=f"{model_1.channel.name}, mX = {model_1.mX:.3g}")))

    # results is ordered (i over x1, j over x2): reshape, then transpose each grid to [j, i]
    results = results.reshape(n1, n2, len(fits))
    grids = {name: results[:, :, i].T for i, name in enumerate(fits)}
    return ScanResult(
        *(grids.get(name) for name in SCAN_EXPERIMENTS), x1_vals, x2_vals, (model_1, model_2), belle2=belle2
    )
