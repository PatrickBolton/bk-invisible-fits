import numpy as np
from scipy.special import xlogy


def reduced_chisq(m, obs, n_unconstrained=None):
    """Reduced chi-squared (deviance per degree of freedom) of a binned Poisson fit.

    Uses the deviance: the fitted -2 ln L (m.fval, including the Gaussian
    constraint terms) minus -2 ln L of the saturated model (exp = obs), which
    removes the constant that the logLL_* functions drop.

    Each constrained nuisance parameter comes with its own Gaussian
    pseudo-measurement, so it adds one measurement and one parameter and the two
    cancel. Only unconstrained parameters use up degrees of freedom:
        dof = N_bins - n_unconstrained

    n_unconstrained: by default the number of free parameters among x0, x1, x2 of the
    fit (the SM signal strength and the NP couplings), which are the only unconstrained
    parameters of the lights likelihoods: 1 for an SM fit, 2 or 3 for NP fits. For a
    combined fit, pass the concatenated observations of all experiments.
    """
    if n_unconstrained is None:
        n_unconstrained = sum(not m.fixed[f"x{i}"] for i in range(3))
    obs = np.asarray(obs)
    saturated = 2 * np.sum(obs - xlogy(obs, obs))
    chisq = m.fval - saturated
    dof = len(obs) - n_unconstrained

    return chisq / dof
