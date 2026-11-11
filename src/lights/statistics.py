import numpy as np
import numpy.random as npr
from numpy.linalg import multi_dot, inv
from scipy.special import xlogy

def reduced_chisq(m, obs, n_unconstrained = 1):
    """Reduced chi-squared (deviance per degree of freedom) of a binned Poisson fit.

    Uses the deviance: the fitted -2 ln L (m.fval, including the Gaussian
    constraint terms) minus -2 ln L of the saturated model (exp = obs), which
    removes the constant that the logLL_* functions drop.

    Each constrained nuisance parameter comes with its own Gaussian
    pseudo-measurement, so it adds one measurement and one parameter and the two
    cancel. Only unconstrained parameters (e.g. the signal strength) use up
    degrees of freedom:
        dof = N_bins - n_unconstrained
    """
    obs = np.asarray(obs)
    saturated = 2 * np.sum(obs - xlogy(obs, obs))
    chisq = m.fval - saturated
    dof = len(obs) - n_unconstrained

    return chisq/dof