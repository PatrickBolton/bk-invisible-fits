"""Check the analytic likelihood gradients against central finite differences.

uv run python scripts/check_gradients.py
"""

import numpy as np

from lights.likelihoods import (
    NPAR_BaBar,
    NPAR_BelleII,
    NPAR_BelleII_public,
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
from lights.models import Model

rng = np.random.default_rng(1)
m1 = Model("phi", 1.0, CSL=1e-8, CSR=3e-9)
m2 = Model("phiphi", 1.0, Lambda=1e4, CSL=1e-3)
tB = np_templates("BelleII", m1, m2)
tb = np_templates("BaBar", m1, m2)
tP = np_templates("BelleII_public", m1, m2)


def random_x(npar):
    x = np.concatenate(([1.3, 0.7, -0.4], 0.3 * rng.normal(size=npar)))
    return x


def check(name, f, g, x, h=1e-6):
    ga = g(x)
    gn = np.array([(f(x + h * np.eye(len(x))[i]) - f(x - h * np.eye(len(x))[i])) / (2 * h) for i in range(len(x))])
    err = np.max(np.abs(ga - gn) / (1 + np.abs(gn)))
    print(f"{name:15s}: max relative error {err:.1e}  ({'OK' if err < 1e-4 else 'FAIL'})")
    return err


errs = []
for _ in range(3):
    x = random_x(NPAR_BelleII)
    x[3 + NPAR_BelleII - 3 :] *= 0.05  # background normalisations ~ 2%
    errs.append(check("BelleII", lambda y: logLL_BelleII(y, *tB), lambda y: grad_BelleII(y, *tB), x))
    x = random_x(NPAR_BaBar)
    errs.append(check("BaBar", lambda y: logLL_BaBar(y, *tb), lambda y: grad_BaBar(y, *tb), x))
    x = random_x(NPAR_BelleII + NPAR_BaBar)
    x[3 + NPAR_BelleII - 3 : 3 + NPAR_BelleII] *= 0.05
    errs.append(check("Combined", lambda y: logLL_Combined(y, *tB, *tb), lambda y: grad_Combined(y, *tB, *tb), x))
    x = random_x(NPAR_BelleII_public)
    errs.append(
        check(
            "BelleII_public", lambda y: logLL_BelleII_public(y, *tP), lambda y: grad_BelleII_public(y, *tP), x, h=1e-5
        )
    )
    x = random_x(NPAR_BelleII_public + NPAR_BaBar)
    errs.append(
        check(
            "Combined_public",
            lambda y: logLL_Combined_public(y, *tP, *tb),
            lambda y: grad_Combined_public(y, *tP, *tb),
            x,
            h=1e-5,
        )
    )
    errs.append(check("LEP", lambda y: logLL_LEP(y, 3e-5, 1e-4), lambda y: grad_LEP(y, 3e-5, 1e-4), x[:5]))
print("ALL OK" if max(errs) < 1e-4 else "SOME CHECKS FAILED")
