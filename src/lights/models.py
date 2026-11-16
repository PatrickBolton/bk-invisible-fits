"""New-physics models: the invisible-particle channels and their couplings.

Each channel (a final state X or X X-bar, see docs/operators.md) is defined once, in
CHANNELS: its coupling names, whether it is a two- or three-body decay, and the functions
for its branching fractions. A Model is one channel with its mass, couplings, and the scale
Lambda of the effective operators:

    gS = Model("phi", mX = 2.1, CSL = 5e-9, CSR = 5e-9)
    fV = Model("psipsi", mX = 0.6, Lambda = 1e4, CVL = 0.5, CVR = 0.5)
    fV_Majorana = Model("psipsi", mX = 0.6, Lambda = 1e4, is_ident = True, CVL = 0.5, CVR = 0.5)

Couplings not given are zero. Models are immutable; with_mass() and scaled() return
modified copies.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .branching_fractions import (  # K, KS, KS_L, KS_T: row indices, re-exported
    B_BKV,
    KS,  # noqa: F401
    KS_L,  # noqa: F401
    KS_T,  # noqa: F401
    B_BKphi,
    B_Bsphiphi,
    B_Bspsipsi,
    K,  # noqa: F401
    dBdqsq_BKphiphi,
    dBdqsq_BKPsiPsi,
    dBdqsq_BKpsipsi,
    dBdqsq_BKVV,
)
from .form_factors import A_VALS_GRDV


@dataclass(frozen=True)
class Channel:
    """One invisible final state and the functions for its rates.

    rate:     two-body:   rate(mX, C, Lambda, a_vals) -> B(B -> K(*) X), shape (4,)
              three-body: rate(qsq, mX, C, Lambda, is_ident, a_vals) -> dB/dq^2, shape (4, ...)
    bs_rate:  bs_rate(mX, C, Lambda, is_ident) -> B(Bs -> X X-bar), or None if not computed
    Rows of the B -> K(*) rates: K, K*, K* longitudinal, K* transverse.
    """

    name: str
    description: str
    couplings: tuple
    two_body: bool
    rate: Callable
    bs_rate: Callable | None = None


_PSI = ("CVL", "CVR", "CVLp", "CVRp", "CSL", "CSR", "CSLp", "CSRp", "CTL", "CTR")

CHANNELS = {
    ch.name: ch
    for ch in [
        Channel(
            "phi",
            "B -> K(*) phi, scalar",
            ("CSL", "CSR"),
            True,
            lambda mX, C, Lambda, a_vals: B_BKphi(mX, C, a_vals),
        ),
        Channel("V", "B -> K(*) V, vector", ("CVL", "CVR", "CTL", "CTR"), True, B_BKV),
        Channel(
            "phiphi",
            "B -> K(*) phi phi-bar, scalar pair",
            ("CSL", "CSR", "CVL", "CVR"),
            False,
            dBdqsq_BKphiphi,
            B_Bsphiphi,
        ),
        Channel("psipsi", "B -> K(*) psi psi-bar, fermion pair", _PSI, False, dBdqsq_BKpsipsi, B_Bspsipsi),
        Channel("VV", "B -> K V V-bar, vector pair (K only)", ("CSL", "CSR"), False, dBdqsq_BKVV),
        Channel(
            "PsiPsi",
            "B -> K Psi Psi-bar, spin-3/2 pair (K only)",
            _PSI + ("CTSL", "CTSR", "CTPL", "CTPR"),
            False,
            dBdqsq_BKPsiPsi,
        ),
    ]
}


class Model:
    """One invisible-particle channel with its mass, couplings and operator scale.

    Model(channel, mX, Lambda = 1, is_ident = False, **couplings)

    channel:   a key of CHANNELS ("phi", "V", "phiphi", "psipsi", "VV", "PsiPsi")
    mX:        mass of the invisible particle(s) [GeV]
    Lambda:    scale of the effective operators [GeV]
    is_ident:  identical (real / Majorana) particles in the pair channels
    couplings: Wilson coefficients by name, e.g. CSL = 1e-8; the rest are zero
    """

    __slots__ = ("C", "Lambda", "channel", "is_ident", "mX")

    def __init__(self, channel, mX, Lambda=1.0, is_ident=False, **couplings):
        if channel not in CHANNELS:
            choices = "\n".join(f"  {name!r:9} {ch.description}" for name, ch in CHANNELS.items())
            raise ValueError(f"Unknown channel {channel!r}; the channels are\n{choices}")
        ch = CHANNELS[channel]
        unknown = set(couplings) - set(ch.couplings)
        if unknown:
            raise ValueError(
                f"Channel {channel!r} has no coupling(s) {sorted(unknown)}; its couplings are {list(ch.couplings)}"
            )
        values = [couplings.get(name, 0.0) for name in ch.couplings]
        dtype = complex if any(np.iscomplexobj(v) for v in values) else float
        object.__setattr__(self, "channel", ch)
        object.__setattr__(self, "mX", float(mX))
        object.__setattr__(self, "Lambda", float(Lambda))
        object.__setattr__(self, "is_ident", bool(is_ident))
        object.__setattr__(self, "C", np.array(values, dtype=dtype))
        self.C.setflags(write=False)

    def __setattr__(self, name, value):
        raise AttributeError("Model is immutable; use with_mass() or scaled() for modified copies")

    def __reduce__(self):  # pickling (joblib workers)
        return (_rebuild, (self.channel.name, self.mX, self.Lambda, self.is_ident, self.couplings))

    @property
    def couplings(self):
        "Non-zero couplings by name."
        return {name: c for name, c in zip(self.channel.couplings, self.C) if c != 0}

    @property
    def two_body(self):
        return self.channel.two_body

    def with_mass(self, mX):
        "Copy with a different mass."
        return Model(self.channel.name, mX, self.Lambda, self.is_ident, **self.couplings)

    def scaled(self, factor):
        "Copy with all couplings multiplied by factor."
        return Model(
            self.channel.name, self.mX, self.Lambda, self.is_ident, **{k: factor * v for k, v in self.couplings.items()}
        )

    def __repr__(self):
        parts = [repr(self.channel.name), f"mX = {self.mX:g}"]
        if self.Lambda != 1.0:
            parts.append(f"Lambda = {self.Lambda:g}")
        if self.is_ident:
            parts.append("is_ident = True")
        parts += [f"{k} = {v:g}" for k, v in self.couplings.items()]
        return f"Model({', '.join(parts)})"

    # Rates
    def B(self, a_vals=A_VALS_GRDV):
        "B(B -> K(*) X) for a two-body model (zeros for three-body models), rows K, K*, K*_L, K*_T."
        if not self.two_body or not np.any(self.C):
            return np.zeros(4)
        return self.channel.rate(self.mX, self.C, self.Lambda, a_vals)

    def dBdqsq(self, qsq, a_vals=A_VALS_GRDV):
        "dB/dq^2(B -> K(*) X X-bar) for a three-body model (zeros for two-body models)."
        if self.two_body or not np.any(self.C):
            return np.zeros((4,) + np.shape(qsq))
        return self.channel.rate(qsq, self.mX, self.C, self.Lambda, self.is_ident, a_vals)

    def B_Bs(self):
        "B(Bs -> X X-bar), the invisible Bs width constrained by LEP (0 where not computed)."
        if self.channel.bs_rate is None or not np.any(self.C):
            return 0.0
        return float(self.channel.bs_rate(self.mX, self.C, self.Lambda, self.is_ident))


def _rebuild(channel, mX, Lambda, is_ident, couplings):
    return Model(channel, mX, Lambda, is_ident, **couplings)


# Rates of an optional model (None: no new physics)
def B_tot(model, a_vals=A_VALS_GRDV):
    "B(B -> K(*) X) of a two-body model, rows K, K*, K*_L, K*_T; zeros for None or three-body models."
    return np.zeros(4) if model is None else model.B(a_vals)


def dBdqsq_tot(qsq, model, a_vals=A_VALS_GRDV):
    "dB/dq^2 of a three-body model, shape (4, ...); zeros for None or two-body models."
    return np.zeros((4,) + np.shape(qsq)) if model is None else model.dBdqsq(qsq, a_vals)


def B_Bs(model):
    "B(Bs -> invisible) of a model; 0 for None."
    return 0.0 if model is None else model.B_Bs()


def check_model(model, name="model", allow_none=True):
    "Raise a TypeError unless model is a Model (or None, if allowed)."
    if (model is None and allow_none) or isinstance(model, Model):
        return
    raise TypeError(
        f"{name} must be a lights.models.Model{' or None' if allow_none else ''}, not {type(model).__name__}"
    )
