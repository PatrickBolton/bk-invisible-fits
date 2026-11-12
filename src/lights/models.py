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

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .form_factors import A_VALS_GRDV
from .branching_fractions import (B_BKphi, B_BKV, dBdqsq_BKphiphi, dBdqsq_BKpsipsi, dBdqsq_BKVV,
                                  dBdqsq_BKPsiPsi, B_Bsphiphi, B_Bspsipsi)

# Rows of the returned B -> K(*) rates
K, KS, KS_L, KS_T = 0, 1, 2, 3


@dataclass(frozen = True)
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
    bs_rate: Optional[Callable] = None
    legacy_row: int = -1         # row in the old 6 x 14 coupling array
    legacy_suffix: str = ""      # suffix of the old cdict keys, e.g. "CSL_s"


_PSI = ("CVL", "CVR", "CVLp", "CVRp", "CSL", "CSR", "CSLp", "CSRp", "CTL", "CTR")

CHANNELS = {ch.name: ch for ch in [
    Channel("phi", "B -> K(*) phi, scalar", ("CSL", "CSR"), True,
            lambda mX, C, Lambda, a_vals: B_BKphi(mX, C, a_vals), None, 0, "s"),
    Channel("V", "B -> K(*) V, vector", ("CVL", "CVR", "CTL", "CTR"), True,
            B_BKV, None, 1, "v"),
    Channel("phiphi", "B -> K(*) phi phi-bar, scalar pair", ("CSL", "CSR", "CVL", "CVR"), False,
            dBdqsq_BKphiphi, B_Bsphiphi, 2, "ss"),
    Channel("psipsi", "B -> K(*) psi psi-bar, fermion pair", _PSI, False,
            dBdqsq_BKpsipsi, B_Bspsipsi, 3, "ff"),
    Channel("VV", "B -> K V V-bar, vector pair (K only)", ("CSL", "CSR"), False,
            dBdqsq_BKVV, None, 4, "vv"),
    Channel("PsiPsi", "B -> K Psi Psi-bar, spin-3/2 pair (K only)",
            _PSI + ("CTSL", "CTSR", "CTPL", "CTPR"), False, dBdqsq_BKPsiPsi, None, 5, "FF"),
]}

# Old coupling-array layout: cdict["CSL_s"] = (row, column) of a 6 x 14 array
cdict = {f"{name}_{ch.legacy_suffix}": (ch.legacy_row, j)
         for ch in CHANNELS.values() for j, name in enumerate(ch.couplings)}


class Model:
    """One invisible-particle channel with its mass, couplings and operator scale.

    Model(channel, mX, Lambda = 1, is_ident = False, **couplings)

    channel:   a key of CHANNELS ("phi", "V", "phiphi", "psipsi", "VV", "PsiPsi")
    mX:        mass of the invisible particle(s) [GeV]
    Lambda:    scale of the effective operators [GeV]
    is_ident:  identical (real / Majorana) particles in the pair channels
    couplings: Wilson coefficients by name, e.g. CSL = 1e-8; the rest are zero
    """

    __slots__ = ("channel", "mX", "Lambda", "is_ident", "C")

    def __init__(self, channel, mX, Lambda = 1., is_ident = False, **couplings):
        if channel not in CHANNELS:
            raise ValueError(f"Unknown channel {channel!r}; choose from {list(CHANNELS)}")
        ch = CHANNELS[channel]
        unknown = set(couplings) - set(ch.couplings)
        if unknown:
            raise ValueError(f"Channel {channel!r} has no coupling(s) {sorted(unknown)}; "
                             f"its couplings are {list(ch.couplings)}")
        values = [couplings.get(name, 0.) for name in ch.couplings]
        dtype = complex if any(np.iscomplexobj(v) for v in values) else float
        object.__setattr__(self, "channel", ch)
        object.__setattr__(self, "mX", float(mX))
        object.__setattr__(self, "Lambda", float(Lambda))
        object.__setattr__(self, "is_ident", bool(is_ident))
        object.__setattr__(self, "C", np.array(values, dtype = dtype))
        self.C.setflags(write = False)

    def __setattr__(self, name, value):
        raise AttributeError("Model is immutable; use with_mass() or scaled() for modified copies")

    def __reduce__(self):     # pickling (joblib workers)
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
        return Model(self.channel.name, self.mX, self.Lambda, self.is_ident,
                     **{k: factor * v for k, v in self.couplings.items()})

    def __repr__(self):
        parts = [repr(self.channel.name), f"mX = {self.mX:g}"]
        if self.Lambda != 1.:
            parts.append(f"Lambda = {self.Lambda:g}")
        if self.is_ident:
            parts.append("is_ident = True")
        parts += [f"{k} = {v:g}" for k, v in self.couplings.items()]
        return f"Model({', '.join(parts)})"

    # Rates
    def B(self, a_vals = A_VALS_GRDV):
        "B(B -> K(*) X) for a two-body model (zeros for three-body models), rows K, K*, K*_L, K*_T."
        if not self.two_body or not np.any(self.C):
            return np.zeros(4)
        return self.channel.rate(self.mX, self.C, self.Lambda, a_vals)

    def dBdqsq(self, qsq, a_vals = A_VALS_GRDV):
        "dB/dq^2(B -> K(*) X X-bar) for a three-body model (zeros for two-body models)."
        if self.two_body or not np.any(self.C):
            return np.zeros((4,) + np.shape(qsq))
        return self.channel.rate(qsq, self.mX, self.C, self.Lambda, self.is_ident, a_vals)

    def B_Bs(self):
        "B(Bs -> X X-bar), the invisible Bs width constrained by LEP (0 where not computed)."
        if self.channel.bs_rate is None or not np.any(self.C):
            return 0.
        return float(self.channel.bs_rate(self.mX, self.C, self.Lambda, self.is_ident))

    # The old 6 x 14 coupling arrays
    @classmethod
    def from_cnp(cls, cNP, mX, Lambda = 1., is_ident = False):
        """Model from an old-style 6 x 14 coupling array (see cdict). None if all couplings are zero."""
        cNP = np.asarray(cNP)
        rows = [ch for ch in CHANNELS.values() if np.any(cNP[ch.legacy_row, :len(ch.couplings)])]
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError("The coupling array sets couplings of more than one channel "
                             f"({[ch.name for ch in rows]}); a Model has a single channel")
        ch = rows[0]
        return cls(ch.name, mX, Lambda, is_ident,
                   **{name: cNP[ch.legacy_row, j] for j, name in enumerate(ch.couplings)})

    def to_cnp(self):
        "The old-style 6 x 14 coupling array."
        cNP = np.zeros((6, 14), dtype = self.C.dtype)
        cNP[self.channel.legacy_row, :len(self.C)] = self.C
        return cNP


def _rebuild(channel, mX, Lambda, is_ident, couplings):
    return Model(channel, mX, Lambda, is_ident, **couplings)


# Rates of an optional model (None: no new physics)
def B_tot(model, a_vals = A_VALS_GRDV):
    "B(B -> K(*) X) of a two-body model, rows K, K*, K*_L, K*_T; zeros for None or three-body models."
    return np.zeros(4) if model is None else model.B(a_vals)

def dBdqsq_tot(qsq, model, a_vals = A_VALS_GRDV):
    "dB/dq^2 of a three-body model, shape (4, ...); zeros for None or two-body models."
    return np.zeros((4,) + np.shape(qsq)) if model is None else model.dBdqsq(qsq, a_vals)

def B_Bs(model):
    "B(Bs -> invisible) of a model; 0 for None."
    return 0. if model is None else model.B_Bs()


def from_legacy(fname, args, kwargs, names, defaults):
    """Translate an old-style call (coupling arrays) into Models.

    names: the old positional argument names; defaults: their default values. Returns the
    dict of old arguments, with each "cNP*" entry replaced by a Model (or None).
    """
    import warnings
    warnings.warn(f"{fname}: coupling arrays (cdict) are deprecated; pass lights.models.Model "
                  "objects instead", DeprecationWarning, stacklevel = 3)
    vals = dict(defaults)
    vals.update(zip(names, args))
    vals.update(kwargs)
    for key in [k for k in vals if k.startswith("cNP") and vals[k] is not None]:
        vals[key] = Model.from_cnp(vals[key], vals["mX"], vals.get("Lambda", 1.), vals.get("is_ident", False))
    return vals
