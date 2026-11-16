"""HistFactory (pyhf JSON) likelihoods as arrays, for compiled likelihoods.

Converts a pyhf workspace into arrays that a numba likelihood evaluates directly (see
lights.likelihoods, Belle II public likelihood). It supports the modifiers of the Belle II
B+ -> K+ nu nu-bar release, with pyhf's default interpolation:

    histosys    additive shift of a sample, interpolated with pyhf's code4p
    normsys     multiplicative factor of a sample, interpolated with pyhf's code4
    staterror   one factor gamma per bin, with a Gaussian constraint of width sigma/nominal;
                each staterror name must belong to a single sample (as in the Belle II release)
    normfactor  free multiplicative parameter: only on the signal sample, where it is replaced

One sample (the signal) is not taken from the workspace: its yields are supplied by the
caller, and its histosys modifiers act on them as relative shifts (delta / nominal), so the
same detector systematics apply to any signal shape. For the nominal signal this is exactly
pyhf's model. Modifiers named in `drop` are removed from the model (e.g. the signal form factor
uncertainties, which lights replaces by its own).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class HistFactoryArrays:
    """A HistFactory model as arrays: N bins (all channels in order), S background samples,
    H histosys parameters, M normsys parameters, G staterror parameters.

    obs (N,)                     observed events
    channel_bins                 ((channel name, first bin, number of bins), ...)
    bkg_names                    ((channel name, sample name), ...) of the S background samples
    bkg_nominal (S, N)           background yields (zero outside the sample's channel)
    histosys_names (H,)          parameter names of the histosys modifiers
    bkg_hs_up, bkg_hs_dn (S, H, N)   histosys deltas hi - nominal and nominal - lo
    signal_nominal (N,)          the workspace's signal yields (zero where there is no signal)
    sig_hs_up, sig_hs_dn (H, N)  signal histosys deltas relative to the signal nominal
    normsys_names (M,)           parameter names of the normsys modifiers
    bkg_ns_up, bkg_ns_dn (S, M)  normsys factors hi and lo (1 where not applied)
    bkg_ns_coef (S, M, 6)        code4 polynomial coefficients a_1 ... a_6 for |alpha| < 1
    staterror_sigma (G,)         relative widths of the gamma constraints
    bkg_stat (S, N), sig_stat (N,)   staterror parameter of each sample bin (-1: none)
    """

    obs: np.ndarray
    channel_bins: tuple
    bkg_names: tuple
    bkg_nominal: np.ndarray
    histosys_names: tuple
    bkg_hs_up: np.ndarray
    bkg_hs_dn: np.ndarray
    signal_nominal: np.ndarray
    sig_hs_up: np.ndarray
    sig_hs_dn: np.ndarray
    normsys_names: tuple
    bkg_ns_up: np.ndarray
    bkg_ns_dn: np.ndarray
    bkg_ns_coef: np.ndarray
    staterror_names: tuple
    staterror_sigma: np.ndarray
    bkg_stat: np.ndarray
    sig_stat: np.ndarray


def code4_coefficients(up, dn):
    """Coefficients a_1 ... a_6 of pyhf's code4 polynomial 1 + sum_i a_i alpha^i for |alpha| < 1.

    They match the value and the first two derivatives of the exponential interpolation
    up^alpha (alpha >= 1) and dn^-alpha (alpha <= -1) at alpha = +-1.
    """
    i = np.arange(1, 7)
    A = np.array([np.ones(6), i, i * (i - 1), (-1.0) ** i, i * (-1.0) ** (i - 1), i * (i - 1) * (-1.0) ** (i - 2)])
    lu, ld = np.log(up), np.log(dn)
    b = np.array([up - 1, up * lu, up * lu**2, dn - 1, -dn * ld, dn * ld**2])
    return np.linalg.solve(A, b)


def build(workspace, signal="signal", drop=()):
    """HistFactoryArrays of a pyhf workspace (a dict in the pyhf JSON format).

    signal: name of the signal sample, whose yields the likelihood takes as input
    drop:   names of modifiers to remove (the signal normfactor is always removed)
    """
    channels = workspace["channels"]
    obs_by_channel = {o["name"]: o["data"] for o in workspace["observations"]}
    channel_bins, start = [], 0
    for ch in channels:
        n = len(ch["samples"][0]["data"])
        channel_bins.append((ch["name"], start, n))
        start += n
    N = start
    obs = np.concatenate([obs_by_channel[ch["name"]] for ch in channels]).astype(float)

    # Parameters, in order of first appearance
    histosys, normsys, staterror = {}, {}, {}
    for ch, (_, b0, n) in zip(channels, channel_bins):
        for s in ch["samples"]:
            for m in s["modifiers"]:
                if m["name"] in drop:
                    continue
                if m["type"] == "histosys":
                    histosys.setdefault(m["name"], len(histosys))
                elif m["type"] == "normsys":
                    if s["name"] == signal:
                        raise ValueError(f"normsys {m['name']!r} on the signal is not supported")
                    normsys.setdefault(m["name"], len(normsys))
                elif m["type"] == "staterror":
                    if m["name"] in staterror:
                        raise ValueError(f"staterror {m['name']!r} is shared between samples")
                    staterror[m["name"]] = (
                        len(staterror),
                        b0,
                        n,
                        np.asarray(m["data"], float),
                        np.asarray(s["data"], float),
                    )
                elif m["type"] == "normfactor":
                    if s["name"] != signal:
                        raise ValueError(f"normfactor {m['name']!r} on a background is not supported")
                else:
                    raise ValueError(f"modifier type {m['type']!r} is not supported")

    # staterror parameters: one per bin of each staterror modifier
    sigma, stat_start = [], {}
    for name, (_, b0, n, err, nom) in staterror.items():
        stat_start[name] = len(sigma)
        sigma += list(err / nom)

    bkg = [
        (ch, s, b0, n) for ch, (_, b0, n) in zip(channels, channel_bins) for s in ch["samples"] if s["name"] != signal
    ]
    S, H, M = len(bkg), len(histosys), len(normsys)
    bkg_nominal, bkg_hs_up, bkg_hs_dn = np.zeros((S, N)), np.zeros((S, H, N)), np.zeros((S, H, N))
    bkg_ns_up, bkg_ns_dn, bkg_stat = np.ones((S, M)), np.ones((S, M)), -np.ones((S, N), dtype=np.int64)
    signal_nominal, sig_hs_up, sig_hs_dn = np.zeros(N), np.zeros((H, N)), np.zeros((H, N))
    sig_stat = -np.ones(N, dtype=np.int64)

    def apply(m, nom, sl, hs_up, hs_dn, stat):
        if m["name"] in drop or m["type"] == "normfactor":
            return
        if m["type"] == "histosys":
            k = histosys[m["name"]]
            hs_up[k, sl] = np.asarray(m["data"]["hi_data"], float) - nom
            hs_dn[k, sl] = nom - np.asarray(m["data"]["lo_data"], float)
        elif m["type"] == "staterror":
            stat[sl] = stat_start[m["name"]] + np.arange(sl.stop - sl.start)

    for i, (ch, s, b0, n) in enumerate(bkg):
        sl, nom = slice(b0, b0 + n), np.asarray(s["data"], float)
        bkg_nominal[i, sl] = nom
        for m in s["modifiers"]:
            if m["type"] == "normsys" and m["name"] not in drop:
                k = normsys[m["name"]]
                bkg_ns_up[i, k], bkg_ns_dn[i, k] = m["data"]["hi"], m["data"]["lo"]
            else:
                apply(m, nom, sl, bkg_hs_up[i], bkg_hs_dn[i], bkg_stat[i])

    for ch, (_, b0, n) in zip(channels, channel_bins):
        for s in ch["samples"]:
            if s["name"] != signal:
                continue
            sl, nom = slice(b0, b0 + n), np.asarray(s["data"], float)
            signal_nominal[sl] = nom
            up, dn = np.zeros((H, N)), np.zeros((H, N))
            for m in s["modifiers"]:
                apply(m, nom, sl, up, dn, sig_stat)
            sig_hs_up[:, sl], sig_hs_dn[:, sl] = up[:, sl] / nom, dn[:, sl] / nom

    coef = np.zeros((S, M, 6))
    for i in range(S):
        for k in range(M):
            if bkg_ns_up[i, k] != 1 or bkg_ns_dn[i, k] != 1:
                coef[i, k] = code4_coefficients(bkg_ns_up[i, k], bkg_ns_dn[i, k])

    return HistFactoryArrays(
        obs=obs,
        channel_bins=tuple(channel_bins),
        bkg_names=tuple((ch["name"], s["name"]) for ch, s, _, _ in bkg),
        bkg_nominal=bkg_nominal,
        histosys_names=tuple(histosys),
        bkg_hs_up=bkg_hs_up,
        bkg_hs_dn=bkg_hs_dn,
        signal_nominal=signal_nominal,
        sig_hs_up=sig_hs_up,
        sig_hs_dn=sig_hs_dn,
        normsys_names=tuple(normsys),
        bkg_ns_up=bkg_ns_up,
        bkg_ns_dn=bkg_ns_dn,
        bkg_ns_coef=coef,
        staterror_names=tuple(staterror),
        staterror_sigma=np.asarray(sigma),
        bkg_stat=bkg_stat,
        sig_stat=sig_stat,
    )
