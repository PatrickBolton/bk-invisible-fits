"""Plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def categorical_cmap(s, nc, nsc, cmap="tab10"):
    """Colour map with nsc shades (from full to pale) of each of the colours s..nc-1 of cmap."""
    ccolors = plt.get_cmap(cmap)(np.arange(s, nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = colors.hsv_to_rgb(arhsv)
        cols[i * nsc : (i + 1) * nsc, :] = rgb
    cmap = colors.ListedColormap(cols[cols > 0].reshape(-1, 3))
    return cmap


def setup_style():
    """Matplotlib style for the lights figures: serif Computer Modern fonts, LaTeX if installed.

    Without LaTeX, falls back to mathtext's Computer Modern fonts. /Library/TeX/texbin is
    the MacTeX location, which GUI apps may not have on PATH.
    """
    import os
    import shutil

    if os.path.isdir("/Library/TeX/texbin") and "/Library/TeX/texbin" not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
    plt.rcParams.update(
        {
            "font.size": 20,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "mathtext.fontset": "cm",
            "text.usetex": shutil.which("latex") is not None,
        }
    )


def sigma_levels(ax, labels=0, label_x=None, label_offset=1.0, n_lines=5, fontsize=14):
    """Faint dashed lines at -2 Delta ln L = -n^2 (n = 1 ... n_lines), the n-sigma levels of a
    Delta chi^2 with one degree of freedom, with "n sigma" labels for the first `labels` of
    them, placed at x = label_x and label_offset below each line.
    """
    for n in range(1, n_lines + 1):
        ax.axhline(-(n**2), ls="--", color="k", alpha=0.1)
    for n in range(1, labels + 1):
        ax.text(label_x, -(n**2) - label_offset, rf"${n}\sigma$", fontsize=fontsize)
