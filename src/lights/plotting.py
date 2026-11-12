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


SIGMA_LEVELS_2D = [2.3, 6.18, 11.83]  # Delta chi^2 of the 1, 2, 3 sigma regions for two parameters


def plot_contours(ax, scan, scale=(1.0, 1.0), lep=False, color=0, shades=200, majorana=None, combined_label="Combined"):
    """Draw the contours of a profile_logLL_scan result on ax and return their legend handles.

    Draws the 1 sigma (Delta chi^2 = 2.3) contours of Belle II (dashed), BaBar (dotted) and,
    with lep = True, LEP (red, dotted), and the 1, 2 and 3 sigma contours and regions of the
    combined fit; grids the scan did not compute are skipped. majorana: an optional second scan whose combined 1 sigma region is overlaid.
    The axes show the coupling multipliers x1, x2 of the scan times scale, e.g. the value of
    the coupling combination they stand for; the limits are the extent of the grid.
    color, shades: categorical_cmap colour index and number of shades of the combined fit.
    Returns {label: handle} for a legend.
    """
    X, Y = np.meshgrid(np.asarray(scan.x1) * scale[0], np.asarray(scan.x2) * scale[1])
    single = {"levels": [2.3], "vmin": -20, "vmax": 10}
    handles = {}

    # 1 sigma contours of the single experiments (skipped if not in the scan)
    for label, grid, style in [
        ("Belle II", scan.BelleII, {"cmap": "Greys", "linestyles": "--"}),
        ("BaBar", scan.BaBar, {"cmap": "Greys", "linestyles": ":"}),
        ("LEP", scan.LEP if lep else None, {"cmap": "Reds", "linestyles": ":"}),
    ]:
        if grid is not None:
            handles[label] = ax.contour(X, Y, grid, **single, **style).legend_elements()[0][0]

    if scan.combined is not None:
        cmap = categorical_cmap(color, color + 1, shades)
        cs = ax.contour(X, Y, scan.combined, levels=SIGMA_LEVELS_2D, cmap=cmap, vmin=5)
        ax.contourf(X, Y, scan.combined, levels=[0] + SIGMA_LEVELS_2D, cmap=cmap, vmin=5, alpha=0.2)
        handles[combined_label] = cs.legend_elements()[0][0]

    if majorana is not None:
        cmap = categorical_cmap(9, 10, 200)
        cs = ax.contour(X, Y, majorana.combined, levels=[2.3], cmap=cmap, vmin=1)
        ax.contourf(X, Y, majorana.combined, levels=[0, 2.3], cmap=cmap, vmin=1, alpha=0.2)
        handles["Combined, Majorana"] = cs.legend_elements()[0][0]

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    return handles
