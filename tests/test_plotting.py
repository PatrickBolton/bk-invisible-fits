"""Plotting helpers."""

import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from lights.fits import ScanResult, profile_logLL_scan
from lights.models import Model
from lights.plotting import plot_contours


def toy_scan():
    x1, x2 = np.linspace(-2, 2, 21), np.linspace(-3, 3, 31)
    X, Y = np.meshgrid(x1, x2)
    grid = X**2 + Y**2  # 2.3 contour is a circle of radius ~1.5
    return ScanResult(grid, grid / 4, grid * 4, grid, x1, x2, (Model("phi", 2.1, CSL=1.0), None))


def test_scan_result_keeps_grids_and_models():
    gS = Model("phi", 2.1, CSL=5e-9, CSR=5e-9)
    gP = Model("phi", 2.1, CSL=5e-9, CSR=-5e-9)
    res = profile_logLL_scan(gS, gP, [-8, 8], [-30, 30], pts=(5, 7), n_jobs=1)
    assert res.x1 == pytest_approx_linspace(-8, 8, 5) and res.x2 == pytest_approx_linspace(-30, 30, 7)
    assert res.models == (gS, gP) and res.combined.shape == (7, 5)
    copy = pickle.loads(pickle.dumps(res))
    assert np.array_equal(copy.x1, res.x1) and repr(copy.models[0]) == repr(gS)
    *_, combined = res  # still unpacks as four grids
    assert len(res) == 4 and np.array_equal(combined, res.combined)


def pytest_approx_linspace(a, b, n):
    import pytest

    return pytest.approx(np.linspace(a, b, n))


def test_plot_contours_legend_matches_the_drawn_contours():
    fig, ax = plt.subplots()
    handles = plot_contours(ax, toy_scan(), scale=(1e-8, 1.0), lep=True, majorana=toy_scan())
    assert list(handles) == ["Belle II", "BaBar", "LEP", "Combined", "Combined, Majorana"]
    assert ax.get_xlim() == (-2e-8, 2e-8) and ax.get_ylim() == (-3.0, 3.0)
    # each legend entry has the colour of the contour it labels (grey, grey, red, blue, ...)
    r, g, b, _ = matplotlib.colors.to_rgba(handles["LEP"].get_color())
    assert r > g and r > b
    r, g, b, _ = matplotlib.colors.to_rgba(handles["BaBar"].get_color())
    assert abs(r - g) < 1e-6 and abs(g - b) < 1e-6
    r, g, b, _ = matplotlib.colors.to_rgba(handles["Combined"].get_color())
    assert b > r
    plt.close(fig)


def test_plot_contours_without_a_contour_still_gives_a_handle():
    scan = toy_scan()
    flat = ScanResult(scan.BelleII, np.zeros_like(scan.BaBar), scan.LEP, scan.combined, scan.x1, scan.x2)
    fig, ax = plt.subplots()
    handles = plot_contours(ax, flat)  # BaBar never reaches Delta chi^2 = 2.3
    assert "BaBar" in handles and handles["BaBar"] is not None
    plt.close(fig)


def test_plot_contours_skips_grids_that_were_not_computed():
    full = toy_scan()
    partial = ScanResult(None, full.BaBar, None, full.combined, full.x1, full.x2, full.models)
    handles = plot_contours(plt.subplots()[1], partial, lep=True)
    assert list(handles) == ["BaBar", "Combined"]
