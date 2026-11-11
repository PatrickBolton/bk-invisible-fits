"""Plotting helpers."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def categorical_cmap(s, nc, nsc, cmap="tab10"):
    """Colour map with nsc shades (from full to pale) of each of the colours s..nc-1 of cmap."""
    ccolors = plt.get_cmap(cmap)(np.arange(s, nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    cmap = colors.ListedColormap(cols[cols > 0].reshape(-1,3))
    return cmap
