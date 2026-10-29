"""
Trevor McCaffrey's plot_contour, copied verbatim from cell 3 of
Dropbox/Trevor/July2024/HST/Notebooks_final_MarchApril2023/CIVplot.ipynb
(the same function is in L_versus_blueshift.ipynb).

Used instead of Plotting Code/plotting_utils.plot_contour, which is a faster
rewrite with a wider kernel (bw_method=0.2) plus an extra Gaussian smoothing
of the density grid; that makes the contours visibly fatter than in Trevor's
published figures.  This version uses scipy's default (Scott) bandwidth and no
extra smoothing.  It evaluates the KDE at every data point, so with ~146k SDSS
quasars it takes about a minute per call.

How it works: a 2-D Gaussian KDE of the data is evaluated on a 100x100 grid
and at each data point, both rescaled to 0-1.  Contours are drawn at the
requested levels, and only the points below the lowest level (outside the
outermost contour) are drawn individually.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def plot_contour(xdata, ydata, c="k", mark=".", nlevels=3, ax=None, linewidths=0.5, s=3, alpha=1, label=None, zorder=2):
    good_data = (((~np.isnan(xdata))&(~np.isnan(ydata))) & ((~np.isinf(xdata))&(~np.isinf(ydata))))
    xdata = xdata[good_data]
    ydata = ydata[good_data]
    if ax is None:
        ax = plt.gca()
    xmin, xmax = xdata.min()-0.25*abs(xdata.mean()), xdata.max()+0.25*abs(xdata.mean())
    ymin, ymax = ydata.min()-0.25*abs(ydata.mean()), ydata.max()+0.25*abs(ydata.mean())
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([xdata, ydata])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    z = kernel(values)

    #transform Z to a 0-1 range
    Z = (Z - Z.min())/(Z.max() - Z.min())
    z = (z - z.min())/(z.max() - z.min())

    #ax.contour(X, Y, Z, levels=nlevels, linewidths=1.5, colors=c)
    levels = ax.contour(X, Y, Z, levels=nlevels, linewidths=linewidths, colors=[c]).levels

    # mask points above density threshold
    x = np.ma.masked_where(z > levels[0], xdata)
    y = np.ma.masked_where(z > levels[0], ydata)
    ax.scatter(x, y, color=c, marker=mark, s=3, alpha=alpha, label=label, zorder=zorder)
