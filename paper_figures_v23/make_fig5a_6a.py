"""
Figures 5a and 6a (v23 sample): C IV blueshift and C IV distance versus L_1450.

Adapted from Trevor McCaffrey's L_versus_blueshift.ipynb
(Dropbox/Trevor/July2024/HST/Notebooks_final_MarchApril2023/), cells 22 and
24, which made L_versus_blue_ICA.pdf and L_versus_c4dist_ICA.pdf (the left
panels of the paper's Figs 5 and 6).  What changed:
  * HST sample = the adopted v23 sample (manmask 2 or 3), CIV_blue and CIV_EW
    from CIV_measurements_v23.csv; M_BH categories and L1450 from hst_sample.py.
  * x axis is L_1450 for HST and L_1350 for SDSS / GNIRS-DQS, as in Fig 1.
    The secondary L_bol axis and the L_bol = 3e45 line are removed (Trevor's
    bolometric correction is for L_2500).
  * HST C IV distance is computed here from (CIV_blue, log CIV_EW) with
    Trevor's own CIV_distance() and best-fit curve, from the CIV_Distance
    repository (github.com/trevormccaffrey/CIV_Distance), assumed cloned at
    ~/Work/git/CIV_Distance.  SDSS and GNIRS distances are the catalogue values
    Trevor used (CIV_Distance, CIV_dist_ICA_tvm).
  * SDSS contours use Trevor's original plot_contour (trevor_contour.py).
  * Fig 5a y axis extended down from -600 to -1300 km/s to take in the HST
    objects redward of systemic.  4C 55.17 (-1603 km/s; next lowest is -1186)
    is deliberately left off, to be noted in the caption.
  * The +-0.02 jitter Trevor added to the SDSS distances (so the KDE does not
    see ties) is kept, with a fixed random seed so the figure is reproducible.

Usage:
    PYTHONPATH=~/Work/git python make_fig5a_6a.py
Output: Figures/Fig5a_CIV_blue_versus_L1450.pdf
        Figures/Fig6a_CIV_dist_versus_L1450.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import palettable
# NOT `import richardsplot`: that resolves to the package's __init__.py, which
# is EMPTY, so the style silently never applies and the figure comes out in
# matplotlib defaults -- sans-serif, ticks outward, no minor ticks, and with
# usetex off so \, prints literally and \AA renders in math italic.  The
# style lives in richardsplot/richardsplot.py and has to be named twice.
from richardsplot import richardsplot as rplot   # noqa: F401
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
from trevor_contour import plot_contour   # Trevor's original, see that file
from hst_sample import load_adopted, add_logL1450

CIVDIST_DIR = os.path.expanduser("~/Work/git/CIV_Distance/")
sys.path.insert(0, CIVDIST_DIR)
from CIVfunctions import CIV_distance

C_ANG = 2.998e18           # speed of light in Angstrom/s

plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True


def plot_hst(ax, x, y, mask_rm, mask_sr, mask_no_mbh):
    """Trevor's three HST symbols."""
    cs = palettable.colorbrewer.qualitative.Dark2_5.mpl_colors
    ax.plot(x[mask_rm], y[mask_rm], marker="o", markersize=15,
            markerfacecolor=cs[3], markeredgecolor="k",
            label="HST, RM $M_{BH}$", linestyle="", zorder=3)
    ax.plot(x[mask_sr], y[mask_sr], marker="o", markersize=13.5,
            markerfacecolor="w", markeredgecolor=cs[3], mew=1.8,
            label="HST, Scaling $M_{BH}$", linestyle="", zorder=2)
    ax.plot(x[mask_no_mbh], y[mask_no_mbh], marker="^", markersize=16.5,
            markerfacecolor=cs[4], markeredgecolor="k",
            label="HST, no $M_{BH}$", linestyle="", zorder=2)


def main():
    data_dir = os.path.join(HERE, "..", "Data")
    out_dir = os.path.join(HERE, "..", "Figures")
    os.makedirs(out_dir, exist_ok=True)
    cs = palettable.colorbrewer.qualitative.Dark2_5.mpl_colors

    # ---- HST: adopted v23 sample with L1450 ----
    hst = add_logL1450(load_adopted(data_dir), data_dir)
    hst = hst[np.isfinite(hst["logLnu1450"])].reset_index(drop=True)
    logL_hst = hst["logLnu1450"].values
    c4blue_hst = hst["CIV_blue"].values

    # C IV distance with Trevor's function (blueshift, log10 EW)
    bestfit = np.load(CIVDIST_DIR + "data/bestfit.npy")
    data_hst = np.array([c4blue_hst, np.log10(hst["CIV_EW"].values)]).T
    c4dist_hst = CIV_distance(data_hst, bestfit, path=CIVDIST_DIR)
    # CIV_distance silently skips a point it cannot place on the curve,
    # which would shift every later value; make sure that did not happen
    assert len(c4dist_hst) == len(hst), "CIV_distance dropped a point"
    masks = (hst["mask_rm"].values, hst["mask_sr"].values,
             hst["mask_no_mbh"].values)
    print("  HST with L1450: %d" % len(hst))

    # ---- SDSS: Rankine+20, good, with a 1350 A luminosity ----
    sdss_tbl = Table.read(os.path.join(
        data_dir, "Rankine20_CIV_HeII_wDelGI_wdr16luminosity.fits"))
    names = [n for n in sdss_tbl.colnames if len(sdss_tbl[n].shape) <= 1]
    sdss = sdss_tbl[names].to_pandas()
    sdss = sdss[(sdss["LOGL1350_dr16"] > 0) & (sdss["good"] == True)]
    logL_sdss = sdss["LOGL1350_dr16"].values - np.log10(C_ANG / 1350.)  # to L_nu
    c4blue_sdss = sdss["CIV_blue"].values
    c4dist_sdss = sdss["CIV_Distance"].values

    # ---- GNIRS-DQS ----
    gnirs = pd.read_csv(os.path.join(data_dir, "gnirsdqs_wRankine_wdr16Lum.csv"))
    gnirs = gnirs[(gnirs["good"] == True) & (gnirs["LOGL1350_dr16"].values > 0)]
    logL_gnirs = gnirs["LOGL1350_dr16"].values - np.log10(C_ANG / 1350.)
    c4blue_gnirs = gnirs["CIV_blue"].values
    c4dist_gnirs = gnirs["CIV_dist_ICA_tvm"].values
    print("  SDSS %d, GNIRS-DQS %d" % (len(sdss), len(gnirs)))

    # Trevor used 27.1-32.4 for L2500; extended down to take in the low-L
    # HST objects (lowest is NGC 4395 at 24.83).  Explicit limits are needed
    # because Trevor's plot_contour pads its grid far beyond the data.
    xlim = (24.5, 32.6)
    xlabel = r'log\,$L_{1450\AA}$ [erg\,s$^{-1}$\,Hz$^{-1}$]'

    # ---- Fig 5a: blueshift vs L (Trevor's cell 22) ----
    fig, ax = plt.subplots(1, 1, figsize=(9, 9.5))
    plot_hst(ax, logL_hst, c4blue_hst, *masks)
    plot_contour(logL_sdss, c4blue_sdss, c=cs[2], mark="o",
                 nlevels=[0.05, 0.25, 0.50, 0.75, 0.95], ax=ax,
                 linewidths=1.5, s=3, alpha=0.4, label=r"SDSS ($L_{1350}$)", zorder=1)
    ax.plot(logL_gnirs, c4blue_gnirs, marker="v", markersize=11,
            markerfacecolor=cs[2], markeredgecolor="k",
            label=r"GNIRS-DQS ($L_{1350}$)", linestyle="", zorder=1)
    ax.set_ylabel(r'CIV Blueshift [km\,s$^{-1}$]', fontsize=35)
    ax.set_xlabel(xlabel, fontsize=35)
    ax.set_xlim(xlim)
    ax.tick_params(axis='both', which='major', labelsize=27.5)
    ax.legend(frameon=True, labelspacing=0.1, loc="upper left", prop={"size": 20})
    ax.set_ylim(-1299.99, 6199)   # Trevor: -599.99; see docstring
    plt.tight_layout()
    out = os.path.join(out_dir, "Fig5a_CIV_blue_versus_L1450.pdf")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace(".pdf", ".png"), dpi=100, bbox_inches='tight')
    print("saved", out)
    off = (c4blue_hst < -1299.99) | (c4blue_hst > 6199)
    print("  Fig 5a HST points outside the y limits: %d %s"
          % (off.sum(), list(hst.loc[off, "common_name"])))

    # ---- Fig 6a: C IV distance vs L (Trevor's cell 24) ----
    fig, ax = plt.subplots(1, 1, figsize=(9, 9.5))
    plot_hst(ax, logL_hst, c4dist_hst, *masks)
    rng = np.random.default_rng(0)
    plot_contour(logL_sdss, c4dist_sdss + rng.uniform(-0.02, 0.02, len(c4dist_sdss)),
                 c=cs[2], mark="o", nlevels=[0.05, 0.25, 0.50, 0.75, 0.95],
                 ax=ax, linewidths=1.5, s=3, alpha=0.4, label=r"SDSS ($L_{1350}$)", zorder=1)
    ax.plot(logL_gnirs, c4dist_gnirs, marker="v", markersize=11,
            markerfacecolor=cs[2], markeredgecolor="k",
            label=r"GNIRS-DQS ($L_{1350}$)", linestyle="", zorder=1)
    ax.set_ylabel('CIV Distance [Scaled Units]', fontsize=35)
    ax.set_xlabel(xlabel, fontsize=35)
    ax.set_xlim(xlim)
    ax.tick_params(axis='both', which='major', labelsize=27.5)
    ax.set_ylim(-0.05, 1.25)
    plt.tight_layout()
    out = os.path.join(out_dir, "Fig6a_CIV_dist_versus_L1450.pdf")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace(".pdf", ".png"), dpi=100, bbox_inches='tight')
    print("saved", out)
    off = (c4dist_hst < -0.05) | (c4dist_hst > 1.25)
    print("  Fig 6a HST points outside the y limits: %d" % off.sum())


if __name__ == "__main__":
    main()
