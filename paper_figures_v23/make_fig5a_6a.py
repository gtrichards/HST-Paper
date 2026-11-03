"""
Figures 5a and 6a (v23 sample): C IV blueshift and C IV distance versus L_1450.

Adapted from Trevor McCaffrey's L_versus_blueshift.ipynb
(Dropbox/Trevor/July2024/HST/Notebooks_final_MarchApril2023/), cells 22 and
24, which made L_versus_blue_ICA.pdf and L_versus_c4dist_ICA.pdf (the left
panels of the paper's Figs 5 and 6).  What changed:
  * HST sample = the adopted v23 sample (manmask 2 or 3), CIV_blue and CIV_EW
    from CIV_measurements_v23.csv; RM-program groups (as in Fig 1) and L1450 from hst_sample.py.
  * x axis is L_1450 for all samples; SDSS / GNIRS-DQS interpolated to
    1450 A per object (ref_samples.py), as in Fig 1.
    The secondary L_bol axis is removed.  The L_bol = 3e45 erg/s line is
    kept, as a vertical line at the same L_nu(1450) as Fig 1's horizontal one
    (Krawczyk et al. 2013; make_fig1_L1450.lbol_to_lognu1450).
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
    PYTHONPATH=~/Work/git python make_fig5a_6a.py            # plain figures
    PYTHONPATH=~/Work/git python make_fig5a_6a.py --labeled  # named objects

--labeled writes ONLY the *_labeled.pdf/png files (never the plain ones), with
the objects listed in Data/labeled_objects_v23.csv named (hst_sample.label_objects).

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
from hst_sample import load_adopted, add_logL1450, plot_hst, label_objects
from ref_samples import load_sdss, load_gnirs
from make_fig1_L1450 import lbol_to_lognu1450   # one definition for all figures

CIVDIST_DIR = os.path.expanduser("~/Work/git/CIV_Distance/")
sys.path.insert(0, CIVDIST_DIR)
from CIVfunctions import CIV_distance

C_ANG = 2.998e18           # speed of light in Angstrom/s

plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True


def main():
    labeled = "--labeled" in sys.argv
    suffix = "_labeled" if labeled else ""
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
    print("  HST with L1450: %d" % len(hst))

    # ---- SDSS: Rankine+20, luminosity interpolated to 1450 A (ref_samples.py) ----
    sdss = load_sdss(data_dir)
    logL_sdss = sdss["logLnu1450"].values
    c4blue_sdss = sdss["CIV_blue"].values
    c4dist_sdss = sdss["CIV_Distance"].values

    # ---- GNIRS-DQS ----
    gnirs = load_gnirs(data_dir)
    logL_gnirs = gnirs["logLnu1450"].values
    c4blue_gnirs = gnirs["CIV_blue"].values
    c4dist_gnirs = gnirs["CIV_dist_ICA_tvm"].values
    print("  SDSS %d, GNIRS-DQS %d" % (len(sdss), len(gnirs)))

    # Trevor used 27.1-32.4 for L2500; extended down to take in the low-L
    # HST objects (lowest is NGC 4395 at 24.83).  Explicit limits are needed
    # because Trevor's plot_contour pads its grid far beyond the data.
    xlim = (24.5, 32.6)
    # same label as Figure 1's y axis: the Angstrom outside the subscript (at
    # subscript size its ring sat clear of the A and read as a degree sign),
    # and L_nu written out because the axis is erg/s/Hz, not nu L_nu
    xlabel = r'$\log L_\nu(1450\,\mathrm{\AA})$ [erg\,s$^{-1}$\,Hz$^{-1}$]'

    # ---- Fig 5a: blueshift vs L (Trevor's cell 22) ----
    fig, ax = plt.subplots(1, 1, figsize=(9, 9.5))
    plot_hst(ax, logL_hst, c4blue_hst, hst)
    plot_contour(logL_sdss, c4blue_sdss, c=cs[2], mark="o",
                 nlevels=[0.05, 0.25, 0.50, 0.75, 0.95], ax=ax,
                 linewidths=1.5, s=3, alpha=0.4, label="SDSS", zorder=1)
    ax.plot(logL_gnirs, c4blue_gnirs, marker="v", markersize=11,
            markerfacecolor=cs[2], markeredgecolor="k",
            label="GNIRS-DQS", linestyle="", zorder=1)
    ax.set_ylabel(r'CIV Blueshift [km\,s$^{-1}$]', fontsize=35)
    ax.set_xlabel(xlabel, fontsize=32)
    ax.set_xlim(xlim)
    # L_bol = 3e45 erg/s, as in Trevor's figure (solid blue, behind the data)
    ax.axvline(lbol_to_lognu1450(np.log10(3e45)), color="b", ls="-", zorder=1)
    ax.tick_params(axis='both', which='major', labelsize=27.5)
    ax.legend(frameon=True, labelspacing=0.1, loc="upper left", prop={"size": 20})
    ax.set_ylim(-1299.99, 6199)   # Trevor: -599.99; see docstring
    plt.tight_layout()
    if labeled:
        label_objects(ax, logL_hst, c4blue_hst, hst, "5a", data_dir)
    out = os.path.join(out_dir, "Fig5a_CIV_blue_versus_L1450%s.pdf" % suffix)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace(".pdf", ".png"), dpi=100, bbox_inches='tight')
    print("saved", out)
    off = (c4blue_hst < -1299.99) | (c4blue_hst > 6199)
    print("  Fig 5a HST points outside the y limits: %d %s"
          % (off.sum(), list(hst.loc[off, "common_name"])))

    # ---- Fig 6a: C IV distance vs L (Trevor's cell 24) ----
    fig, ax = plt.subplots(1, 1, figsize=(9, 9.5))
    plot_hst(ax, logL_hst, c4dist_hst, hst)
    rng = np.random.default_rng(0)
    plot_contour(logL_sdss, c4dist_sdss + rng.uniform(-0.02, 0.02, len(c4dist_sdss)),
                 c=cs[2], mark="o", nlevels=[0.05, 0.25, 0.50, 0.75, 0.95],
                 ax=ax, linewidths=1.5, s=3, alpha=0.4, label="SDSS", zorder=1)
    ax.plot(logL_gnirs, c4dist_gnirs, marker="v", markersize=11,
            markerfacecolor=cs[2], markeredgecolor="k",
            label="GNIRS-DQS", linestyle="", zorder=1)
    ax.set_ylabel('CIV Distance [Scaled Units]', fontsize=35)
    ax.set_xlabel(xlabel, fontsize=32)
    ax.set_xlim(xlim)
    # L_bol = 3e45 erg/s, as in Trevor's figure (solid blue, behind the data)
    ax.axvline(lbol_to_lognu1450(np.log10(3e45)), color="b", ls="-", zorder=1)
    ax.tick_params(axis='both', which='major', labelsize=27.5)
    ax.set_ylim(-0.05, 1.25)
    plt.tight_layout()
    if labeled:
        label_objects(ax, logL_hst, c4dist_hst, hst, "6a", data_dir)
    out = os.path.join(out_dir, "Fig6a_CIV_dist_versus_L1450%s.pdf" % suffix)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace(".pdf", ".png"), dpi=100, bbox_inches='tight')
    print("saved", out)
    off = (c4dist_hst < -0.05) | (c4dist_hst > 1.25)
    print("  Fig 6a HST points outside the y limits: %d" % off.sum())


if __name__ == "__main__":
    main()
