"""
Figure 4 (v23 sample): log C IV EW versus C IV blueshift.

Adapted from Trevor McCaffrey's CIVplot.ipynb, cell 19
(Dropbox/Trevor/July2024/HST/Notebooks_final_MarchApril2023/CIVplot.ipynb),
which made figures/CIV_logEW_versus_blue_ICA.pdf.  What changed:
  * HST sample = the adopted v23 sample (manmask 2 or 3), with CIV_blue and
    CIV_EW from CIV_measurements_v23.csv instead of CIV_blue_tvm / CIV_EW_tvm.
  * HST symbols mark RM-program membership, as in Fig 1 (hst_sample.plot_hst).
  * The C IV distance curve (CIV_Distance/data/bestfit.npy) is overplotted.
  * SDSS contours use Trevor's original plot_contour (trevor_contour.py).
  * y axis extended from 2.72 to 2.92 to take in the high-EW HST objects.
    Two objects are deliberately left off the plot, each far from the rest
    (to be noted in the caption): Mrk 1310 (EW 1139 A, log 3.06; next highest
    is log 2.87) and 4C 55.17 (blueshift -1603 km/s; next lowest is -1186).
Everything else (SDSS and GNIRS selections, symbols, x limits) is as in cell 19.

Blueshift convention: positive = blueshifted.

Usage:
    PYTHONPATH=~/Work/git python make_fig4.py
Output: Figures/Fig4_CIV_logEW_versus_blue.pdf
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
from hst_sample import load_adopted, plot_hst, label_objects

plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True


def main():
    # --labeled writes ONLY Fig4_..._labeled.pdf/png, naming the objects in
    # Data/labeled_objects_v23.csv; the plain figure is not touched.
    labeled = "--labeled" in sys.argv
    suffix = "_labeled" if labeled else ""
    data_dir = os.path.join(HERE, "..", "Data")
    out_dir = os.path.join(HERE, "..", "Figures")
    cs = palettable.colorbrewer.qualitative.Dark2_5.mpl_colors

    # HST: adopted v23 sample
    hst = load_adopted(data_dir)
    c4blueICA_hst = hst["CIV_blue"].values
    c4logewICA_hst = np.log10(hst["CIV_EW"].values)

    # SDSS: Rankine+20, all rows (Trevor's 'good' cut was commented out)
    sdss_tbl = Table.read(os.path.join(
        data_dir, "Rankine20_CIV_HeII_wDelGI_wdr16luminosity.fits"))
    names = [n for n in sdss_tbl.colnames if len(sdss_tbl[n].shape) <= 1]
    sdss = sdss_tbl[names].to_pandas()
    with np.errstate(divide="ignore", invalid="ignore"):
        c4blueICA_sdss = sdss["CIV_blue"].values
        c4logewICA_sdss = np.log10(sdss["CIV_EW"].values)
    ok = np.isfinite(c4blueICA_sdss) & np.isfinite(c4logewICA_sdss)
    c4blueICA_sdss, c4logewICA_sdss = c4blueICA_sdss[ok], c4logewICA_sdss[ok]

    # GNIRS-DQS, good == True
    gnirs = pd.read_csv(os.path.join(data_dir, "gnirsdqs_wRankine_wdr16Lum.csv"))
    gnirs = gnirs[gnirs["good"] == True]
    c4blueICA_gnirs = gnirs["CIV_blue"].values
    c4logewICA_gnirs = np.log10(gnirs["CIV_EW"].values)
    print("  SDSS %d, GNIRS-DQS %d" % (len(c4blueICA_sdss), len(c4blueICA_gnirs)))

    # ---- plot (Trevor's cell 19) ----
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    plot_hst(ax, c4blueICA_hst, c4logewICA_hst, hst)

    # The C IV distance curve (Trevor's best-fit curve through the SDSS quasars,
    # CIV_Distance/data/bestfit.npy, columns blueshift and log10 EW) -- the
    # line D_CIV is measured along.  It runs from blueshift 50 km/s,
    # log EW 2.50 (EW 316 A, D = 0) to 6000 km/s, log EW 1.12; objects beyond
    # its high-EW end all project onto that end and get D = 0.
    bestfit = np.load(os.path.expanduser("~/Work/git/CIV_Distance/data/bestfit.npy"))
    ax.plot(bestfit[:, 0], bestfit[:, 1], "-k", lw=2.0, zorder=8)   # above all points

    plot_contour(c4blueICA_sdss, c4logewICA_sdss, c=cs[2], mark="o",
                 nlevels=[0.05, 0.25, 0.50, 0.75, 0.95], ax=ax,
                 linewidths=1.5, s=3, alpha=0.4, label="SDSS", zorder=1)

    ax.plot(c4blueICA_gnirs, c4logewICA_gnirs, marker="v", markersize=13.5,
            markerfacecolor=cs[2], markeredgecolor="k", label="GNIRS-DQS",
            linestyle="", zorder=1)

    ax.set_xlabel(r'CIV Blueshift [km\,s$^{-1}$]', fontsize=35)
    ax.set_ylabel(r'$\log_{10}$\,CIV\,EW [Å]', fontsize=35)
    ax.tick_params(axis='both', which='major', labelsize=27.5)
    ax.legend(frameon=True, labelspacing=0.1, loc="best", prop={"size": 16})
    ax.set_xlim(-1299.99, 6199)
    ax.set_ylim(0.62, 2.92)   # Trevor: 2.72; see docstring
    plt.tight_layout()

    # report HST points that fall outside Trevor's axis limits
    off = ((c4blueICA_hst < -1299.99) | (c4blueICA_hst > 6199) |
           (c4logewICA_hst < 0.62) | (c4logewICA_hst > 2.92))
    print("  HST points outside the axis limits: %d" % off.sum())
    for _, r in hst[off].iterrows():
        print("    row %d %s  blue %.0f  EW %.1f"
              % (r.row_num, r.common_name, r.CIV_blue, r.CIV_EW))

    os.makedirs(out_dir, exist_ok=True)
    if labeled:
        label_objects(ax, c4blueICA_hst, c4logewICA_hst, hst, "4", data_dir)
    out = os.path.join(out_dir, "Fig4_CIV_logEW_versus_blue%s.pdf" % suffix)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace(".pdf", ".png"), dpi=100, bbox_inches='tight')
    print("saved", out)


if __name__ == "__main__":
    main()
