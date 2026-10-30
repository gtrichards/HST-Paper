"""
Figure 4 (v23 sample): log C IV EW versus C IV blueshift.

Adapted from Trevor McCaffrey's CIVplot.ipynb, cell 19
(Dropbox/Trevor/July2024/HST/Notebooks_final_MarchApril2023/CIVplot.ipynb),
which made figures/CIV_logEW_versus_blue_ICA.pdf.  What changed:
  * HST sample = the adopted v23 sample (manmask 2 or 3), with CIV_blue and
    CIV_EW from CIV_measurements_v23.csv instead of CIV_blue_tvm / CIV_EW_tvm.
  * M_BH categories from the old catalogue, matched by position (hst_sample.py).
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
from hst_sample import load_adopted

plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True


def main():
    data_dir = os.path.join(HERE, "..", "Data")
    out_dir = os.path.join(HERE, "..", "Figures")
    cs = palettable.colorbrewer.qualitative.Dark2_5.mpl_colors

    # HST: adopted v23 sample
    hst = load_adopted(data_dir)
    c4blueICA_hst = hst["CIV_blue"].values
    c4logewICA_hst = np.log10(hst["CIV_EW"].values)
    maskMBH_rm = hst["mask_rm"].values
    maskMBH_sr = hst["mask_sr"].values
    mask_no_mbh = hst["mask_no_mbh"].values

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

    ax.plot(c4blueICA_hst[maskMBH_rm], c4logewICA_hst[maskMBH_rm], marker="o",
            markersize=15, markerfacecolor=cs[3], markeredgecolor="k",
            label="HST, RM $M_{BH}$", linestyle="", zorder=3)
    ax.plot(c4blueICA_hst[maskMBH_sr], c4logewICA_hst[maskMBH_sr], marker="o",
            markersize=13.5, markerfacecolor="w", markeredgecolor=cs[3], mew=1.8,
            label="HST, Scaling Relation $M_{BH}$", linestyle="", zorder=2)
    ax.plot(c4blueICA_hst[mask_no_mbh], c4logewICA_hst[mask_no_mbh], marker="^",
            markersize=16.5, markerfacecolor=cs[4], markeredgecolor="k",
            label="HST, no $M_{BH}$", linestyle="", zorder=2)

    plot_contour(c4blueICA_sdss, c4logewICA_sdss, c=cs[2], mark="o",
                 nlevels=[0.05, 0.25, 0.50, 0.75, 0.95], ax=ax,
                 linewidths=1.5, s=3, alpha=0.4, label="SDSS", zorder=1)

    ax.plot(c4blueICA_gnirs, c4logewICA_gnirs, marker="v", markersize=13.5,
            markerfacecolor=cs[2], markeredgecolor="k", label="GNIRS-DQS",
            linestyle="", zorder=1)

    ax.set_xlabel(r'CIV Blueshift [km\,s$^{-1}$]', fontsize=35)
    ax.set_ylabel(r'$\log_{10}$\,CIV\,EW [Å]', fontsize=35)
    ax.tick_params(axis='both', which='major', labelsize=27.5)
    ax.legend(frameon=True, labelspacing=0.1, loc="best", prop={"size": 20})
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
    out = os.path.join(out_dir, "Fig4_CIV_logEW_versus_blue.pdf")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace(".pdf", ".png"), dpi=100, bbox_inches='tight')
    print("saved", out)


if __name__ == "__main__":
    main()
