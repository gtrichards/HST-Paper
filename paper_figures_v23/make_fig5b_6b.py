"""
Figures 5b and 6b (v23 sample): C IV blueshift and C IV distance versus L_W1,
the right-hand panels of Figs 5 and 6 (panel a, versus L_1450, is
make_fig5a_6a.py).

Adapted from Trevor McCaffrey's Lwise_versus_blueshift.ipynb
(Dropbox/Trevor/July2024/HST/Notebooks_final_MarchApril2023/), cells 30 and 32.
  * L_W1 is Trevor's: nu L_nu at the observed W1 wavelength,
        L_W1 = 4 pi D_L^2 * nu F_nu,   F_nu = 309.540 Jy * 10^(-W1/2.5),
        nu = c / 3.368 um,
    in erg/s (no K-correction: observed-frame W1 is rest 1.5-3.4 um for the
    HST sample and 0.8-1.3 um for SDSS).
  * W1 for all three samples is AllWISE (Data/AllWISE_W1_v23.csv, from
    fetch_allwise_w1.py), nearest source within 3 arcsec.  Trevor used unWISE
    for SDSS; that matched file is no longer available.
  * HST sample, symbols, SDSS / GNIRS selections and C IV distances are the
    same as in panel a (hst_sample.plot_hst, ref_samples, civ_distance_ext).
  * The L_bol = 3e45 erg/s line is placed as Trevor placed it: through a
    straight-line fit of log L_W1 against the UV luminosity for SDSS, here
    against log L_nu(1450), evaluated at panel a's line (Krawczyk et al. 2013).
  * --labeled writes ONLY the *_labeled files, naming the same objects as the
    panel-a labeled figures (Data/labeled_objects_v23.csv, figures 5a/6a).

Usage:  PYTHONPATH=~/Work/git python make_fig5b_6b.py [--labeled]
Output: Figures/Fig5b_CIV_blue_versus_LW1.pdf/png
        Figures/Fig6b_CIV_dist_versus_LW1.pdf/png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import palettable
# NOT `import richardsplot`: the package __init__ is empty (see make_fig1_L1450.py)
from richardsplot import richardsplot as rplot   # noqa: F401
from astropy.cosmology import FlatLambdaCDM

HERE = os.path.dirname(os.path.abspath(__file__))
from trevor_contour import plot_contour   # Trevor's original, see that file
from hst_sample import load_adopted, add_logL1450, plot_hst, label_objects
from ref_samples import load_sdss, load_gnirs
from civ_distance_ext import civ_distance, EXTEND_CURVE
from make_fig1_L1450 import lbol_to_lognu1450

cosmo = FlatLambdaCDM(Om0=0.3, H0=70)
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True


def log_lw1(w1mag, z):
    """Trevor's L_W1 (erg/s): 4 pi D_L^2 nu F_nu at the W1 band (3.368 um)."""
    DLcm = cosmo.luminosity_distance(z).value * 3.086e24          # Mpc to cm
    Fnu = 309.540 * 10.**(-np.asarray(w1mag) / 2.5) * 1e-23       # Jy -> cgs
    nuFnu = Fnu * (2.998e8 / 3.368e-6)
    return np.log10(4. * np.pi) + np.log10(nuFnu) + 2 * np.log10(DLcm)


def main():
    labeled = "--labeled" in sys.argv
    suffix = "_labeled" if labeled else ""
    data_dir = os.path.join(HERE, "..", "Data")
    out_dir = os.path.join(HERE, "..", "Figures")
    cs = palettable.colorbrewer.qualitative.Dark2_5.mpl_colors
    w1 = pd.read_csv(os.path.join(data_dir, "AllWISE_W1_v23.csv"), dtype={"key": str})

    # ---- HST: adopted sample with W1 ----
    hst = add_logL1450(load_adopted(data_dir), data_dir)
    hw = w1[w1["sample"] == "HST"].set_index("key")["w1mpro"]
    hst["W1"] = hst["row_num"].astype(str).map(hw)
    hst = hst[hst["W1"].notna()].reset_index(drop=True)
    hst["logLW1"] = log_lw1(hst["W1"].values, hst["z"].values)
    c4blue_hst = hst["CIV_blue"].values
    c4dist_hst = civ_distance(c4blue_hst, np.log10(hst["CIV_EW"].values))

    # ---- SDSS (same selection as panel a) with W1 ----
    sdss = load_sdss(data_dir)
    sdss["key"] = sdss["SDSS_NAME"].astype(str).str.strip("b'")
    sdss = sdss.merge(w1[w1["sample"] == "SDSS"][["key", "w1mpro"]], on="key", how="left")
    sdss = sdss[sdss["w1mpro"].notna()]
    sdss["logLW1"] = log_lw1(sdss["w1mpro"].values, sdss["z_paul"].values)

    # ---- GNIRS-DQS (same selection as panel a) with W1 ----
    gn = load_gnirs(data_dir)
    gw = w1[w1["sample"] == "GNIRS"].set_index("key")["w1mpro"]
    gn["W1"] = gn.index.astype(str).map(gw)
    gn = gn[gn["W1"].notna()]
    gn["logLW1"] = log_lw1(gn["W1"].values, gn["z_paul"].values)
    print("  with W1: HST %d, SDSS %d, GNIRS-DQS %d" % (len(hst), len(sdss), len(gn)))

    # L_bol = 3e45 line: Trevor's route, a straight-line fit of L_W1 on the UV
    # luminosity for SDSS, evaluated at panel a's line
    m, b = np.polyfit(sdss["logLnu1450"].values, sdss["logLW1"].values, 1)
    L_wind = m * lbol_to_lognu1450(np.log10(3e45)) + b
    print("  SDSS fit log L_W1 = %.3f log L_nu(1450) + %.2f; L_bol line at log L_W1 = %.2f"
          % (m, b, L_wind))

    xlabel = r'$\log L_{\rm W1}$ [erg\,s$^{-1}$]'
    lo = min(hst["logLW1"].min(), np.percentile(sdss["logLW1"], 0.1))
    hi = max(hst["logLW1"].max(), np.percentile(sdss["logLW1"], 99.9))
    xlim = (np.floor(lo * 2) / 2 - 0.25, np.ceil(hi * 2) / 2 + 0.25)

    for fig_id, yh, ys, yg, ylab, ylim, out in [
            ("5a", c4blue_hst, sdss["CIV_blue"].values, gn["CIV_blue"].values,
             r'CIV Blueshift [km\,s$^{-1}$]', (-1299.99, 6199),
             "Fig5b_CIV_blue_versus_LW1"),
            ("6a", c4dist_hst, sdss["CIV_Distance"].values, gn["CIV_dist_ICA_tvm"].values,
             'CIV Distance [Scaled Units]', (-0.40 if EXTEND_CURVE else -0.05, 1.25),
             "Fig6b_CIV_dist_versus_LW1")]:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9.5))
        plot_hst(ax, hst["logLW1"].values, yh, hst)
        jit = np.random.default_rng(0).uniform(-0.02, 0.02, len(ys)) if fig_id == "6a" else 0
        plot_contour(sdss["logLW1"].values, ys + jit, c=cs[2], mark="o",
                     nlevels=[0.05, 0.25, 0.50, 0.75, 0.95], ax=ax,
                     linewidths=1.5, s=3, alpha=0.4, label="SDSS", zorder=1)
        ax.plot(gn["logLW1"].values, yg, marker="v", markersize=11, markerfacecolor=cs[2],
                markeredgecolor="k", label="GNIRS-DQS", linestyle="", zorder=1)
        ax.axvline(L_wind, color="b", ls="-", zorder=1)
        ax.set_xlabel(xlabel, fontsize=32)
        ax.set_ylabel(ylab, fontsize=35)
        ax.tick_params(axis='both', which='major', labelsize=27.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if fig_id == "5a":
            ax.legend(frameon=True, labelspacing=0.1, loc="upper left", prop={"size": 16})
        if labeled:
            label_objects(ax, hst["logLW1"].values, yh, hst, fig_id, data_dir)
        plt.tight_layout()
        path = os.path.join(out_dir, out + suffix + ".pdf")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.savefig(path.replace(".pdf", ".png"), dpi=100, bbox_inches='tight')
        off = (yh < ylim[0]) | (yh > ylim[1])
        print("saved %s  (HST outside y limits: %d)" % (path, off.sum()))


if __name__ == "__main__":
    main()
