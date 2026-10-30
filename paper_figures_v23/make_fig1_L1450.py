"""
Figure 1 (v23 sample): log L_1450 versus redshift.

Adapted from Plotting Code/make_fig1.py (Trevor McCaffrey).  What changed:
  * HST sample = the adopted v23 sample: CIV_measurements_v23.csv rows with
    CIVgood_manmask 2 or 3 (349 objects; 345 have a 1450 A flux).
  * HST luminosity is at 1450 A, from Data/HST_F1450_v23.csv (made by
    extract_f1450.py), computed in hst_sample.py.  Same formula Trevor used at 2500 A, including (1+z).
  * M_BH categories still come from the old catalogue
    (HST_CIV_Sulentic2007_HSLA2018_finalprops_f2500AP.csv), matched by
    position within 5 arcsec (hst_sample.py).  Objects not in it are plotted as "no M_BH".
  * SDSS and GNIRS-DQS have no 1450 A luminosity; their 1350 A luminosity
    (LOGL1350_dr16) is plotted instead and labelled as such.  For a typical
    quasar slope the difference is ~0.02 dex.
  * The secondary L_bol axis and the L_bol = 3e45 line are removed, because
    Trevor's bolometric correction is for L_2500.  The Temple option is
    removed because Temple+21 gives L3000, not L1450.
  * SDSS contours use Trevor's original plot_contour (trevor_contour.py).

Usage:
    PYTHONPATH=~/Work/git python make_fig1_L1450.py
Output: Figures/Fig1_L1450_versus_z.pdf
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

C_ANG = 2.998e18           # speed of light in Angstrom/s

plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True


def load_hst_data(data_dir):
    # adopted sample (manmask 2 or 3) with Trevor's M_BH categories and L1450
    hst = add_logL1450(load_adopted(data_dir), data_dir)
    hst = hst[np.isfinite(hst["logLnu1450"])].reset_index(drop=True)
    # F1450_est marks the four objects whose 1450 A continuum normalisation
    # could not be measured and was extrapolated (extract_f1450.py).  The
    # Scaling-Relation group already uses a white face, so "estimated" cannot
    # also be open; it is drawn as a ring around whichever symbol the object
    # already has.
    est = (hst["F1450_est"].values == 1) if "F1450_est" in hst.columns \
        else np.zeros(len(hst), bool)
    return {
        'z': hst["z"].values,
        'logL': hst["logLnu1450"].values,
        'mask_rm': hst["mask_rm"].values,
        'mask_sr': hst["mask_sr"].values,
        'mask_no_mbh': hst["mask_no_mbh"].values,
        'est': est
    }


def load_sdss_rankine_data(data_dir):
    sdss_tbl = Table.read(os.path.join(
        data_dir, "Rankine20_CIV_HeII_wDelGI_wdr16luminosity.fits"))
    names = [n for n in sdss_tbl.colnames if len(sdss_tbl[n].shape) <= 1]
    sdss = sdss_tbl[names].to_pandas()
    sdss = sdss[(sdss["LOGL1350_dr16"] > 0) & (sdss["good"] == True)]

    # LOGL1350_dr16 is log(lambda L_lambda); convert to L_nu
    logLnu1350_sdss = sdss["LOGL1350_dr16"].values - np.log10(C_ANG / 1350.)
    return {'z': sdss["z_paul"].values, 'logL': logLnu1350_sdss}


def load_gnirs_data(data_dir):
    gnirs = pd.read_csv(os.path.join(data_dir, "gnirsdqs_wRankine_wdr16Lum.csv"))
    gnirs = gnirs[(gnirs["good"] == True) & (gnirs["LOGL1350_dr16"].values > 0)]
    logLnu1350_gnirs = gnirs["LOGL1350_dr16"].values - np.log10(C_ANG / 1350.)
    return {'z': gnirs["z_paul"].values, 'logL': logLnu1350_gnirs}


def create_figure(hst_data, sdss_data, gnirs_data):
    cs = palettable.colorbrewer.qualitative.Dark2_5.mpl_colors
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    # HST sample (same symbols as Trevor's figure)
    ax.plot(hst_data['z'][hst_data['mask_rm']],
            hst_data['logL'][hst_data['mask_rm']],
            marker="o", markersize=15, markerfacecolor=cs[3],
            markeredgecolor="k", label="HST, RM $M_{BH}$",
            linestyle="", zorder=3)
    ax.plot(hst_data['z'][hst_data['mask_sr']],
            hst_data['logL'][hst_data['mask_sr']],
            marker="o", markersize=13.5, markerfacecolor="w",
            markeredgecolor=cs[3], mew=1.8,
            label="HST, Scaling Relation $M_{BH}$", linestyle="", zorder=2)
    ax.plot(hst_data['z'][hst_data['mask_no_mbh']],
            hst_data['logL'][hst_data['mask_no_mbh']],
            marker="^", markersize=16.5, markerfacecolor=cs[4],
            markeredgecolor="k", label="HST, no $M_{BH}$",
            linestyle="", zorder=2)

    # the four estimated L1450 values, ringed so they are visibly not measured
    if hst_data['est'].any():
        ax.plot(hst_data['z'][hst_data['est']],
                hst_data['logL'][hst_data['est']],
                marker="o", markersize=24, markerfacecolor="none",
                markeredgecolor="k", mew=1.4, linestyle="", zorder=4,
                label=r"$L_{1450}$ extrapolated (%d)" % hst_data['est'].sum())

    # SDSS contours and GNIRS-DQS (both at 1350 A)
    plot_contour(sdss_data['z'], sdss_data['logL'], c=cs[2], mark="o",
                 nlevels=[0.05, 0.25, 0.50, 0.75, 0.95], ax=ax,
                 linewidths=1.5, s=3, alpha=0.4, label=r"SDSS ($L_{1350}$)", zorder=1)
    ax.plot(gnirs_data['z'], gnirs_data['logL'], marker="v", markersize=11,
            markerfacecolor=cs[2], markeredgecolor="k",
            label=r"GNIRS-DQS ($L_{1350}$)", linestyle="", zorder=3)

    ax.set_xlabel('Redshift', fontsize=35)
    # The Angstrom is OUT of the subscript: \AA at subscript size in Computer
    # Modern sets the ring well clear of the A and reads as a degree sign.
    # Written as L_nu(1450 A) rather than L_{1450A} because the axis is
    # erg/s/Hz -- this is the specific luminosity per unit frequency, and the
    # old label left it ambiguous whether it meant nu L_nu.
    ax.set_ylabel(r'$\log L_\nu(1450\,\mathrm{\AA})$ [erg\,s$^{-1}$\,Hz$^{-1}$]',
                  fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=27.5)
    ax.legend(frameon=True, labelspacing=0.1, loc="lower right",
              prop={"size": 20})
    ax.set_xlim(-0.16575, 3.25)
    # Trevor used 27.1-32.5 for L2500; extended down to take in the low-L
    # HST objects (lowest is NGC 4395 at 24.83).  Explicit limits are needed
    # because Trevor's plot_contour pads its grid far beyond the data.
    ax.set_ylim(24.5, 32.6)
    plt.tight_layout()
    return fig, ax


def main():
    data_dir = os.path.join(HERE, "..", "Data")
    out_dir = os.path.join(HERE, "..", "Figures")
    os.makedirs(out_dir, exist_ok=True)

    hst_data = load_hst_data(data_dir)
    sdss_data = load_sdss_rankine_data(data_dir)
    gnirs_data = load_gnirs_data(data_dir)
    print("  HST %d, SDSS %d, GNIRS-DQS %d"
          % (len(hst_data['z']), len(sdss_data['z']), len(gnirs_data['z'])))
    print("  HST log L1450 range %.2f to %.2f"
          % (hst_data['logL'].min(), hst_data['logL'].max()))

    create_figure(hst_data, sdss_data, gnirs_data)
    out = os.path.join(out_dir, "Fig1_L1450_versus_z.pdf")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace(".pdf", ".png"), dpi=100, bbox_inches='tight')
    print("saved", out)


if __name__ == "__main__":
    main()
