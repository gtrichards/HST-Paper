"""
Figure 1 (v23 sample): log L_1450 versus redshift.

Adapted from Plotting Code/make_fig1.py (Trevor McCaffrey).  What changed:
  * HST sample = the adopted v23 sample: CIV_measurements_v23.csv rows with
    CIVgood_manmask 2 or 3 (349 objects; 345 have a 1450 A flux).
  * HST luminosity is at 1450 A, from Data/HST_F1450_v23.csv (made by
    extract_f1450.py), computed in hst_sample.py.  Same formula Trevor used at 2500 A, including (1+z).
  * HST symbols mark RM-program membership (hst_sample.py, from Table 1's
    catalogue bitmask): classical RM (Kaspi+00, Barth+15, Bentz & Katz 15),
    SDSS-RM / SEAMBH only (Shen+16, Du+15), and not RM.  Trevor's M_BH
    categories were dropped: the masses exist only for objects in the old
    catalogue, so "no M_BH" had come to mean "not in the old catalogue".
  * The four objects whose L1450 is extrapolated (F1450_est in
    HST_F1450_v23.csv) are not marked: tested on the 345 measured objects the
    extrapolation is good to 0.02-0.03 dex (1 sigma), well inside the symbol
    (half-height 0.11-0.12 dex on this axis).
  * SDSS and GNIRS-DQS are interpolated to 1450 A per object
    (ref_samples.py).
  * The secondary L_bol axis is removed (Trevor's bolometric correction is
    for L_2500).  The L_bol = 3e45 erg/s line -- where AGN wind feedback has
    been argued to matter (Veilleux+13; Zakamska & Greene 14) -- is kept,
    placed with Krawczyk et al. (2013) eq. 9 and moved from 2500 to 1450 A
    with the median SDSS continuum: log L_nu(1450) = 29.79 (see
    lbol_to_lognu1450).  The Temple option is
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
from hst_sample import load_adopted, add_logL1450, plot_hst
from ref_samples import load_sdss, load_gnirs

C_ANG = 2.998e18           # speed of light in Angstrom/s


#: median log[lambda L_lambda(2500) / lambda L_lambda(1450)] for the 118,638
#: SDSS quasars with both luminosities (Wu & Shen 2022, via ref_samples.py),
#: the population Krawczyk et al. built their mean SED from.
LAML2500_MINUS_LAML1450 = -0.089


def lbol_to_lognu1450(logLbol):
    """log L_nu(1450 A) [erg/s/Hz] for a given log L_bol [erg/s].

    Krawczyk et al. 2013 (ApJS 206, 4; 10.1088/0067-0049/206/1/4), eq. 9:
        log L_bol = 0.9869 log(nu L_nu at 2500 A) + 1.051
    (no anisotropy correction, as published), inverted for lambda L_lambda at
    2500 A and moved to 1450 A with the median SDSS continuum above.
    For L_bol = 3e45 this gives 29.79.

    For comparison: adding Runnoe et al. 2012's 0.75 anisotropy factor gives
    29.92; Runnoe et al. 2012 eq. 9 at 1450 A gives 29.59.  Trevor's old line
    (Plotting Code/make_fig1.py) used this same Krawczyk relation but converted
    L_nu(2500) with c/1549.48 A instead of c/2500 A, which put it 0.21 dex low.
    """
    log_lamL2500 = (logLbol - 1.051) / 0.9869
    log_lamL1450 = log_lamL2500 - LAML2500_MINUS_LAML1450
    return log_lamL1450 - np.log10(C_ANG / 1450.)       # lambda L_lambda -> L_nu


plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True


def load_hst_data(data_dir):
    # adopted sample (manmask 2 or 3) with RM-program groups and L1450
    hst = add_logL1450(load_adopted(data_dir), data_dir)
    hst = hst[np.isfinite(hst["logLnu1450"])].reset_index(drop=True)
    return {
        'z': hst["z"].values,
        'logL': hst["logLnu1450"].values,
        'table': hst,
    }


def load_sdss_rankine_data(data_dir):
    sdss = load_sdss(data_dir)
    return {'z': sdss["z_paul"].values, 'logL': sdss["logLnu1450"].values}


def load_gnirs_data(data_dir):
    gnirs = load_gnirs(data_dir)
    return {'z': gnirs["z_paul"].values, 'logL': gnirs["logLnu1450"].values}


def create_figure(hst_data, sdss_data, gnirs_data):
    cs = palettable.colorbrewer.qualitative.Dark2_5.mpl_colors
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    plot_hst(ax, hst_data['z'], hst_data['logL'], hst_data['table'])

    # SDSS contours and GNIRS-DQS, both interpolated to 1450 A
    plot_contour(sdss_data['z'], sdss_data['logL'], c=cs[2], mark="o",
                 nlevels=[0.05, 0.25, 0.50, 0.75, 0.95], ax=ax,
                 linewidths=1.5, s=3, alpha=0.4, label="SDSS", zorder=1)
    ax.plot(gnirs_data['z'], gnirs_data['logL'], marker="v", markersize=11,
            markerfacecolor=cs[2], markeredgecolor="k",
            label="GNIRS-DQS", linestyle="", zorder=3)

    # L_bol = 3e45 erg/s, as in Trevor's figure (solid blue, behind the data)
    L_wind = lbol_to_lognu1450(np.log10(3e45))
    ax.plot([-1e5, 1e5], [L_wind, L_wind], "-b", zorder=1)

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
              prop={"size": 16})
    ax.set_xlim(-0.16575, 3.25)
    # Trevor used 27.1-32.5 for L2500; extended down to take in the low-L
    # HST objects (lowest is NGC 4395 at 24.27, after the 2026-11-03 morph-coefficient correction).  Explicit limits are needed
    # because Trevor's plot_contour pads its grid far beyond the data.
    ax.set_ylim(24.0, 32.6)
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
    print("  L_bol = 3e45 line at log L_nu(1450) = %.3f"
          % lbol_to_lognu1450(np.log10(3e45)))
    print("  HST log L1450 range %.2f to %.2f"
          % (hst_data['logL'].min(), hst_data['logL'].max()))

    create_figure(hst_data, sdss_data, gnirs_data)
    out = os.path.join(out_dir, "Fig1_L1450_versus_z.pdf")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace(".pdf", ".png"), dpi=100, bbox_inches='tight')
    print("saved", out)


if __name__ == "__main__":
    main()
