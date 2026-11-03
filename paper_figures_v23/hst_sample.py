"""
The adopted v23 HST sample, shared by the figure scripts in this folder.

load_adopted(data_dir) returns CIV_measurements_v23.csv restricted to the
adopted sample (CIVgood_manmask 2 or 3; 349 objects), with three columns added:

    logMBH, MBH_source_qual  -- from the old catalogue
                                HST_CIV_Sulentic2007_HSLA2018_finalprops_f2500AP.csv,
                                matched by position within 5 arcsec (NaN if the
                                object is not in the old catalogue)
    rm_classical, rm_other, rm_none -- RM-program membership from cat_flags
    mask_rm, mask_sr, mask_no_mbh -- Trevor's M_BH categories:
                                RM mass (qual < 2), scaling-relation mass
                                (qual >= 2), no mass

add_logL1450(hst, data_dir) adds F1450 (from Data/HST_F1450_v23.csv, made by
extract_f1450.py) and logLnu1450, NaN where the spectrum has no 1450 A flux.
"""

import os
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(Om0=0.3, H0=70)
C_ANG = 2.998e18           # speed of light in Angstrom/s


def designation_to_coord(des):
    """'J000543.05-500654.5' -> SkyCoord."""
    s = des[1:]
    sign = "+" if "+" in s else "-"
    ra, dec = s.split(sign)
    ra = "%sh%sm%ss" % (ra[0:2], ra[2:4], ra[4:])
    dec = "%s%sd%sm%ss" % (sign, dec[0:2], dec[2:4], dec[4:])
    return SkyCoord(ra, dec)


def load_adopted(data_dir):
    hst = pd.read_csv(os.path.join(data_dir, "CIV_measurements_v23.csv"))
    hst = hst[hst["CIVgood_manmask"].isin([2, 3])].reset_index(drop=True)

    # M_BH from the old catalogue, matched by position
    old = pd.read_csv(os.path.join(
        data_dir, "HST_CIV_Sulentic2007_HSLA2018_finalprops_f2500AP.csv"))
    c_new = SkyCoord([designation_to_coord(d) for d in hst["designation"]])
    c_old = SkyCoord(old["_RA"].values * u.deg, old["_DE"].values * u.deg)
    idx, sep, _ = c_new.match_to_catalog_sky(c_old)
    in_old = sep < 5 * u.arcsec
    hst["logMBH"] = np.where(in_old, old["log M_BH"].values[idx], np.nan)
    hst["MBH_source_qual"] = np.where(in_old, old["MBH_source_qual"].values[idx], np.nan)

    # Trevor's three categories
    maskMBH = ~np.isnan(hst["logMBH"].values)
    hst["mask_rm"] = maskMBH & (hst["MBH_source_qual"].values < 2)
    hst["mask_sr"] = maskMBH & (hst["MBH_source_qual"].values >= 2)
    hst["mask_no_mbh"] = ~maskMBH
    print("  HST adopted: %d, in old catalogue %d, with M_BH %d (RM %d)"
          % (len(hst), in_old.sum(), maskMBH.sum(), hst["mask_rm"].sum()))

    # RM-program membership, from Table 1's catalogue bitmask (cat_flags), which
    # covers all objects -- unlike the M_BH columns above, which exist only for
    # objects in the old catalogue.  Bits: Kaspi+00 = 1, Barth+15 = 2,
    # Du+15 (SEAMBH) = 4, Bentz & Katz 15 = 8, Shen+16 (SDSS-RM) = 16.
    f = hst["cat_flags"].fillna(0).astype(int).values
    hst["rm_classical"] = (f & (1 | 2 | 8)) > 0
    hst["rm_other"] = ((f & (4 | 16)) > 0) & ~hst["rm_classical"].values
    hst["rm_none"] = ~hst["rm_classical"].values & ~hst["rm_other"].values
    print("  RM programs: classical %d, SDSS-RM/SEAMBH only %d, none %d"
          % (hst["rm_classical"].sum(), hst["rm_other"].sum(), hst["rm_none"].sum()))
    return hst


def add_logL1450(hst, data_dir):
    """log L_nu(1450 A) in erg/s/Hz, with Trevor's formula from make_fig1.py."""
    # F1450: observed-frame f_lambda at rest 1450 A, erg/s/cm2/A
    f1450 = pd.read_csv(os.path.join(data_dir, "HST_F1450_v23.csv"))
    hst = hst.merge(f1450, on="row_num", how="left")

    z = hst["z"].values
    DLcm = cosmo.luminosity_distance(z).value * 3.086e24  # Mpc to cm

    # f_nu = f_lambda * lambda^2 / c, then L_nu = 4 pi D_L^2 f_nu (1+z)
    fnu1450 = hst["F1450"].values * 1450. / (C_ANG / 1450.)
    hst["logLnu1450"] = (np.log10(4. * np.pi) + np.log10(fnu1450) +
                         2 * np.log10(DLcm) + np.log10(1. + z))
    return hst


def plot_hst(ax, x, y, hst):
    """HST symbols by RM program, shared by Figures 1, 4, 5a and 6a.

    hst is the table from load_adopted (rows aligned with x and y).  zorder:
    "other" triangles at the bottom, classical RM above them, and the open
    SDSS-RM/SEAMBH circles on top so none is hidden.
    """
    import palettable
    cs = palettable.colorbrewer.qualitative.Dark2_5.mpl_colors
    m = hst["rm_classical"].values
    ax.plot(x[m], y[m], marker="o", markersize=15, markerfacecolor=cs[3],
            markeredgecolor="k", linestyle="", zorder=3,
            label="HST, classical RM (%d)" % m.sum())
    m = hst["rm_other"].values
    ax.plot(x[m], y[m], marker="o", markersize=13.5, markerfacecolor="w",
            markeredgecolor=cs[3], mew=1.8, linestyle="", zorder=4,
            label="HST, SDSS-RM/SEAMBH (%d)" % m.sum())
    m = hst["rm_none"].values
    ax.plot(x[m], y[m], marker="^", markersize=16.5, markerfacecolor=cs[4],
            markeredgecolor="k", linestyle="", zorder=2,
            label="HST, other (%d)" % m.sum())
