"""
Figure 2 (v23 sample): optical "Eigenvector 1" plane, FWHM(Hbeta) versus R(Fe II).

Adapted from Plotting Code/make_fig2.py (and Notebooks/EV1_optical.ipynb).
What changed:
  * HST sample = the adopted v23 sample (manmask 2 or 3), with the same symbols
    as Figs 1, 4, 5a, 6a (hst_sample.plot_hst).
  * HST optical values from Rakshit et al. (2020) where they exist, otherwise
    Sulentic et al. (2007) (GTR 2026-11-03, "B"): Rakshit's R(Fe II) =
    EW(Fe II, 4434-4684 A) / EW(Hbeta broad) and FWHM(Hbeta broad) are the
    same measurements as the SDSS contours.  --sulentic-first reverses the
    priority ("C") and writes Fig2_EV1_optical_sulentic_first.pdf/png, for
    comparison.  94 adopted objects have neither and are not plotted.
  * SDSS quasars with R(Fe II) = 0 exactly (25%: Fe II not measured, recorded
    as 0) are left out of the contours and the scatter, which they distorted
    into a vertical streak; the number left out is written on the plot.
  * Both sets of values, and the source used, are written to
    Data/HST_optical_EV1_v23.csv; for the objects with both, the comparison is
    printed (for the text).
  * The one Sulentic R(Fe II) upper limit is drawn with a leftward arrow.
  * SDSS contours use Trevor's original plot_contour (trevor_contour.py).
  * --labeled writes ONLY Fig2_EV1_optical_labeled.pdf/png, naming the HST
    objects with the 4 largest R(Fe II) and all with FWHM(Hbeta) > 10000 km/s,
    plus hst_sample.ALWAYS_LABEL (I Zw 1, PDS 456) (GTR 2026-11-03); the plain
    figure is not touched.

Data:   Data/Sulentic2007Tables1_2.fit (VizieR, Sulentic+07 Tables 1-2)
        Data/master_catalog_v22.csv    (hbeta_fwhm_rk20, rfeii_rk20 =
                                        Rakshit+20 FWHMHb-BR, EWFe-OP/EWHb-BR)
        Data/Rakshit2020.fit, Data/gnirsdqs_wRankine_wdr16Lum.csv
Usage:  PYTHONPATH=~/Work/git python make_fig2.py
Output: Figures/Fig2_EV1_optical.pdf/.png, Data/HST_optical_EV1_v23.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import palettable
# NOT `import richardsplot`: the package __init__ is empty (see make_fig1_L1450.py)
from richardsplot import richardsplot as rplot   # noqa: F401
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from trevor_contour import plot_contour   # Trevor's original, see that file
from hst_sample import load_adopted, plot_hst, designation_to_coord, ALWAYS_LABEL

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True


def hst_optical(data_dir, sulentic_first=False):
    """Adopted sample with FWHM(Hbeta) and R(Fe II).

    Rakshit+20 where available, else Sulentic+07; or the reverse when
    sulentic_first is True.
    """
    hst = load_adopted(data_dir)
    c = SkyCoord([designation_to_coord(d) for d in hst["designation"]])

    # Sulentic+07, matched by position within 5 arcsec
    sul = Table.read(os.path.join(data_dir, "Sulentic2007Tables1_2.fit")).to_pandas()
    i, s, _ = c.match_to_catalog_sky(SkyCoord(sul["_RA"].values * u.deg,
                                              sul["_DE"].values * u.deg))
    ok = s.arcsec < 5
    hst["Hb_sul"] = np.where(ok, sul["Hbeta"].values[i], np.nan)
    hst["RFe_sul"] = np.where(ok, sul["RFeII"].values[i], np.nan)
    flag = sul["f_RFeII"].astype(str).str.contains("<").values[i]
    hst["RFe_sul_upper"] = ok & flag             # R(Fe II) is an upper limit

    # Rakshit+20, from the master catalogue (matched by position)
    mc = pd.read_csv(os.path.join(data_dir, "master_catalog_v22.csv"))
    j, s2, _ = c.match_to_catalog_sky(SkyCoord(mc["ra_deg"].values * u.deg,
                                               mc["dec_deg"].values * u.deg))
    assert s2.arcsec.max() < 1
    hst["Hb_rk"] = mc["hbeta_fwhm_rk20"].values[j]
    hst["RFe_rk"] = mc["rfeii_rk20"].values[j]

    # adopted values: the preferred source where both its numbers exist,
    # otherwise the other one
    has_sul = hst["Hb_sul"].notna() & hst["RFe_sul"].notna()
    has_rk = hst["Hb_rk"].notna() & hst["RFe_rk"].notna() & (hst["Hb_rk"] > 0)
    use_sul = has_sul if sulentic_first else (has_sul & ~has_rk)
    use_rk = has_rk & ~use_sul
    hst["opt_source"] = np.where(use_sul, "Sulentic07", np.where(use_rk, "Rakshit20", ""))
    hst["Hb_fwhm"] = np.where(use_sul, hst["Hb_sul"], np.where(use_rk, hst["Hb_rk"], np.nan))
    hst["RFeII"] = np.where(use_sul, hst["RFe_sul"], np.where(use_rk, hst["RFe_rk"], np.nan))
    hst["RFeII_upper"] = use_sul & hst["RFe_sul_upper"]

    # for the text: how the two sources compare where both exist
    b = has_sul & has_rk
    ratio = hst.loc[b, "Hb_rk"] / hst.loc[b, "Hb_sul"]
    dR = hst.loc[b, "RFe_rk"] - hst.loc[b, "RFe_sul"]
    print("  optical values used: Rakshit %d, Sulentic %d, neither %d"
          % (use_rk.sum(), use_sul.sum(), (~use_sul & ~use_rk).sum()))
    print("  overlap (both sources): N = %d" % b.sum())
    print("    FWHM(Hb) Rakshit/Sulentic: median %.2f, 16-84%% %.2f-%.2f"
          % (ratio.median(), ratio.quantile(.16), ratio.quantile(.84)))
    print("    R(FeII) Rakshit-Sulentic:  median %+.2f, 16-84%% %+.2f to %+.2f"
          % (dR.median(), dR.quantile(.16), dR.quantile(.84)))
    return hst


def main():
    import sys
    sulentic_first = "--sulentic-first" in sys.argv
    labeled = "--labeled" in sys.argv
    suffix = ("_sulentic_first" if sulentic_first else "") + ("_labeled" if labeled else "")
    data_dir = os.path.join(HERE, "..", "Data")
    out_dir = os.path.join(HERE, "..", "Figures")
    cs = palettable.colorbrewer.qualitative.Dark2_5.mpl_colors

    hst = hst_optical(data_dir, sulentic_first)
    if not sulentic_first:   # the saved table records the adopted (Rakshit-first) choice
      hst[["row_num", "common_name", "Hb_sul", "RFe_sul", "RFe_sul_upper", "Hb_rk",
         "RFe_rk", "opt_source", "Hb_fwhm", "RFeII", "RFeII_upper"]].to_csv(
          os.path.join(data_dir, "HST_optical_EV1_v23.csv"), index=False)
    hst = hst[hst["opt_source"] != ""].reset_index(drop=True)

    # SDSS: Rakshit+20, objects with Fe II, Hbeta EW and FWHM (as in the old code)
    sdss = Table.read(os.path.join(data_dir, "Rakshit2020.fit")).to_pandas()
    sdss = sdss[sdss["EWFe-OP"].notna() & sdss["EWHb-BR"].notna() & sdss["FWHMHb-BR"].notna()]
    RFeII_sdss = (sdss["EWFe-OP"] / sdss["EWHb-BR"]).values
    Hb_sdss = sdss["FWHMHb-BR"].values
    # R(Fe II) = 0 exactly: Fe II not measured; left out (see docstring)
    zero = RFeII_sdss == 0
    n_zero = int(zero.sum())
    RFeII_sdss, Hb_sdss = RFeII_sdss[~zero], Hb_sdss[~zero]

    # GNIRS-DQS (as in the old code)
    gnirs = pd.read_csv(os.path.join(data_dir, "gnirsdqs_wRankine_wdr16Lum.csv"))
    RFeII_gnirs = (gnirs["EW-Fe2"] / gnirs["EW-Hb"]).values
    Hb_gnirs = gnirs["FWHM-Hb"].values
    print("  HST %d, SDSS %d, GNIRS-DQS %d" % (len(hst), len(sdss), len(gnirs)))

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    x, y = hst["RFeII"].values, hst["Hb_fwhm"].values
    plot_hst(ax, x, y, hst)
    # the R(Fe II) upper limit: an arrow pointing to lower R(Fe II)
    for xi, yi in zip(x[hst["RFeII_upper"].values], y[hst["RFeII_upper"].values]):
        ax.annotate("", xy=(xi - 0.35, yi), xytext=(xi, yi), zorder=9,
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    plot_contour(RFeII_sdss, Hb_sdss, c=cs[2], mark="o",
                 nlevels=[0.05, 0.25, 0.50, 0.75, 0.95], ax=ax,
                 linewidths=1.5, s=3, alpha=0.4, label="SDSS", zorder=1)
    ax.plot(RFeII_gnirs, Hb_gnirs, marker="v", markersize=11, markerfacecolor=cs[2],
            markeredgecolor="k", label="GNIRS-DQS", linestyle="", zorder=2)

    if labeled:   # 4 largest R(Fe II), and every FWHM(Hbeta) > 10000 km/s
        lab = hst["RFeII"].rank(ascending=False, method="first").le(4).values | (y > 10000)
        lab = lab | hst["row_num"].isin(ALWAYS_LABEL).values   # I Zw 1, PDS 456
        for xi, yi, n in zip(x[lab], y[lab], hst["common_name"].values[lab]):
            ax.annotate(n, (xi, yi), xytext=(6, 3), textcoords="offset points",
                        fontsize=11, zorder=10)
        print("  labelled %d objects" % lab.sum())
    ax.text(0.97, 0.60, "SDSS with R(Fe\\,{\\sc ii})\\,=\\,0 not shown\n(%s, %d\\%%)"
            % (format(n_zero, ","), round(100 * n_zero / (n_zero + len(Hb_sdss)))),
            transform=ax.transAxes, ha="right", va="top", fontsize=14)
    ax.set_xlabel('R(Fe\\,{\\sc ii})', fontsize=35)
    ax.set_ylabel(r'FWHM(H$\beta$) [km\,s$^{-1}$]', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=27.5)
    ax.legend(frameon=True, labelspacing=0.1, loc="upper right", prop={"size": 16})
    ax.set_xlim(-0.3, 5)
    ax.set_ylim(0, 20000)
    plt.tight_layout()

    off = (x < -0.3) | (x > 5) | (y < 0) | (y > 20000)
    print("  HST points outside the axis limits: %d %s"
          % (off.sum(), list(hst.loc[off, "common_name"])))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "Fig2_EV1_optical%s.pdf" % suffix)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace(".pdf", ".png"), dpi=100, bbox_inches='tight')
    print("saved", out)


if __name__ == "__main__":
    main()
