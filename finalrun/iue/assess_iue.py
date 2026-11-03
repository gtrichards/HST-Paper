"""Is there a measurable C IV line in the IUE data for each manmask-1 object?

This is the "determine if some really shouldn't be included" half of GTR's
plan, done the same way the HST pass judged objects: look at what is actually
in the C IV window rather than at whether a file exists.

For each object every MXLO exposure covering C IV is read, shifted to the rest
frame with the adopted redshift, and co-added by inverse-variance weighting.
IUE quality flags are used: the MXLO QUALITY column is a bit mask and anything
non-zero is excluded, as is any pixel with a non-positive error.

Reported per object:
  n_exp        exposures that cover the C IV window
  civ_px       usable rest-frame pixels in 1500-1600 A
  sn_civ       median per-pixel signal-to-noise there
  sn_coadd     the same for the inverse-variance co-add
  peak_over_err  the line peak above a local continuum, in units of the
                 median error -- the test that decided several HST objects
  anchor_px    usable pixels in the 1445-1465 A luminosity window

Thresholds used in the HST pass, for reference: per-pixel S/N below about 1-2
is the hollow-residual regime where the error bars swallow any smooth curve,
and a line peak below about 1 times the median error is not a detection.
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = "/Users/gtr/Work/git/HST-Paper/Data/CIV_measurements_v23.csv"
OUT = os.path.join(HERE, "iue_assessment.csv")
ALREADY = {"3C 120", "MCG8-11-11", "PG1310-1051"}


def safe(name):
    return "".join(c if (c.isalnum() or c in "+-._") else "_" for c in str(name))


def read_one(path, z):
    with fits.open(path) as h:
        d = h[1].data
        w = np.asarray(d["WAVE"][0], float)
        f = np.asarray(d["FLUX"][0], float)
        s = np.asarray(d["SIGMA"][0], float)
        q = np.asarray(d["QUALITY"][0], int)
    good = (q == 0) & np.isfinite(f) & np.isfinite(s) & (s > 0)
    return w / (1.0 + z), f, s, good


def main():
    tab = pd.read_csv(TABLE)
    m1 = tab[tab["CIVgood_manmask"] == 1].sort_values("row_num")
    rows = []
    for _, r in m1.iterrows():
        name = str(r["common_name"]); z = float(r["z"])
        d = os.path.join(HERE, "spectra", safe(name))
        files = sorted(set(glob.glob(os.path.join(d, "**", "*mxlo_vo.fits"), recursive=True)))
        specs = []
        for p in files:
            try:
                w, f, s, g = read_one(p, z)
            except Exception:
                continue
            if ((w >= 1500) & (w <= 1600) & g).sum() >= 5:
                specs.append((w, f, s, g))
        if not specs:
            rows.append(dict(row_num=int(r["row_num"]), common_name=name, z=z,
                             n_exp=0, civ_px=0, sn_civ=np.nan, sn_coadd=np.nan,
                             peak_over_err=np.nan, anchor_px=0,
                             in_amogh=int(name in ALREADY)))
            print("  %-14s no exposure reaches C IV" % name, flush=True)
            continue
        # per-exposure S/N in the C IV window
        sns = []
        for w, f, s, g in specs:
            k = g & (w >= 1500) & (w <= 1600)
            if k.sum(): sns.append(np.median(f[k] / s[k]))
        # inverse-variance co-add on a common grid
        grid = np.arange(1450.0, 1620.0, 1.5)
        num = np.zeros(grid.size); den = np.zeros(grid.size)
        for w, f, s, g in specs:
            if g.sum() < 5: continue
            fi = np.interp(grid, w[g], f[g], left=np.nan, right=np.nan)
            si = np.interp(grid, w[g], s[g], left=np.nan, right=np.nan)
            ok = np.isfinite(fi) & np.isfinite(si) & (si > 0)
            num[ok] += fi[ok] / si[ok] ** 2
            den[ok] += 1.0 / si[ok] ** 2
        co = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        ce = np.where(den > 0, 1.0 / np.sqrt(np.where(den > 0, den, 1)), np.nan)
        kc = np.isfinite(co) & np.isfinite(ce) & (ce > 0) & (grid >= 1500) & (grid <= 1600)
        sn_co = float(np.median(co[kc] / ce[kc])) if kc.sum() else np.nan
        # line peak above a local continuum, in units of the median error
        poe = np.nan
        kb = np.isfinite(co) & (grid >= 1450) & (grid < 1500)
        kr = np.isfinite(co) & (grid > 1590) & (grid <= 1615)
        if kc.sum() > 5 and kb.sum() > 2 and kr.sum() > 2:
            cb, cr = np.median(co[kb]), np.median(co[kr])
            cont = np.interp(grid[kc], [np.median(grid[kb]), np.median(grid[kr])], [cb, cr])
            poe = float(np.nanmax(co[kc] - cont) / np.nanmedian(ce[kc]))
        anchor = 0
        for w, f, s, g in specs:
            anchor = max(anchor, int((g & (w >= 1445) & (w <= 1465)).sum()))
        rows.append(dict(row_num=int(r["row_num"]), common_name=name, z=z,
                         n_exp=len(specs),
                         civ_px=int(sum(int((g & (w >= 1500) & (w <= 1600)).sum()) for w, f, s, g in specs)),
                         sn_civ=float(np.median(sns)) if sns else np.nan,
                         sn_coadd=sn_co, peak_over_err=poe, anchor_px=anchor,
                         in_amogh=int(name in ALREADY)))
        print("  %-14s %2d exp | S/N/exp %5.2f | co-add S/N %5.2f | peak/err %5.2f | 1450 px %2d%s"
              % (name, len(specs), rows[-1]["sn_civ"], sn_co, poe, anchor,
                 "   <- in Amogh's set" if name in ALREADY else ""), flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("\n%d objects -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
