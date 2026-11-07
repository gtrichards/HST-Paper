"""Does the rebin invent flux where the native exposure has none?

Found on Mrk 1014.  Its chosen-run exposure swp26578 has NO QUALITY==0 pixel
between 1540 and 1545 A rest, yet the rebinned product carries 2.801e-14 there
-- the largest value anywhere in the C IV line.  Everywhere else native and
rebinned agree to about one per cent.  The invented flux sits on the blue flank,
so it becomes the profile's peak and drags the half-flux point to 1540.2 A, i.e.
a blueshift of +1644 km/s.  GTR, looking at the fit: "The answer is robust, but
is flat out wrong."

The pixels are not silently wrong -- their errors are inflated four- to fivefold,
which is the rebin honestly reporting that little real data went into them, and
that is why masking them barely moves the fit: the error weighting already
discounts them.  What masking CANNOT undo is the SHAPE they impose, because the
ICA reconstruction is fitted to a profile whose blue flank those pixels hold up.
The fix therefore has to be at the rebin, by masking rather than filling.

This measures the scope before anything is changed.  For every IUE object it
compares the rebinned product against the native exposures of its chosen run:

    native_px    QUALITY==0 native pixels inside the band
    rebin_px     unmasked, finite, positive-error rebinned pixels there
    INVENTED     rebin_px > 0 while native_px == 0

Bands are 2 A, a third of IUE's 6 A resolution element, so a gap narrower than
one resolution element is still seen.  Only 1500-1600 A is reported, because a
gap outside C IV cannot move the blueshift; --all widens it to the fitted range.

    python check_gap_fill.py          # the C IV window
    python check_gap_fill.py --all    # 1150-2000 A
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
CHOSEN = os.path.join(HERE, "iue_chosen_run.csv")
REBIN = os.path.join(HERE, "rebin")
OUT = os.path.join(HERE, "iue_gap_fill.csv")

STEP = 2.0


def native_rest(files, z):
    """(rest wavelength, flux) of every QUALITY==0 native pixel of the run."""
    w, f = [], []
    for p in files:
        if not os.path.exists(p):
            continue
        try:
            d = fits.getdata(p, 1)
        except Exception:
            continue
        ww = np.asarray(d["WAVE"][0], float)
        ff = np.asarray(d["FLUX"][0], float)
        ss = np.asarray(d["SIGMA"][0], float)
        qq = np.asarray(d["QUALITY"][0], float)
        g = (qq == 0) & np.isfinite(ff) & np.isfinite(ss) & (ss > 0)
        w.append(ww[g] / (1.0 + z))
        f.append(ff[g])
    if not w:
        return np.array([]), np.array([])
    return np.concatenate(w), np.concatenate(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    lo_lim, hi_lim = (1150.0, 2000.0) if a.all else (1500.0, 1600.0)

    ch = pd.read_csv(CHOSEN)
    rows = []
    for _, r in ch.iterrows():
        name = str(r["common_name"])
        z = float(r["z"])
        files = [x for x in str(r["files"]).split(";") if x]
        nw, nf = native_rest(files, z)
        p = os.path.join(REBIN, "%s_IUE.fits" % name)
        if not os.path.exists(p) or not nw.size:
            continue
        t = Table.read(p)
        w = np.asarray(t["Rest-frame Wavelength"], float)
        f = np.asarray(t["Coadded Flux (Arbitrary Units)"], float)
        e = np.asarray(t["Coadded Flux Errors"], float)
        m = np.asarray(t["Bad Pixel Mask"], float)
        g = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)

        bands, worst = [], None
        for lo in np.arange(lo_lim, hi_lim, STEP):
            hi = lo + STEP
            n_nat = int(((nw >= lo) & (nw < hi)).sum())
            sel = g & (w >= lo) & (w < hi)
            n_reb = int(sel.sum())
            if n_reb > 0 and n_nat == 0:
                bands.append((lo, hi, n_reb, float(np.nanmedian(f[sel]))))
        if bands:
            # how much of the C IV line sits on invented pixels?
            civ = g & (w >= 1500) & (w <= 1600)
            loc = float(np.nanmedian(f[civ])) if civ.any() else np.nan
            inv = np.zeros(len(w), bool)
            for lo, hi, _, _ in bands:
                inv |= (w >= lo) & (w < hi)
            inv &= g
            frac = (float(np.nansum(np.clip(f[inv & civ] - loc, 0, None)) /
                          max(np.nansum(np.clip(f[civ] - loc, 0, None)), 1e-30))
                    if civ.any() else np.nan)
            span = "%.1f-%.1f" % (bands[0][0], bands[-1][1])
            rows.append(dict(common_name=name, z=z, n_bands=len(bands),
                             invented_px=int(inv.sum()), span=span,
                             frac_line_flux=round(frac, 3)))
            print("  %-24s %2d band(s) %-16s %4d px   %.1f%% of the C IV line flux"
                  % (name[:24], len(bands), span, int(inv.sum()), 100 * frac))
        else:
            rows.append(dict(common_name=name, z=z, n_bands=0, invented_px=0,
                             span="", frac_line_flux=0.0))

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    bad = d[d.n_bands > 0]
    print()
    print("%d of %d objects have rebinned pixels with NO native data behind them"
          % (len(bad), len(d)))
    if len(bad):
        print("worst by share of the C IV line flux:")
        for _, r in bad.sort_values("frac_line_flux", ascending=False).head(8).iterrows():
            print("   %-24s %5.1f%%  (%d px over %s)"
                  % (r.common_name[:24], 100 * r.frac_line_flux, r.invented_px, r.span))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
