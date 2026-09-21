"""Sanity-check every rebinned spectrum for the ways this pipeline has
silently produced wrong data.

Each test here exists because something got through unnoticed:

  zero_run_in_window   a run of exactly-zero flux inside 1500-1600 A.  Dead
                       detector-edge pixels have zero flux but a non-zero
                       error, so the err==0 mask misses them; concatenating
                       COS segments dropped such pixels into the middle of the
                       line and Mrk 290 was reported with EW 0.0 A and a
                       blueshift of -9767 km/s.
  not_monotonic        wavelengths out of order.  Detector segments are not
                       stored in wavelength order and everything downstream
                       assumes they are.
  lattice_too_long     more points than the log lattice can hold between the
                       spectrum's own limits, which means duplicated
                       wavelengths.  NGC 985 reached 20923 points and broke
                       the morph against its 11177-point reference continuum.
  wave_below_900       an unphysical wavelength solution.  COS G140L writes
                       segment FUVB from -28 A with clean DQ flags.
  no_good_pixels       nothing usable at all.
  civ_all_masked       the C IV window is covered but entirely flagged.

Reports one row per spectrum that fails a test; silence is the good outcome.
"""
import os
import sys

import numpy as np
import pandas as pd
from astropy.table import Table

from rebin_path import REBIN

W, F, E, MK = ("Rest-frame Wavelength", "Coadded Flux (Arbitrary Units)",
               "Coadded Flux Errors", "Bad Pixel Mask")
LO, HI = 1500.0, 1600.0
DEX = 1e-4          # the log lattice step, 69.03 km/s
MIN_RUN = 5         # a run of zeros this long inside the window is suspicious


def longest_zero_run(flux, sel):
    """Longest run of exactly-zero flux among the selected pixels."""
    z = (flux[sel] == 0.0)
    if not z.any():
        return 0
    best = run = 0
    for v in z:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def check(path):
    t = Table.read(path)
    w = np.asarray(t[W], float)
    f = np.asarray(t[F], float)
    e = np.asarray(t[E], float)
    m = np.asarray(t[MK])
    out = {}
    if len(w) == 0:
        return {"empty": True}

    good = (m == 0) & (e > 0) & np.isfinite(f)
    out["npix"] = len(w)
    out["ngood"] = int(good.sum())
    if good.sum() == 0:
        out["no_good_pixels"] = True

    if np.any(np.diff(w) < 0):
        out["not_monotonic"] = int((np.diff(w) < 0).sum())

    if w.min() > 0:
        span = int(round((np.log10(w.max()) - np.log10(w.min())) / DEX)) + 1
        if len(w) > span * 1.05:
            out["lattice_too_long"] = "%d px for a %d-point lattice" % (len(w), span)

    if w.min() < 900.0:
        out["wave_below_900"] = round(float(w.min()), 1)

    win = (w >= LO) & (w <= HI)
    if win.sum():
        run = longest_zero_run(f, win & good)
        if run >= MIN_RUN:
            out["zero_run_in_window"] = int(run)
        if (win & good).sum() == 0:
            out["civ_all_masked"] = int(win.sum())
    return out


def main():
    files = sorted(x for x in os.listdir(REBIN) if x.endswith(".fits"))
    rows = []
    for fn in files:
        try:
            r = check(os.path.join(REBIN, fn))
        except Exception as exc:
            r = {"read_error": str(exc)[:80]}
        flags = {k: v for k, v in r.items() if k not in ("npix", "ngood")}
        if flags:
            rows.append(dict(spectrum=fn[:-5], npix=r.get("npix"), ngood=r.get("ngood"),
                             **flags))
    print("checked %d spectra in %s" % (len(files), REBIN))
    if not rows:
        print("no spectrum failed any check")
        return 0
    d = pd.DataFrame(rows)
    d.to_csv("pipeline_output/final/tree_audit.csv", index=False)
    for c in d.columns:
        if c in ("spectrum", "npix", "ngood"):
            continue
        n = int(d[c].notna().sum())
        if n:
            print("  %-22s %d spectra" % (c, n))
    print("-> pipeline_output/final/tree_audit.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
