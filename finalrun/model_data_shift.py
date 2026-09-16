"""
Is the ICA reconstruction systematically shifted relative to the data?

Measures the velocity offset between model and data independently in windows
around Si IV, C IV, C III] and Mg II, by cross-correlation.

Cross-correlation is used rather than centroids for two reasons. It compares
the model to the *data*, never to a laboratory wavelength, so it is unaffected
by C III] being a blend of C III] 1908.73, Si III] 1892.03 and Al III 1857.40 --
whatever the blend's true centroid is, model and data should agree on it. And
it is insensitive to the continuum level and to line asymmetry, both of which
made earlier centroid measurements unreliable.

A single coherent offset across all four lines behaves like a redshift error.
Offsets that differ line to line are a deficiency of the component fit.

Two summary statistics are reported per object. "median_dv" is the median over
every line measured; "anchor_dv" uses only C III] and Mg II. See ANCHOR_LINES
for why the distinction matters -- they are not interchangeable, and the second
is the one implied by how a fit is judged good in this project.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")

from ica.manual_fix_gui import _make_processor
from ica.manual_fix_store import resolve_override

C = 299792.458
SPLICED = "/Users/gtr/Dropbox/HST/RebinnedSpec_HSTSDSS_test15/spliced"
OVERRIDES = "/Users/gtr/Dropbox/HST/splice_test_overrides.json"

# vacuum rest wavelengths, with the window used for each
LINES = [
    ("SiIV+OIV]", 1399.8, 1360.0, 1440.0),
    ("CIV",       1549.48, 1500.0, 1600.0),
    ("CIII]",     1908.73, 1860.0, 1960.0),
    ("MgII",      2799.94, 2740.0, 2860.0),
]

# Lines that may legitimately set the alignment. C IV is excluded deliberately:
# its offset from the model IS the wind measurement this paper reports, so
# folding it into a shift that is then applied back to the data would subtract
# part of the signal from itself. Si IV is excluded for the same reason, more
# weakly -- it is also an outflow line and carries some of the same blueshift.
# C III] and Mg II are the low-ionisation anchors that should agree with the
# model when the redshift is right, which is exactly the condition being tested.
ANCHOR_LINES = ("CIII]", "MgII")

VGRID = np.arange(-2000.0, 2001.0, 10.0)


def xcorr_shift(wave, data, model, lo, hi):
    """Velocity shift maximising correlation between model and data in [lo,hi].

    Positive means the MODEL must be shifted redward to match the data, i.e.
    the model currently sits blueward of the data.
    """
    m = (wave >= lo) & (wave <= hi) & np.isfinite(data) & np.isfinite(model)
    if m.sum() < 40:
        return np.nan, 0
    w, d, mo = wave[m], data[m], model[m]
    # remove a linear continuum from each so we correlate line shape only
    p = np.polyfit(w, d, 1); d = d - np.polyval(p, w)
    p = np.polyfit(w, mo, 1); mo = mo - np.polyval(p, w)
    if np.std(d) == 0 or np.std(mo) == 0:
        return np.nan, m.sum()
    best, bestv = -2.0, np.nan
    for v in VGRID:
        shifted = np.interp(w, wave * (1.0 + v / C), model, left=np.nan, right=np.nan)
        g = np.isfinite(shifted)
        if g.sum() < 40:
            continue
        s = shifted[g] - np.polyval(np.polyfit(w[g], shifted[g], 1), w[g])
        dd = d[g] - d[g].mean()
        ss = s - s.mean()
        den = np.sqrt(np.sum(dd ** 2) * np.sum(ss ** 2))
        if den <= 0:
            continue
        r = float(np.sum(dd * ss) / den)
        if r > best:
            best, bestv = r, v
    return bestv, m.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Default to every spliced spectrum. Object names contain spaces
    # ("3C 196_STIS"), so enumerating them here avoids shell word-splitting.
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    if args.objects is None:
        import glob
        args.objects = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(SPLICED, "*.fits"))) \
            if args.all else ["3C345_FOS"]
    summary = []

    ov_all = json.load(open(OVERRIDES)) if os.path.exists(OVERRIDES) else {}
    proc = _make_processor(SPLICED)

    for name in args.objects:
        ov = resolve_override(name, overrides=ov_all)
        try:
            res = proc.fit_for_gui(
                name, mask_ranges=ov["mask_ranges"], mask_pixels=ov["mask_pixels"],
                comps_use=ov["comps_use"], unmask_ranges=ov["unmask_ranges"],
                unmask_pixels=ov["unmask_pixels"])
        except Exception as exc:
            print("\n=== %s  FIT FAILED: %s ===" % (name, type(exc).__name__))
            continue
        wave, flux = res["wave_arb"], res["flux_arb"]
        model = np.interp(wave, res["wave_ica"], res["flux_ica"],
                          left=np.nan, right=np.nan)

        print("\n=== %s  (z used = %.6f) ===" % (name, res["z"]))
        print("%-12s %10s %8s   %s" % ("line", "dv km/s", "npix",
                                       "(+ve = model sits blueward of data)"))
        vals = []
        for lab, rest, lo, hi in LINES:
            v, n = xcorr_shift(wave, flux, model, lo, hi)
            vals.append(v)
            print("%-12s %10s %8d" % (lab, "%+.0f" % v if np.isfinite(v) else "--", n))
        good = [v for v in vals if np.isfinite(v)]
        anchors = [v for (lab, _r, _lo, _hi), v in zip(LINES, vals)
                   if lab in ANCHOR_LINES and np.isfinite(v)]
        anch = float(np.median(anchors)) if anchors else float("nan")
        if len(good) >= 2:
            print("%-12s %10s   spread %.0f km/s"
                  % ("median", "%+.0f" % np.median(good), max(good) - min(good)))
            print("%-12s %10s   (%s only)"
                  % ("anchors", "%+.0f" % anch if np.isfinite(anch) else "--",
                     ", ".join(ANCHOR_LINES)))
            summary.append((name, np.median(good), max(good) - min(good), vals, anch))

    if summary:
        print("\n=== summary: model-vs-data shift (+ve = model blueward) ===")
        print("%-30s %9s %9s %9s" % ("object", "median", "spread", "anchors"))
        for n, med, spr, _, anch in summary:
            print("%-30s %+9.0f %9.0f %9s"
                  % (n[:30], med, spr,
                     "%+.0f" % anch if np.isfinite(anch) else "--"))
        meds = np.array([m for _, m, _, _, _ in summary])
        print("\nacross %d objects: median %+.0f km/s, scatter %.0f, "
              "fraction positive %.0f%%"
              % (len(meds), np.median(meds), meds.std(), 100 * (meds > 0).mean()))

        # How much the choice of statistic actually matters. 69 km/s is the
        # grid quantum the C IV blueshift is reported in, so a difference below
        # that cannot move the published number at all.
        anchs = np.array([a for _, _, _, _, a in summary])
        diff = np.abs(meds - anchs)
        diff = diff[np.isfinite(diff)]
        if diff.size:
            print("median |median_dv - anchor_dv| = %.0f km/s; %d of %d objects "
                  "differ by >= 69 km/s (one C IV grid quantum)"
                  % (np.median(diff), int((diff >= 69).sum()), diff.size))
        if args.csv:
            import csv as _csv
            with open(args.csv, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["object", "median_dv", "anchor_dv", "spread"]
                           + [l[0] for l in LINES])
                for n, med, spr, vals, anch in summary:
                    w.writerow([n, round(med, 1),
                                ("" if not np.isfinite(anch) else round(anch, 1)),
                                round(spr, 1)]
                               + [("" if not np.isfinite(v) else round(v, 1)) for v in vals])
            print("-> %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
