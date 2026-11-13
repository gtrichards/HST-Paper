"""Does a product's error array agree with the data's own noise?

THE CHECK THAT WOULD HAVE CAUGHT MY MISTAKE.  On 2026-11-11 a wrong fix to the
FOS accumulation test averaged cumulative groups, which double-counts photons:
the spectrum came out noisier while the reported errors came out smaller.  I
reported S/N as flux/error and called it an improvement.  GTR saw the figure and
said "I don't understand why the new versions have lower S/N than iter10."

flux/error cannot detect that, because both halves of the ratio moved the wrong
way.  What detects it is comparing the error array against the DATA'S OWN
pixel-to-pixel scatter, which is independent of what the errors claim:

    scatter = rms of the second difference / sqrt(6)   over a line-free band

For a sound product scatter/error is about 1.  Below 1 the errors are overstated
(conservative); ABOVE 1 they are understated, which is the dangerous direction --
it makes residual-per-sigma, the sub-continuum veto and the inverse-variance
weighting all read optimistically.

Run it before and after any change to the reader, on the same objects.  A
rebuild that is right should move products toward 1.0 and must not move any
product away from it.

    python gate_error_scatter.py --snap pipeline_output/archive_pre_nread_...
"""
import argparse
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE_REB = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
LIVE_FITS = os.path.join(HERE, "pipeline_output", "chosen_visit", "fits")

#: Line-free bands to measure the noise in, rest frame.  Chosen to avoid Si IV,
#: C IV, C III] and Mg II; the widest available is used.
BANDS = ((1620.0, 1700.0), (1250.0, 1350.0), (1430.0, 1480.0), (1950.0, 2200.0))


def ratio(path):
    """scatter/error in the widest usable line-free band, and which band."""
    try:
        t = Table.read(path)
    except Exception:
        return np.nan, ""
    w = np.asarray(t["Rest-frame Wavelength"], float)
    f = np.asarray(t["Coadded Flux (Arbitrary Units)"], float)
    e = np.asarray(t["Coadded Flux Errors"], float)
    m = np.asarray(t["Bad Pixel Mask"], float)
    g = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    best, lab, n_best = np.nan, "", 0
    for lo, hi in BANDS:
        s = g & (w >= lo) & (w <= hi)
        if s.sum() < 40:
            continue
        x = f[s]
        d2 = x[2:] - 2 * x[1:-1] + x[:-2]
        sc = np.nanstd(d2) / np.sqrt(6.0)
        er = np.nanmedian(e[s])
        if er > 0 and s.sum() > n_best:
            best, lab, n_best = sc / er, "%d-%d" % (lo, hi), int(s.sum())
    return best, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    idx = [int(x) for x in open(os.path.join(a.snap, "INDICES.txt")).read().split()]

    rows = []
    for i in idx:
        for d in glob.glob(os.path.join(LIVE_FITS, "%03d_*" % i)):
            recs = (sorted(glob.glob(os.path.join(d, "records", "BEST_*.json")))
                    or sorted(glob.glob(os.path.join(d, "records", "*.json"))))
            if not recs:
                continue
            nm = json.load(open(recs[0])).get("name", "")
            before, b1 = ratio(os.path.join(a.snap, "rebin", "%s.fits" % nm))
            after, b2 = ratio(os.path.join(LIVE_REB, "%s.fits" % nm))
            rows.append(dict(order=i, product=nm, band=b2 or b1,
                             before=before, after=after,
                             d_abs=(abs(after - 1) - abs(before - 1))
                             if np.isfinite(after) and np.isfinite(before) else np.nan))
            break
    d = pd.DataFrame(rows)
    ok = d[d.before.notna() & d.after.notna()]
    print("ERROR-vs-SCATTER GATE   (scatter/error; 1.0 = errors match the data's own noise)")
    print("  products measured on both sides: %d of %d" % (len(ok), len(d)))
    print()
    for lab, col in (("BEFORE", "before"), ("AFTER ", "after")):
        v = ok[col].values
        print("  %s  median %.2f   within 20%% of 1.0: %3d   errors UNDERSTATED >1.2: %3d   "
              "overstated <0.8: %3d"
              % (lab, np.median(v), int((np.abs(v - 1) <= 0.2).sum()),
                 int((v > 1.2).sum()), int((v < 0.8).sum())))
    print()
    better = ok[ok.d_abs < -0.02]
    worse = ok[ok.d_abs > 0.02]
    print("  moved TOWARD 1.0 : %d" % len(better))
    print("  moved AWAY       : %d   <-- any of these is a reason to stop" % len(worse))
    if len(worse):
        print()
        print("  %-5s %-28s %8s %8s %8s" % ("order", "product", "before", "after", "band"))
        for _, r in worse.sort_values("d_abs", ascending=False).iterrows():
            print("  %-5d %-28s %8.2f %8.2f %8s"
                  % (r.order, str(r["product"])[:28], r.before, r.after, r.band))
    if a.csv:
        d.to_csv(a.csv, index=False)
        print()
        print("wrote %s" % a.csv)
    return 1 if len(worse) else 0


if __name__ == "__main__":
    raise SystemExit(main())
