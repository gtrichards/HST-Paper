"""Does the reconstruction sit on the data where the luminosities are read?

The three monochromatic luminosities are taken from the ICA reconstruction over
1445-1465, 1700-1705 and 2490-2510 A.  A fit can satisfy every other test -- the
veto, the line residuals, the C III] and Mg II anchors -- and still be wrong
there, and the error goes straight into the luminosity axis the paper measures
its result against.  Mrk 1310 (index 23) is why this exists: both its fits sit a
factor of two under the data at 1450 A while looking sound everywhere else.

Reports model/data per window for every adopted (BEST_) fit.

    python lum_window_check.py
    python lum_window_check.py --tree pipeline_output/fit_iterations_v23
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = (("1450", 1445.0, 1465.0), ("1700", 1700.0, 1705.0), ("2500", 2490.0, 2510.0))


def data_level(path, lo, hi):
    """Median flux and its uncertainty over a window, unmasked pixels only."""
    try:
        d = fits.open(path)[1].data
    except Exception:
        return None
    w = d["Rest-frame Wavelength"]
    f = d["Coadded Flux (Arbitrary Units)"]
    e = d["Coadded Flux Errors"]
    m = d["Bad Pixel Mask"]
    ok = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0) & (w >= lo) & (w <= hi)
    if ok.sum() < 5:
        return None
    # 1.253 sigma/sqrt(n) is the standard error of a median for Gaussian noise
    return (float(np.median(f[ok])),
            float(1.253 * np.median(e[ok]) / np.sqrt(ok.sum())),
            int(ok.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=os.path.join(HERE, "pipeline_output",
                                                   "chosen_visit", "fits"))
    ap.add_argument("--rebin", default=os.path.join(HERE, "pipeline_output",
                                                    "chosen_visit", "rebin"))
    ap.add_argument("--out", default=os.path.join(HERE, "pipeline_output", "final",
                                                  "lum_window_check.csv"))
    a = ap.parse_args()
    rows = []
    for rec in sorted(glob.glob(os.path.join(glob.escape(a.tree), "*", "records",
                                             "BEST_*.json"))):
        folder = rec.split(os.sep)[-3]
        try:
            j = json.load(open(rec))
        except Exception:
            continue
        mono = j.get("monochromatic") or {}
        spec = os.path.join(a.rebin, j["name"] + ".fits")
        if not os.path.exists(spec):
            print("  %-34s no spectrum at %s" % (folder[:34], os.path.basename(spec)))
            continue
        row = dict(folder=folder, name=j["name"],
                   blue=round(j["result"]["civ_blue"], 1),
                   ew=round(j["result"]["civ_ew"], 2))
        for k, lo, hi in WIN:
            m = mono.get(k) or {}
            mod = m.get("f_lambda")
            dl = data_level(spec, lo, hi)
            if mod is None or not np.isfinite(mod) or dl is None:
                row["r%s" % k] = ""
                row["s%s" % k] = ""
                continue
            dat, err, n = dl
            row["r%s" % k] = round(mod / dat, 3) if dat else ""
            row["s%s" % k] = round((mod - dat) / err, 1) if err else ""
            row["n%s" % k] = n
        rows.append(row)
    if not rows:
        print("no adopted fits found under %s" % a.tree)
        return 1
    fields = ["folder", "name", "blue", "ew"]
    for k, _, _ in WIN:
        fields += ["r%s" % k, "s%s" % k, "n%s" % k]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print("%-34s %8s %9s   %8s %9s   %8s %9s" %
          ("object", "r1450", "sigma", "r1700", "sigma", "r2500", "sigma"))
    for r in rows:
        print("%-34s %8s %9s   %8s %9s   %8s %9s" %
              (r["folder"][:34], r.get("r1450", ""), r.get("s1450", ""),
               r.get("r1700", ""), r.get("s1700", ""),
               r.get("r2500", ""), r.get("s2500", "")))
    for k, _, _ in WIN:
        v = np.array([r["r%s" % k] for r in rows if r.get("r%s" % k) not in ("", None)],
                     dtype=float)
        if len(v) < 2:
            continue
        print("\n%s A: n=%d  median model/data %.3f  16-84%% %.3f to %.3f  "
              "more than 20%% off: %d" %
              (k, len(v), np.median(v), *np.percentile(v, [16, 84]),
               int((np.abs(v - 1) > 0.2).sum())))
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
