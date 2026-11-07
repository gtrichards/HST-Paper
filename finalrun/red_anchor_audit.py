"""How far short of the 1700-1705 A continuum anchor does each adopted object fall?

GTR, 2026-11-07: "am I consistent with manmask=2 vs. 3 with regard to the number
of pixels shy of 1700A?"  This answers that across the WHOLE adopted sample, HST
and IUE together, because the question is about consistency and a rule applied
to three IUE objects has to be checked against the 360-odd others.

WHY THE ANCHOR MATTERS.  get_CIV_parameters fits the C IV continuum as a straight
line through the RECONSTRUCTION at 1445-1465 and 1700-1705 A.  C IV itself at
1500-1600 sits BETWEEN those anchors, so it is interpolated; but if the data stop
before 1700 A the red anchor is pinned by the ICA reconstruction extrapolating
past the last measured pixel.  Continuum TILT sets the blueshift (measured on 091
Mrk 1501: one quantum per 2e-3 of slope, while continuum LEVEL moves the centroid
not at all) and continuum LEVEL sets the EW (measured on 226 E1423+201: a 15 per
cent level change runs the EW from 22.7 to 39.7).  So an unanchored red end is a
systematic on both quantities, and how big a shortfall is tolerable is a judgement
GTR has now made on three objects.

    shortfall_A   1700.0 minus the reddest usable rest wavelength, 0 if covered
    anchor_px     usable pixels inside 1700-1705 A
    usable        mask == 0, finite flux, POSITIVE error -- the fitter's criterion

    python red_anchor_audit.py            # the cross-tabulation
    python red_anchor_audit.py --csv      # also write pipeline_output/final/red_anchor.csv
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from astropy.table import Table

LEDGER = "/Users/gtr/Work/git/HST-Paper/Data/decisions.csv"
QUEUE = os.path.join(HERE, "pipeline_output", "work_queue.csv")
TREES = [(os.path.join(HERE, "pipeline_output", "chosen_visit", "fits"),
          os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin"), "HST"),
         (os.path.join(HERE, "iue", "fits"), os.path.join(HERE, "iue", "rebin"), "IUE")]
OUT = os.path.join(HERE, "pipeline_output", "final", "red_anchor.csv")

ANCHOR = (1700.0, 1705.0)


def coverage(path):
    t = Table.read(path)
    w = np.asarray(t["Rest-frame Wavelength"], float)
    f = np.asarray(t["Coadded Flux (Arbitrary Units)"], float)
    e = np.asarray(t["Coadded Flux Errors"], float)
    m = np.asarray(t["Bad Pixel Mask"], float)
    g = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    if not g.any():
        return None
    return dict(wmax=float(w[g].max()),
                anchor_px=int((g & (w >= ANCHOR[0]) & (w <= ANCHOR[1])).sum()),
                civ_px=int((g & (w >= 1500) & (w <= 1600)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()

    led = pd.read_csv(LEDGER)
    led["order"] = led.folder.astype(str).str.extract(r"^(\d+)").astype(int)
    mm = dict(zip(led.order, led.manmask.astype(str).str.strip()))
    q = pd.read_csv(QUEUE)
    pub = dict(zip(q.order.astype(int), q.name_pub.astype(str)))
    zz = dict(zip(q.order.astype(int), pd.to_numeric(q.z, errors="coerce")))

    rows = []
    for fits_dir, rebin_dir, inst in TREES:
        for rec in sorted(glob.glob(os.path.join(fits_dir, "*", "records", "BEST_*.json"))):
            o = int(re.search(r"/(\d+)_[^/]*/records/", rec).group(1))
            if mm.get(o) not in ("2", "3"):
                continue
            nm = json.load(open(rec)).get("name", "")
            p = os.path.join(rebin_dir, "%s.fits" % nm)
            if not os.path.exists(p):
                continue
            c = coverage(p)
            if c is None:
                continue
            rows.append(dict(order=o, name_pub=pub.get(o, nm), z=zz.get(o, np.nan),
                             manmask=int(mm[o]), source=inst, product=nm,
                             data_max_A=round(c["wmax"], 1),
                             shortfall_A=round(max(0.0, ANCHOR[0] - c["wmax"]), 1),
                             anchor_px=c["anchor_px"], civ_px=c["civ_px"]))
    d = pd.DataFrame(rows).drop_duplicates("order").sort_values("shortfall_A", ascending=False)

    print("RED CONTINUUM ANCHOR (1700-1705 A) ACROSS THE ADOPTED SAMPLE: %d objects" % len(d))
    print("  anchor_px = usable pixels in 1700-1705 A ; shortfall = 1700 minus the reddest usable pixel")
    print()
    print("  %-10s %8s %8s %8s" % ("manmask", "covered", "EMPTY", "total"))
    for g in (2, 3):
        s = d[d.manmask == g]
        print("  %-10d %8d %8d %8d" % (g, (s.anchor_px > 0).sum(), (s.anchor_px == 0).sum(), len(s)))
    s = d
    print("  %-10s %8d %8d %8d" % ("ALL", (s.anchor_px > 0).sum(), (s.anchor_px == 0).sum(), len(s)))
    print()
    empty = d[d.anchor_px == 0].sort_values("shortfall_A", ascending=False)
    print("EVERY ADOPTED OBJECT WITH AN EMPTY RED ANCHOR (%d), worst shortfall first:" % len(empty))
    print("  %-5s %-28s %7s %-4s %-3s %9s %9s %7s"
          % ("order", "name_pub", "z", "src", "mm", "data_max", "short_A", "civ_px"))
    for _, r in empty.iterrows():
        print("  %-5d %-28s %7.4f %-4s %-3d %9.1f %9.1f %7d"
              % (r.order, str(r.name_pub)[:28], r.z, r.source, r.manmask,
                 r.data_max_A, r.shortfall_A, r.civ_px))
    if len(empty):
        print()
        print("  by manmask:  %s" % dict(empty.manmask.value_counts().sort_index()))
        for g in (2, 3):
            s = empty[empty.manmask == g]
            if len(s):
                print("    manmask %d shortfalls (A): %s" % (g, sorted(s.shortfall_A.tolist())))
    if a.csv:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        d.to_csv(OUT, index=False)
        print()
        print("  wrote %s" % OUT)


if __name__ == "__main__":
    main()
