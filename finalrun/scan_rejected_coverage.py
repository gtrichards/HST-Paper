"""Did the single-visit policy cost us measurements that earlier passes made?

GTR spotted it on one object: "Table 1 has 1 object with an odd flag (meaning a
Sulentic object) that has manmask=0.  That shouldn't be the case.  Ton 469. ...
Perhaps the spectrum that Sulentic used didn't get downloaded (or included)
somehow?"  On Ton 469 nothing was missing from the download -- all 42 exposures
are on disk -- but the CHOSEN VISIT staged a subset spanning 1120-2139 A rest
while the full FOS holding spans 1120-5034 A, and the ICA fit is inadmissible on
the former (117-251 sub-continuum pixels under every component basis) and
admissible on the latter (zero).  Trevor's earlier pass measured it from the full
co-add and got -839.8 km/s; refitting the all-exposure product reproduces
-839.8 exactly.  Sulentic 2007 reports -969.0.  So the object is measurable and
our pass rejected it for want of coverage the archive already holds.

THIS SCAN ASKS HOW MANY MORE THERE ARE, by two independent tests over every
manmask-0 row.

TEST 1, THE SHARPEST: did an earlier measurement succeed?  The Sulentic/HSLA
tables carry CIV_blue_tvm (Trevor's ICA fit) and CIV_blue_sul07 (Sulentic 2007).
A row we reject that either of them measured is a disagreement to explain, not
necessarily an error -- their fits may be the ones at fault, as GTR judged for
PG 1216+069 and PG 1049-005 in the IUE cross-check -- but it is the right place
to look.

TEST 2: does the v23 all-exposure product cover materially more than the chosen
visit, and in particular does it reach redward of 2000 A rest where the chosen
visit does not?  That is the Ton 469 signature: the components need the long
baseline to anchor, and C III] alone was not enough.

Reports only.  Nothing is refitted and no verdict is changed.

    python scan_rejected_coverage.py
    python scan_rejected_coverage.py --csv
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from astropy.table import Table

PAPER = "/Users/gtr/Work/git/HST-Paper"
LEDGER = os.path.join(PAPER, "Data", "decisions.csv")
QUEUE = os.path.join(HERE, "pipeline_output", "work_queue.csv")
CHOSEN = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
V23 = os.path.join(HERE, "data_v23", "RebinnedSpec_v23")
SUL = os.path.join(PAPER, "Data", "HST_CIV_Sulentic2007_HSLA2018_finalprops.csv")
SUL2 = os.path.join(PAPER, "Data", "HST_CIV_Sulentic2007_HSLA2018_1Jan2023.csv")
OUT = os.path.join(HERE, "pipeline_output", "final", "rejected_coverage.csv")

#: The test for "materially more coverage".  An ABSOLUTE threshold of 2000 A
#: rest was tried first and EXCLUDED TON 469 ITSELF, whose chosen visit reaches
#: 2139 A -- already past 2000 -- while its full holding reaches 5034.  What
#: matters is the GAIN, not the absolute reach, so the test is a gain of at
#: least MIN_GAIN Angstroms in a single product that also covers C IV.
MIN_GAIN = 300.0
MIN_CIV = 10


def cover(path):
    t = Table.read(path)
    w = np.asarray(t["Rest-frame Wavelength"], float)
    f = np.asarray(t["Coadded Flux (Arbitrary Units)"], float)
    e = np.asarray(t["Coadded Flux Errors"], float)
    m = np.asarray(t["Bad Pixel Mask"], float)
    g = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    if not g.any():
        return None
    civ = g & (w >= 1500) & (w <= 1600)
    return dict(lo=float(w[g].min()), hi=float(w[g].max()), px=int(g.sum()),
                civ=int(civ.sum()),
                sn=(float(np.median(f[civ] / e[civ])) if civ.sum() else np.nan))


def best_of(paths):
    """The product that best serves a C IV measurement: most usable C IV pixels,
    ties broken by the longest red reach.

    The first version of this ranked on red reach alone, which is wrong and
    produced a misleading Test 2: an object's longest-reaching product is often a
    DIFFERENT grating from the one covering C IV, so the scan reported pairs like
    order 20 with v23_hi = 10080 A and v23_civ = 0 -- two separate spectra read as
    one.  The Ton 469 signature needs BOTH properties in a SINGLE product, since
    that is what the fitter is handed.
    """
    out, out_p = None, ""
    for p in paths:
        try:
            c = cover(p)
        except Exception:
            continue
        if c is None:
            continue
        if out is None or (c["civ"], c["hi"]) > (out["civ"], out["hi"]):
            out, out_p = c, p
    return (out, out_p) if out else (None, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()

    led = {}
    for r in csv.DictReader(open(LEDGER)):
        o = int(re.match(r"^(\d+)", r["folder"]).group(1))
        led[o] = (str(r["manmask"]).strip(), r["folder"], r["reject_reason"])
    q = {int(r["order"]): r for r in csv.DictReader(open(QUEUE))}

    # earlier measurements, keyed on SDSS_NAME-free designation matching
    prev = {}
    for path in (SUL, SUL2):
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        for _, r in d.iterrows():
            ra, de = r.get("_RA"), r.get("_DE")
            if pd.isna(ra) or pd.isna(de):
                continue
            key = (round(float(ra), 3), round(float(de), 3))
            tvm = r.get("CIV_blue_tvm")
            s07 = r.get("CIV_blue_sul07")
            cur = prev.get(key, {})
            if pd.notna(tvm):
                cur["tvm"] = float(tvm)
                cur["tvm_ew"] = float(r.get("CIV_EW_tvm")) if pd.notna(r.get("CIV_EW_tvm")) else np.nan
            if pd.notna(s07):
                cur["sul"] = float(s07)
                cur["sul_ew"] = float(r.get("CIV_EW_sul07")) if pd.notna(r.get("CIV_EW_sul07")) else np.nan
            cur["name"] = str(r.get("Final_Name", ""))
            prev[key] = cur

    def desig_to_radec(d):
        s = str(d).strip().lstrip("J")
        i = max(s.rfind("+"), s.rfind("-"))
        ra_s, de_s = s[:i], s[i:]
        ra = (float(ra_s[0:2]) + float(ra_s[2:4]) / 60 + float(ra_s[4:]) / 3600) * 15.0
        sg = -1.0 if de_s[0] == "-" else 1.0
        de_s = de_s[1:]
        de = sg * (float(de_s[0:2]) + float(de_s[2:4]) / 60 + float(de_s[4:]) / 3600)
        return ra, de

    pk = {k: v for k, v in prev.items()}
    rows = []
    for o in sorted(led):
        mm, folder, reason = led[o]
        if mm != "0":
            continue
        r = q.get(o)
        if r is None:
            continue
        stems = [s.strip() for s in str(r.get("stems") or "").split(";") if s.strip()]
        ch, chp = best_of([p for s in stems
                           for p in glob.glob(os.path.join(glob.escape(CHOSEN),
                                                           "%s*.fits" % glob.escape(s)))])
        v2, v2p = best_of([p for s in stems
                           for p in glob.glob(os.path.join(glob.escape(V23),
                                                           "%s*.fits" % glob.escape(s)))])
        # earlier measurement, matched on position to 4 arcsec
        hit = None
        try:
            ra, de = desig_to_radec(r["designation"])
            for (kra, kde), v in pk.items():
                if abs(kra - ra) * np.cos(np.radians(de)) < 0.0011 and abs(kde - de) < 0.0011:
                    hit = v
                    break
        except Exception:
            pass
        rows.append(dict(
            order=o, row_num=r.get("row_num", ""), name_pub=str(r.get("name_pub") or "").strip(),
            z=float(r["z"]) if r.get("z") else np.nan, reject_reason=reason,
            chosen_hi=(round(ch["hi"], 1) if ch else np.nan),
            chosen_civ=(ch["civ"] if ch else 0),
            chosen_sn=(round(ch["sn"], 2) if ch and np.isfinite(ch["sn"]) else np.nan),
            v23_hi=(round(v2["hi"], 1) if v2 else np.nan),
            v23_civ=(v2["civ"] if v2 else 0),
            v23_sn=(round(v2["sn"], 2) if v2 and np.isfinite(v2["sn"]) else np.nan),
            gain_A=(round(v2["hi"] - ch["hi"], 1) if (ch and v2) else np.nan),
            v23_reaches_long=bool(v2 and v2["civ"] >= MIN_CIV and ch
                                  and (v2["hi"] - ch["hi"]) >= MIN_GAIN),
            tvm=(hit or {}).get("tvm", np.nan),
            tvm_ew=(hit or {}).get("tvm_ew", np.nan),
            sul=(hit or {}).get("sul", np.nan),
            prev_name=(hit or {}).get("name", "")))
    d = pd.DataFrame(rows)

    print("MANMASK-0 ROWS SCANNED: %d" % len(d))
    print()
    meas_before = d[(d.tvm.notna()) | (d.sul.notna())]
    print("*** TEST 1 -- REJECTED BY US, MEASURED BY AN EARLIER PASS: %d ***" % len(meas_before))
    if len(meas_before):
        print("  %-5s %-26s %7s %9s %9s %9s %8s %8s  %s"
              % ("order", "name_pub", "z", "tvm", "tvm_EW", "sul07", "ch_civ", "v23_hi", "reject_reason"))
        for _, r in meas_before.sort_values("order").iterrows():
            print("  %-5d %-26s %7.4f %9s %9s %9s %8d %8s  %s"
                  % (r.order, str(r.name_pub)[:26], r.z,
                     ("%+.1f" % r.tvm) if pd.notna(r.tvm) else "--",
                     ("%.1f" % r.tvm_ew) if pd.notna(r.tvm_ew) else "--",
                     ("%+.1f" % r.sul) if pd.notna(r.sul) else "--",
                     r.chosen_civ, r.v23_hi, str(r.reject_reason)[:30]))
    print()
    gain = d[d.v23_reaches_long]
    print("*** TEST 2 -- a SINGLE v23 product covers C IV and reaches >= %.0f A FURTHER"
          " RED than the chosen visit: %d ***" % (MIN_GAIN, len(gain)))
    if len(gain):
        print("  %-5s %-26s %7s %9s %9s %8s %8s  %s"
              % ("order", "name_pub", "z", "chosen_hi", "v23_hi", "ch_civ", "v23_civ", "reject_reason"))
        for _, r in gain.sort_values("gain_A", ascending=False).iterrows():
            print("  %-5d %-26s %7.4f %9s %9s %8d %8d  %s"
                  % (r.order, str(r.name_pub)[:26], r.z, r.chosen_hi, r.v23_hi,
                     r.chosen_civ, r.v23_civ, str(r.reject_reason)[:30]))
    print()
    both = d[d.v23_reaches_long & ((d.tvm.notna()) | (d.sul.notna()))]
    print("*** BOTH TESTS -- the Ton 469 signature: %d ***" % len(both))
    for _, r in both.sort_values("order").iterrows():
        print("  %-5d %-26s  tvm=%s  chosen_hi=%s v23_hi=%s"
              % (r.order, str(r.name_pub)[:26],
                 ("%+.1f" % r.tvm) if pd.notna(r.tvm) else "--", r.chosen_hi, r.v23_hi))
    if a.csv:
        d.to_csv(OUT, index=False)
        print()
        print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
