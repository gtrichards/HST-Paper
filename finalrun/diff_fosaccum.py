"""What did the FOS accumulation fix change?  Snapshot records vs current ones.

The 47 objects whose chosen visit held a misclassified, C IV-covering FOS
exposure were rebuilt and their adopted records replayed.  This compares the
ADOPTED measurement before and after, reading "before" from the snapshot taken
immediately prior (archive_pre_fosaccum_*/fits) and "after" from the live tree.

Reports per object: blueshift, EW, log L1450, usable C IV pixels, the
sub-continuum veto, and whether the CHOSEN VISIT itself changed -- that last
matters because visit selection is made by exposure_decisions, which computes
S/N per readout and is NOT changed by this fix, so the visit should be identical
and a change would mean something else moved too.

    python diff_fosaccum.py --snap pipeline_output/archive_pre_fosaccum_YYYYmmdd_HHMM
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(HERE, "pipeline_output", "chosen_visit", "fits")


def adopted(tree, i):
    """The BEST_ (or REJECT_ALL_) record for index i under a fits tree."""
    for pat in ("BEST_*.json", "REJECT_ALL_*.json"):
        hits = glob.glob(os.path.join(tree, "%03d_*" % i, "records", pat))
        if hits:
            return sorted(hits)[0]
    return None


def vals(path):
    d = json.load(open(path))
    r = d.get("result", {}) or {}
    m = d.get("monochromatic", {}) or {}
    s = d.get("civ_subcontinuum", {}) or {}

    def mono(k):
        v = m.get(k) or {}
        return float(v["logL"]) if v.get("covered") and v.get("logL") == v.get("logL") else np.nan
    return dict(record=os.path.basename(path), spectrum=d.get("name", ""),
                blue=r.get("civ_blue", np.nan), ew=r.get("civ_ew", np.nan),
                l1450=mono("1450"), l1700=mono("1700"),
                civ_px=s.get("n_civ_px", np.nan), veto=s.get("n_px", np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--label", default="fosaccum",
                    help="the replay label to compare against (iterNN_<label>)")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    idx = [int(x) for x in open(os.path.join(a.snap, "INDICES.txt")).read().split()]
    snap_fits = os.path.join(a.snap, "fits")

    rows = []
    for i in idx:
        po, pn = adopted(snap_fits, i), adopted(LIVE, i)
        if po is None or pn is None:
            rows.append(dict(order=i, status=("no adopted record before" if po is None
                                              else "no adopted record after")))
            continue
        o, n = vals(po), vals(pn)
        rows.append(dict(order=i, status="ok", record_before=o["record"], record_after=n["record"],
                         spectrum_before=o["spectrum"], spectrum_after=n["spectrum"],
                         blue_before=o["blue"], blue_after=n["blue"],
                         d_blue=(n["blue"] - o["blue"]) if np.isfinite(n["blue"]) and np.isfinite(o["blue"]) else np.nan,
                         ew_before=o["ew"], ew_after=n["ew"],
                         d_ew=(n["ew"] - o["ew"]) if np.isfinite(n["ew"]) and np.isfinite(o["ew"]) else np.nan,
                         l1450_before=o["l1450"], l1450_after=n["l1450"],
                         d_l1450=n["l1450"] - o["l1450"],
                         civ_px_before=o["civ_px"], civ_px_after=n["civ_px"],
                         veto_before=o["veto"], veto_after=n["veto"]))
    d = pd.DataFrame(rows)
    ok = d[d.status == "ok"].copy()
    ok["d_quanta"] = ok.d_blue / 69.03

    print("FOS ACCUMULATION FIX -- CHANGE IN THE ADOPTED MEASUREMENTS")
    print("  objects rebuilt           : %d" % len(d))
    print("  with an adopted record    : %d" % len(ok))
    bad = d[d.status != "ok"]
    if len(bad):
        print("  without one (manmask 0)   : %d  (%s)"
              % (len(bad), ", ".join(str(int(x)) for x in bad.order)))
    moved = ok[ok.d_blue.abs() > 1e-6]
    print()
    print("  measurements that MOVED   : %d of %d" % (len(moved), len(ok)))
    if len(ok):
        print("  blueshift |delta| median %.1f km/s (%.2f quanta), max %.1f (%.2f quanta)"
              % (ok.d_blue.abs().median(), ok.d_quanta.abs().median(),
                 ok.d_blue.abs().max(), ok.d_quanta.abs().max()))
        print("  EW |delta| median %.2f A, max %.2f A" % (ok.d_ew.abs().median(), ok.d_ew.abs().max()))
        print("  log L1450 |delta| median %.3f dex, max %.3f dex"
              % (ok.d_l1450.abs().median(), ok.d_l1450.abs().max()))
    sp = ok[ok.spectrum_before != ok.spectrum_after]
    print()
    print("  CHOSEN VISIT CHANGED for %d object(s) -- expected 0, since the fix does not touch"
          " visit selection" % len(sp))
    for _, r in sp.iterrows():
        print("     %-5d %s -> %s" % (r.order, r.spectrum_before, r.spectrum_after))
    print()
    if len(moved):
        print("  %-5s %-11s %9s %9s %9s %8s %8s %7s %7s"
              % ("order", "record", "blue_old", "blue_new", "d_quanta", "EW_old", "EW_new", "veto_o", "veto_n"))
        for _, r in moved.reindex(moved.d_quanta.abs().sort_values(ascending=False).index).iterrows():
            print("  %-5d %-11s %9.1f %9.1f %9.2f %8.2f %8.2f %7s %7s"
                  % (r.order, str(r.record_after)[:11], r.blue_before, r.blue_after, r.d_quanta,
                     r.ew_before, r.ew_after, r.veto_before, r.veto_after))
    vg = ok[(ok.veto_before > 0) & (ok.veto_after == 0)]
    vb = ok[(ok.veto_before == 0) & (ok.veto_after > 0)]
    print()
    print("  veto newly PASSES for %d object(s); newly FAILS for %d" % (len(vg), len(vb)))
    for lab, s in (("now passes", vg), ("now FAILS", vb)):
        for _, r in s.iterrows():
            print("     %-11s %-5d veto %s -> %s" % (lab, r.order, r.veto_before, r.veto_after))
    if a.csv:
        d.to_csv(a.csv, index=False)
        print()
        print("wrote %s" % a.csv)


if __name__ == "__main__":
    main()
