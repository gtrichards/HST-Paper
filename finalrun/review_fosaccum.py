"""One object's before/after, for walking through the FOS accumulation rebuild.

Prints what a grade needs, in the order GTR asks for it: identity, what the
rebuild did to the spectrum, the adopted measurement before and after, every
component family on the new product, the luminosity windows and the continuum
anchors, and which figures to look at.

"Before" comes from the snapshot taken prior to the rebuild; "after" from the
live tree.  Nothing is written.

    python review_fosaccum.py --index 118
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from astropy.table import Table

LIVE_FITS = os.path.join(HERE, "pipeline_output", "chosen_visit", "fits")
LIVE_REB = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
QUEUE = os.path.join(HERE, "pipeline_output", "work_queue.csv")
LEDGER = "/Users/gtr/Work/git/HST-Paper/Data/decisions.csv"
C = 1549.48
CKMS = 3e5


def snapdir():
    c = sorted(glob.glob(os.path.join(HERE, "pipeline_output", "archive_pre_fosaccum_*")))
    if not c:
        sys.exit("no archive_pre_fosaccum_* snapshot found")
    return c[-1]


def half(w, f, cont):
    EW = (w >= 1500) & (w <= 1600)
    ww, ff, cc = w[EW], f[EW], cont[EW]
    tot, cum = 0.0, [0.0]
    for i in range(len(ww)):
        dl = (ww[i + 1] - ww[i]) if i + 1 < len(ww) else (ww[i] - ww[i - 1])
        tot += max((ff[i] - cc[i]) / cc[i] * dl, 0.0)
        cum.append(tot)
    j = np.abs(tot / 2.0 - np.array(cum)).argmin()
    return ((C - ww[j]) / C) * CKMS, tot


def sn_table(path, label):
    t = Table.read(path)
    w = np.asarray(t["Rest-frame Wavelength"], float)
    f = np.asarray(t["Coadded Flux (Arbitrary Units)"], float)
    e = np.asarray(t["Coadded Flux Errors"], float)
    m = np.asarray(t["Bad Pixel Mask"], float)
    g = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    cells = []
    for lo, hi, wn in ((1445, 1465, "L1450"), (1500, 1600, "C IV"),
                       (1700, 1705, "cont_red"), (1690, 1710, "L1700")):
        s = g & (w >= lo) & (w <= hi)
        cells.append("%s %s(%d)" % (wn, ("%.2f" % np.median(f[s] / e[s])) if s.sum() else "--", s.sum()))
    print("   %-7s rest %.0f-%.0f  %4d px | %s"
          % (label, w[g].min(), w[g].max(), g.sum(), "  ".join(cells)))
    return w, g


def rec_line(label, path):
    d = json.load(open(path))
    r = d["result"]
    m = d.get("monochromatic", {}) or {}

    def mono(k):
        v = m.get(k) or {}
        return ("%.3f" % v["logL"]) if v.get("covered") and v.get("logL") == v.get("logL") else "--"
    s = d.get("civ_subcontinuum", {}) or {}
    print("   %-26s %+9.1f %8.2f %10s %10s %7d %6s"
          % (label, r["civ_blue"], r["civ_ew"], mono("1450"), mono("1700"),
             s.get("n_px", -1), s.get("n_civ_px", "?")))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    i = a.index
    SNAP = snapdir()

    q = {int(r["order"]): r for r in csv.DictReader(open(QUEUE))}
    mm = {}
    for r in csv.DictReader(open(LEDGER)):
        o = int(re.match(r"^(\d+)", r["folder"]).group(1))
        mm[o] = (str(r["manmask"]).strip(), r["folder"], r["reject_reason"])
    acc_p = os.path.join(HERE, "pipeline_output", "final", "fos_accum.csv")
    exposures = []
    if os.path.exists(acc_p):
        import pandas as pd
        acc = pd.read_csv(acc_p)
        h = acc[(acc.order == i) & acc.changed & acc.old_cumulative & ~acc.new_cumulative
                & acc.covers_civ & acc.in_chosen_visit]
        exposures = [(r.exposure, r.grating, int(r.groups)) for _, r in h.iterrows()]

    qr = q.get(i, {})
    mk, folder, reason = mm.get(i, ("?", "?", ""))
    print("INDEX %d   v22 row %s   %s   %s" % (i, qr.get("row_num", "?"),
          str(qr.get("name_pub", "?")).strip(), folder))
    print("   z = %s   manmask %s   %s" % (qr.get("z", "?"), mk, reason))
    print("   misclassified C IV exposures in the chosen visit: %s"
          % (", ".join("%s (%s, %d groups, S/N x%.1f)" % (e, g, n, n ** 0.5)
                       for e, g, n in exposures) or "none recorded"))
    print()

    live = sorted(glob.glob(os.path.join(LIVE_FITS, "%03d_*" % i)))
    if not live:
        sys.exit("no fit folder for index %d" % i)
    live = live[0]
    base = os.path.basename(live)
    adopted = (sorted(glob.glob(os.path.join(live, "records", "BEST_*.json")))
               or sorted(glob.glob(os.path.join(live, "records", "REJECT_ALL_*.json"))))
    if not adopted:
        print("   NO ADOPTED RECORD (manmask 0)")
    nm = json.load(open(adopted[0]))["name"] if adopted else None

    if nm:
        print("SPECTRUM, before vs after the rebuild:")
        sn_table(os.path.join(SNAP, "rebin", "%s.fits" % nm), "before")
        w, g = sn_table(os.path.join(LIVE_REB, "%s.fits" % nm), "after")
        print()
        print("ADOPTED MEASUREMENT:")
        print("   %-26s %9s %8s %10s %10s %7s %6s"
              % ("variant", "blue", "EW", "logL1450", "logL1700", "veto", "civ_px"))
        snap_rec = os.path.join(SNAP, "fits", base, "records", os.path.basename(adopted[0]))
        if os.path.exists(snap_rec):
            rec_line("before (old spectrum)", snap_rec)
        fa = os.path.join(live, "records", "iter%02d_fosaccum.json" % 0)
        cand = sorted(glob.glob(os.path.join(live, "records", "iter*_fosaccum.json")))
        for c_ in cand:
            rec_line("same masks, new spectrum", c_)
        rec_line("LIVE adopted", adopted[0])
        ov = json.load(open(adopted[0]))["override"]
        print()
        print("   adopted override: %d mask ranges, %d mask pixels, comps=%s"
              % (len(ov.get("mask_ranges") or []), len(ov.get("mask_pixels") or []),
                 ov.get("comps_use")))
        print()
        from ica.manual_fix import ICAManualFixProcessor
        proc = ICAManualFixProcessor(rebin_path=LIVE_REB, master_mode=True)
        print("EVERY FAMILY ON THE NEW PRODUCT, no masks:")
        print("   %-6s %9s %8s %9s %8s" % ("family", "model", "EW", "data", "veto px"))
        for fam in (None, "low", "mod", "high"):
            try:
                r = proc.fit_with_overrides(nm, mask_ranges=[], mask_pixels=[],
                                            unmask_ranges=[], unmask_pixels=[], comps_use=fam)
            except Exception as ex:
                print("   %-6s FAILED %s" % (fam or "auto", type(ex).__name__))
                continue
            wi, fi = r["wave_ica"], r["flux_ica"]
            wa, fa_, ea, ma = r["wave_arb"], r["flux_arb"], r["errs_arb"], r["mask_arb"]
            gg = (ma == 0) & np.isfinite(fa_) & np.isfinite(ea) & (ea > 0)
            c1 = (wi >= 1445) & (wi <= 1465)
            c2 = (wi >= 1700) & (wi <= 1705)
            m_, b_ = np.polyfit(np.concatenate((wi[c1], wi[c2])),
                                np.concatenate((fi[c1], fi[c2])), 1)
            bm, em = half(wi, fi, wi * m_ + b_)
            bd, _ = half(wa[gg], fa_[gg], (wa * m_ + b_)[gg])
            win = (wi >= 1500) & (wi <= 1600)
            print("   %-6s %+9.1f %8.2f %+9.1f %8d"
                  % (fam or "auto", bm, em, bd, int(((fi - (wi * m_ + b_)) < 0)[win].sum())))
    print()
    print("FIGURES in %s :" % live)
    for p in sorted(glob.glob(os.path.join(live, "*.png")), key=os.path.getmtime, reverse=True)[:6]:
        print("   %-34s %s" % (os.path.basename(p),
              "<-- adopted" if os.path.basename(p).startswith(("BEST_", "REJECT_ALL_"))
              else ("<-- post-fix, same masks" if "fosaccum" in p else "")))


if __name__ == "__main__":
    main()
