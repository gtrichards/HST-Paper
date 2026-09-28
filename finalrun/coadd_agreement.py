"""Do an object's exposures agree well enough to be co-added?

Co-adding assumes the exposures are repeated measurements of the same spectrum,
differing only by noise.  Where that fails -- a variable source between epochs,
a normalisation or calibration offset between gratings, one bad exposure -- the
weighted median is combining things that are not the same quantity, and picking
one exposure or one consistent set can be better than averaging them all.

Two statistics per co-add, over the C IV window:

  disagree_sigma   sqrt(chi2/dof) of the exposures about their median, pixel by
                   pixel.  1 means they scatter no more than their errors allow.
  offset_sigma     the largest coherent offset of any one exposure from the
                   others, in units of its own uncertainty on that mean.  This
                   is the one that catches a difference that is small per pixel
                   but points the same way across the window.
  offset_frac      that same offset as a fraction of the flux -- the size that
                   actually matters for the fit, as opposed to its significance.

Reports rather than decides: the output is a list to look at, not a cut.

    python coadd_agreement.py [--all] [--max-rows N]
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figure_coadd_nanmedian import place, exposure_disagreement, band_offsets, CIV

MAST = os.path.join(HERE, "data_v23", "MAST_v23")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="every object in the work queue, not just the reviewed ones")
    ap.add_argument("--max-rows", type=int, default=400,
                    help="skip co-adds with more rows than this (echelle); they are slow")
    ap.add_argument("--out", default=os.path.join(HERE, "pipeline_output", "final",
                                                  "coadd_agreement.csv"))
    a = ap.parse_args()

    queue = {int(r["order"]): r for r in csv.DictReader(
        open(os.path.join(HERE, "pipeline_output", "work_queue.csv")))}
    dec = pd.read_csv(os.path.join(HERE, "pipeline_output", "final",
                                   "pass_decisions.csv")).set_index("index")
    if a.all:
        idxs = sorted(queue)
    else:
        cmp_dir = os.path.join(HERE, "pipeline_output", "final", "review_compare")
        idxs = sorted({int(f[:3]) for f in os.listdir(cmp_dir) if f.endswith(".png")})

    rows = []
    for i in idxs:
        if i not in dec.index:
            continue
        name = queue[i]["name_mast_key"]
        try:
            z = float(dec.loc[i, "z_adopted"])
        except Exception:
            continue
        base = os.path.join(MAST, name)
        if not os.path.isdir(base) or not np.isfinite(z):
            continue
        for inst in sorted(os.listdir(base)):
            d = os.path.join(base, inst)
            if not os.path.isdir(d):
                continue
            try:
                lam, F, E, M = place(name, d, z, inst)
            except Exception as exc:
                rows.append(dict(index=i, name=name, inst=inst, status="read failed: %s" % exc))
                continue
            if F.shape[0] > a.max_rows:
                rows.append(dict(index=i, name=name, inst=inst, rows=F.shape[0],
                                 status="skipped (%d rows)" % F.shape[0]))
                continue
            rest = lam / (1 + z)
            sel = (rest >= CIV[0]) & (rest <= CIV[1])
            contrib = [k for k in range(F.shape[0]) if np.isfinite(F[k, sel]).sum() > 10]
            if len(contrib) < 2:
                rows.append(dict(index=i, name=name, inst=inst, rows=F.shape[0],
                                 n_contrib=len(contrib), status="single exposure"))
                continue
            dis = exposure_disagreement(F, E, sel, contrib)
            osig, ofrac, nbad = band_offsets(F, E, sel, contrib)
            rows.append(dict(index=i, row_num=dec.loc[i, "row_num"],
                             name_pub=dec.loc[i, "name_pub"], name=name, inst=inst,
                             rows=F.shape[0], n_contrib=len(contrib),
                             disagree_sigma=dis, offset_sigma=osig, offset_frac=ofrac, n_bad_px=nbad,
                             status="ok"))
            print("%3d %-28s %-5s %3d exposures  disagree %6.2f  offset %7.1f sigma  "
                  "%5.1f%%  bad px %d"
                  % (i, str(dec.loc[i, "name_pub"])[:28], inst, len(contrib),
                     dis, osig, 100 * ofrac, nbad), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    ok = df[df.status == "ok"] if "status" in df else df
    print("\n%d co-adds measured" % len(ok))
    if len(ok):
        print("  coherent offset above 10%% of the flux : %d" % (ok.offset_frac > 0.10).sum())
        print("  coherent offset above 25%% of the flux : %d" % (ok.offset_frac > 0.25).sum())
        print("  with catastrophic pixels (>10 sigma)  : %d" % (ok.n_bad_px > 0).sum())
    print("-> %s" % a.out)


if __name__ == "__main__":
    sys.exit(main())
