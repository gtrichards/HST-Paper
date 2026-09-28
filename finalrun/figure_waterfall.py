"""Every exposure of an object, stacked, so they can be judged one at a time.

The overlaid version cannot answer "should this one be kept": with 80 exposures
drawn on common axes the individual spectra are lost in the band, and a low-state
exposure at a tenth of the peak looks like a line at zero.  GTR asked for these
"in portrait mode with vertical shifts" for objects with many spectra.

Each exposure is divided by its own median so that shape, not brightness, is what
the eye compares, then offset by one unit per row.  Colour carries the proposed
verdict -- kept, already dropped, proposed to drop, to review -- and each row is
labelled with its rootname, visit and level relative to the rest of its visit, so
a row can be named when discussing it.  Pages are split so no page is unreadable.

    python figure_waterfall.py --index 11 19
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from exposure_decisions import (read_exposures, per_file, assign_epochs, recommend,
                                alignment, CIV)
from rebinning import read_spec_data as R

MAST = os.path.join(HERE, "data_v23", "MAST_v23")
OUT = os.path.join(HERE, "pipeline_output", "final", "data_review")
ICA = (1260.0, 3000.0)
PER_PAGE = 30
COLOURS = {"keep": "#2c6fbb", "drop": "#e08214", "review": "#7b3294",
           "already": "#c0392b"}


def klass(action, kept_by_pipeline):
    if not kept_by_pipeline:
        return "already"
    if action.startswith("drop"):
        return "drop"
    if action.startswith("review"):
        return "review"
    return "keep"


def draw(index, name, inst, z, recs, rows_by_file, prov, out_base, label):
    order = sorted(recs, key=lambda r: (r["epoch"], str(r["grating"]), r["file"]))
    pages = [order[i:i + PER_PAGE] for i in range(0, len(order), PER_PAGE)] or [[]]
    made = []
    for pg, chunk in enumerate(pages, start=1):
        h = max(4.0, 0.34 * len(chunk) + 2.0)
        fig, ax = plt.subplots(figsize=(9.5, h))
        lo, hi = ICA
        for n, r in enumerate(chunk):
            rows = rows_by_file.get(r["file"], [])
            if not rows:
                continue
            w = np.concatenate([q["w"] for q in rows])
            f = np.concatenate([q["f"] for q in rows])
            o = np.argsort(w); w, f = w[o], f[o]
            g = np.isfinite(w) & np.isfinite(f) & (f != 0) & (w > ICA[0]) & (w < ICA[1])
            if g.sum() < 20:
                continue
            med = np.median(f[g])
            if not np.isfinite(med) or med == 0:
                continue
            y = f[g] / med * 0.38 + n
            c = COLOURS[klass(r["action"], prov.get(r["file"], {"kept": True})["kept"])]
            ax.plot(w[g], y, color=c, lw=0.45, alpha=0.9)
            ax.axhline(n, color="0.9", lw=0.4, zorder=0)
            ratio = r.get("level_ratio", np.nan)
            ax.text(ICA[0] - 25, n, "%s  v%d %s  %s"
                    % (r["file"].split("_")[0], r["epoch"], str(r["grating"])[:5],
                       ("%.2fx" % ratio) if np.isfinite(ratio) else ""),
                    ha="right", va="center", fontsize=6, color=c)
        ax.axvspan(*CIV, color="#f5d76e", alpha=0.35, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-1, len(chunk))
        ax.set_yticks([])
        ax.set_xlabel(r"Rest wavelength ($\AA$)")
        ax.set_title("%s   %s   %d of %d exposures   (each divided by its own median)"
                     % (label, inst, len(chunk), len(order)), fontsize=10)
        fig.subplots_adjust(left=0.22, right=0.98, top=1 - 0.5 / h, bottom=0.6 / h)
        p = "%s_%s%s.png" % (out_base, inst, "" if len(pages) == 1 else "_p%d" % pg)
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, nargs="+", required=True)
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args()
    queue = {int(r["order"]): r for r in csv.DictReader(
        open(os.path.join(HERE, "pipeline_output", "work_queue.csv")))}
    dec = pd.read_csv(os.path.join(HERE, "pipeline_output", "final",
                                   "pass_decisions.csv")).set_index("index")
    for i in a.index:
        name = queue[i]["name_mast_key"]
        z = float(dec.loc[i, "z_adopted"])
        nm = dec.loc[i, "name_pub"] if i in dec.index else name
        label = "%03d  row %s  %s" % (i, dec.loc[i, "row_num"] if i in dec.index else "?", nm)
        base = os.path.join(MAST, name)
        per_inst, rows_all, recs_all = {}, {}, []
        for inst in sorted(os.listdir(base)):
            d = os.path.join(base, inst)
            if not os.path.isdir(d):
                continue
            raw = read_exposures(d, inst, z)
            recs = per_file(raw)
            if not recs:
                continue
            for q in raw:
                rows_all.setdefault(q["file"], []).append(q)
            recs = assign_epochs(recs)
            per_inst[inst] = recs
            recs_all += recs
        al = alignment(recs_all, rows_all)
        for inst, recs in per_inst.items():
            for r in recs:
                r["align_rms"] = al.get(r["file"], np.nan)
            recs, _ = recommend(recs)
            try:
                R.read_data_flat(name, os.path.join(base, inst), inst, z)
                prov = {p["file"]: p for p in R.PROVENANCE}
            except Exception:
                prov = {}
            out_base = os.path.join(a.out_dir, "%03d_%s_stack" % (i, name.replace("/", "_")))
            made = draw(i, name, inst, z, recs, rows_all, prov, out_base, label)
            print("  %-5s %2d exposures -> %s" % (inst, len(recs),
                                                  ", ".join(os.path.basename(m) for m in made)),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
