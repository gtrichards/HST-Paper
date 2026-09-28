"""What the individual exposures behind one co-add actually look like.

The agreement scan (coadd_agreement.py) says which co-adds contain an exposure
that sits coherently away from the rest, or pixels tens of sigma from it.  It
cannot say what kind of thing went wrong -- a genuinely variable source caught
at two epochs, a grating whose normalisation is off, a single bad exposure, a
detector artefact -- and those want different responses.  This draws the
exposures so that can be judged by eye.

Every contributing exposure is drawn; the one furthest from the others is drawn
heavy and dark, the co-add dashed on top.  Left is the whole C IV window, right
a zoom on the line.

    python figure_exposures.py OBJECT --inst COS --z 0.12
    python figure_exposures.py --batch pipeline_output/final/coadd_agreement.csv --top 20
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figure_coadd_nanmedian import (place, coadd, weights_of, exposure_disagreement,
                                    band_offsets, CIV)

MAST = os.path.join(HERE, "data_v23", "MAST_v23")
OUT = os.path.join(HERE, "pipeline_output", "final", "exposure_review")


def worst_exposure(F, E, sel, contrib):
    """Index into contrib of the exposure furthest from the others."""
    if len(contrib) < 2:
        return None
    cols = np.where(sel)[0]
    f = F[np.ix_(contrib, cols)]
    e = E[np.ix_(contrib, cols)]
    good = np.isfinite(f) & np.isfinite(e) & (e > 0)
    with np.errstate(invalid="ignore"):
        ref = np.nanmedian(np.where(good, f, np.nan), axis=0)
    best, k_best = -1.0, None
    for k in range(len(contrib)):
        g = good[k] & np.isfinite(ref)
        if g.sum() < 20:
            continue
        scale = np.nanmedian(np.abs(ref[g]))
        if scale <= 0:
            continue
        v = abs(np.median(f[k][g] - ref[g])) / scale
        if v > best:
            best, k_best = v, k
    return k_best


def draw(name, inst, z, out_path, zoom=(1540.0, 1560.0), label=""):
    path = os.path.join(MAST, name, inst)
    lam, F, E, M = place(name, path, z, inst)
    rest = lam / (1 + z)
    sel = (rest >= CIV[0]) & (rest <= CIV[1])
    contrib = [k for k in range(F.shape[0]) if np.isfinite(F[k, sel]).sum() > 10]
    if len(contrib) < 1:
        print("   %s %s: nothing reaches C IV" % (name, inst))
        return None
    W = weights_of(E, M, lam, z)
    fco, _ = coadd(F, E, W, drop_nan=True)
    dis = exposure_disagreement(F, E, sel, contrib)
    osig, ofrac, nbad = band_offsets(F, E, sel, contrib)
    kw = worst_exposure(F, E, sel, contrib)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2))
    for ax, (lo, hi) in zip(axes, [CIV, zoom]):
        s = (rest >= lo) & (rest <= hi)
        few = len(contrib) <= 10
        cyc = plt.cm.tab10(np.linspace(0, 1, 10))
        for n, i in enumerate(contrib):
            if kw is not None and n == kw:
                continue
            ax.plot(rest[s], F[i, s],
                    color=cyc[n % 10] if few else "0.78",
                    lw=0.9 if few else 0.6, alpha=0.9 if few else 0.7, zorder=2)
        if kw is not None:
            ax.plot(rest[s], F[contrib[kw], s], color="#8e44ad", lw=2.0, zorder=4,
                    label="furthest exposure")
        ax.plot(rest[s], fco[s], color="k", lw=1.7, dashes=(5, 2), zorder=5,
                label="co-add")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r"Rest wavelength ($\AA$)")
    axes[0].set_ylabel("Continuum-normalised flux")
    axes[0].legend(fontsize=9, loc="upper left")
    # The line dominates the range; keep the window panel readable.
    s = sel & np.isfinite(fco)
    if s.sum():
        hi_y = np.nanpercentile(np.where(np.isfinite(F[np.ix_(contrib, np.where(s)[0])]),
                                         F[np.ix_(contrib, np.where(s)[0])], np.nan), 99.5)
        lo_y = np.nanpercentile(np.where(np.isfinite(F[np.ix_(contrib, np.where(s)[0])]),
                                         F[np.ix_(contrib, np.where(s)[0])], np.nan), 0.5)
        pad = 0.12 * (hi_y - lo_y) if np.isfinite(hi_y - lo_y) else 1.0
        for ax in axes:
            ax.set_ylim(lo_y - pad, hi_y + pad)
    fig.suptitle("%s   %s %s   %d exposures   disagreement %.1f sigma/px   "
                 "worst offset %.0f%% of the flux   %d pixels >10 sigma"
                 % (label, name, inst, len(contrib), dis, 100 * ofrac, nbad), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return ofrac


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("object", nargs="?")
    ap.add_argument("--inst", default="COS")
    ap.add_argument("--z", type=float)
    ap.add_argument("--batch", default=None,
                    help="coadd_agreement.csv; draws the worst co-adds in it")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args()

    if a.batch:
        d = pd.read_csv(a.batch)
        ok = d[d.status == "ok"].copy()
        pick = pd.concat([ok.nlargest(a.top, "offset_frac"),
                          ok.nlargest(8, "n_bad_px")]).drop_duplicates(["index", "inst"])
        dec = pd.read_csv(os.path.join(HERE, "pipeline_output", "final",
                                       "pass_decisions.csv")).set_index("index")
        pick = pick.sort_values("offset_frac", ascending=False)
        for _, r in pick.iterrows():
            i = int(r["index"])
            z = float(dec.loc[i, "z_adopted"])
            tag = "%03d_%s_%s" % (i, str(r["name"]).replace("/", "_"), r["inst"])
            lab = "%03d  row %s  %s" % (i, r.get("row_num", "?"), r.get("name_pub", r["name"]))
            try:
                v = draw(str(r["name"]), str(r["inst"]), z,
                         os.path.join(a.out_dir, tag + ".png"), label=lab)
                print("%-46s offset %5.0f%%" % (tag, 100 * v) if v is not None else tag,
                      flush=True)
            except Exception as exc:
                print("%-46s FAILED: %s" % (tag, exc), flush=True)
        print("-> %s" % a.out_dir)
        return 0

    if not (a.object and a.z):
        sys.exit("give OBJECT and --z, or --batch")
    draw(a.object, a.inst, a.z, os.path.join(a.out_dir, "%s_%s.png" % (a.object, a.inst)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
