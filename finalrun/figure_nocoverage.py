"""Show, for the manmask-0 objects marked without individual review, where their
coverage lies relative to C IV.

Seven rows in Data/decisions.csv carry manmask 0 on a BLANKET authorisation
rather than an individual look -- their notes say "Marked without individual
review per GTR, who authorised that where the verdict agrees with the sheet".
They are 166 Ton 730, 226 E1423+201, 243 Q 1217+023, 274 FBQS J2226-0901,
284 2MASS J14314604+0301444, 307 2MASS J13304515+2813216 and 334 QSO J1619+3342,
and all seven have civ_frac_best = 0.0 in the work queue, along with
ciii_frac_best = 0.0 and mgii_frac_best = 0.0.

GTR, on being told they are the only unreviewed zeros in the sample: "As for
those 7, I should probably visually confirm them."  The interactive fitter cannot
show them anything -- there is no C IV in the spectrum to display -- so this
plots what there IS: the full rest-frame extent of every spectrum the object has,
with the C IV measurement window marked, so the gap between the two is visible
and the verdict can be confirmed or overturned by eye.

One figure per object, into pipeline_output/final/nocoverage/.

    python figure_nocoverage.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from richardsplot import richardsplot as rplot   # SM-style defaults
except Exception:
    pass

from rebin_path import REBIN, V23

#: Where to look for a product.  The chosen-visit tree only holds objects that
#: were actually fitted, and a manmask-0 object never was, so three of the seven
#: are absent from it; the regenerated v23 tree holds every object retrieved from
#: MAST and is the fallback.
TREES = [REBIN, V23]

QUEUE = os.path.join(HERE, "pipeline_output", "work_queue.csv")
OUTDIR = os.path.join(HERE, "pipeline_output", "final", "nocoverage")

#: The seven, by work-queue order.
ORDERS = [166, 226, 243, 274, 284, 307, 334]

CIV = (1500.0, 1600.0)
LINES = [(1215.67, r"Ly$\alpha$"), (1240.0, "N V"), (1397.0, "Si IV"),
         (1549.48, "C IV"), (1909.0, "C III]"), (2798.0, "Mg II")]


def spectra_for(stems):
    """Every rebinned product matching the queue's stems, as (label, wave, flux, err)."""
    out = []
    for stem in [s.strip() for s in str(stems).split(";") if s.strip()]:
        cand = []
        for tree in TREES:
            cand = sorted(glob.glob(os.path.join(tree, "%s*.fits" % stem)))
            if cand:
                break
        for p in cand:
            try:
                from astropy.table import Table
                t = Table.read(p)
                w = np.asarray(t[t.colnames[0]], float)
                f = np.asarray(t[t.colnames[1]], float)
                m = np.asarray(t["Bad Pixel Mask"], float) if "Bad Pixel Mask" in t.colnames \
                    else np.zeros(len(w))
                g = np.isfinite(f) & (m == 0)
                e = (np.asarray(t["Coadded Flux Errors"], float)
                     if "Coadded Flux Errors" in t.colnames else np.full(len(w), np.nan))
                if g.sum() > 10:
                    out.append((os.path.basename(p).replace(".fits", ""),
                                w[g], f[g], e[g]))
            except Exception:
                continue
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    q = pd.read_csv(QUEUE)
    made = []
    for o in ORDERS:
        r = q[q.order == o]
        if not len(r):
            print("  order %d not in the queue" % o)
            continue
        r = r.iloc[0]
        z = float(r["z"])
        sp = spectra_for(r["stems"])
        fig, ax = plt.subplots(figsize=(9.5, 4.4))
        if sp:
            lo = min(w.min() for _, w, _, _ in sp)
            hi = max(w.max() for _, w, _, _ in sp)
            allf = np.concatenate([f for _, _, f, _ in sp])
            # Robust limits.  An earlier version divided by the median, which on
            # these faint objects is near zero, so the noise exploded off-scale
            # and the figure could not be judged from.
            q1, q99 = np.nanpercentile(allf, [1, 99])
            pad = 0.35 * (q99 - q1) if q99 > q1 else 1.0
            for lab, w, f, e in sp:
                ax.plot(w, f, lw=0.4, color="0.75", zorder=1)
                # binned to about one resolution element so real structure shows
                nb = max(8, int((w.max() - w.min()) / 2.0))
                idx = np.clip(((w - w.min()) / (w.max() - w.min()) * nb).astype(int), 0, nb - 1)
                # Keep EMPTY bins as NaN rather than dropping them, so the curve
                # breaks across a detector gap instead of drawing a straight line
                # over it -- on 284 that line looked like an enormous error bar
                # spanning the COS gap at 1345-1397 A.
                edges = np.linspace(w.min(), w.max(), nb + 1)
                bw = 0.5 * (edges[:-1] + edges[1:])
                bf = np.full(nb, np.nan)
                be = np.full(nb, np.nan)
                for k in range(nb):
                    sel = idx == k
                    if sel.sum():
                        bf[k] = np.median(f[sel])
                        be[k] = np.median(e[sel]) / max(1.0, np.sqrt(sel.sum()))
                ax.plot(bw, bf, lw=1.0, zorder=3,
                        label="%s  (%.0f-%.0f $\\AA$)" % (lab, w.min(), w.max()))
                ax.fill_between(bw, bf - be, bf + be, alpha=0.25, lw=0, zorder=2)
            ax.set_ylim(q1 - pad, q99 + pad)
            ax.axhline(0.0, color="0.5", lw=0.5, ls="-")
        else:
            lo, hi = 1000.0, 3000.0
            ax.text(0.5, 0.6, "no rebinned product found for this object",
                    transform=ax.transAxes, ha="center")
        ax.axvspan(CIV[0], CIV[1], color="#f2d7d7", zorder=0)
        ax.axvline(1549.48, color="firebrick", ls="--", lw=1.0, zorder=4)
        y0, y1 = ax.get_ylim()
        ax.text(1549.48, y1, " C IV", color="firebrick", va="top", ha="left", fontsize=9)
        for wl, nm in LINES:
            if wl != 1549.48 and lo - 20 < wl < hi + 20:
                ax.axvline(wl, color="0.55", lw=0.6, ls=":", zorder=4)
                ax.text(wl, y1, " " + nm, rotation=90, va="top", ha="left",
                        fontsize=7, color="0.4")
        ax.set_xlim(min(lo, CIV[0]) - 25, max(hi, CIV[1]) + 25)
        gap = CIV[0] - hi if hi < CIV[0] else (lo - CIV[1] if lo > CIV[1] else 0.0)
        ax.set_xlabel(r"rest wavelength ($\AA$)")
        ax.set_ylabel("flux")
        if gap:
            ax.annotate("", xy=(CIV[0], y0 + 0.08 * (y1 - y0)),
                        xytext=(hi if hi < CIV[0] else lo, y0 + 0.08 * (y1 - y0)),
                        arrowprops=dict(arrowstyle="<->", color="firebrick", lw=1.0))
            ax.text(0.5 * (CIV[0] + (hi if hi < CIV[0] else lo)), y0 + 0.13 * (y1 - y0),
                    "%.0f $\\AA$ gap" % abs(gap), color="firebrick", ha="center", fontsize=8)
        verdict = ("data end %.0f $\\AA$ short of the C IV window, %.0f $\\AA$ short of line centre"
                   % (abs(gap), 1549.48 - hi) if gap else
                   "C IV window OVERLAPS the coverage -- CHECK THIS")
        ax.set_title("%03d  %s   z = %.4f\n%s" % (o, str(r["name_pub"]).strip(), z, verdict),
                     fontsize=10)
        ax.legend(fontsize=7, loc="upper left", frameon=False)
        fig.tight_layout()
        p = os.path.join(OUTDIR, "%03d_nocoverage.png" % o)
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append((o, str(r["name_pub"]).strip(), gap, len(sp)))
        print("  %03d %-28s %d spectrum/spectra, C IV %s -> %s"
              % (o, str(r["name_pub"]).strip()[:28], len(sp),
                 ("%.0f A outside" % abs(gap)) if gap else "INSIDE coverage",
                 os.path.basename(p)))
    print()
    print("wrote %d figures to %s" % (len(made), OUTDIR))


if __name__ == "__main__":
    main()
