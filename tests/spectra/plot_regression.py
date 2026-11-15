"""What a change to the analysis code does to the measured quantities.

GTR, 2026-11-14: "I just care that the pipeline reports the 'correct' EW,
blueshift, and L1450. ... The point being to recognize when changes to the code
produce changes to the test set.  Showing that the test set doesn't change after
a change in the pipeline doesn't prove that all is well, but changes to the test
set require detailed investigation to see whether it is the old or new version
(or neither!) that is correct."

So this plots the DIFFERENCE in the three published quantities, test case by
test case, between two versions of the code.  Nothing here says which version is
right -- it says where to look.

A point inside the shaded band moved by less than the tolerance and needs no
attention.  A point outside it is labelled, and somebody has to decide whether
the old value, the new value, or neither is correct.

    python plot_regression.py BEFORE.json AFTER.json --label-a "..." --label-b "..."

where the JSON files come from run_spectra_tests.py --json, taken under each
version of the code.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from richardsplot import richardsplot as rplot
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))

#: the tolerances the suite uses.  The blueshift is quantised at 69.03 km/s so
#: it cannot move by less than one pixel; EW and luminosity are sums and need a
#: real tolerance.  GTR: "We're not trying to prevent *any* changes, just
#: changes that matter."
TOL = (("blueshift", "blueshift", r"$\Delta$ blueshift (km s$^{-1}$)", 0.0),
       ("ew", "EW", r"$\Delta$ EW ($\AA$)", 0.1),
       ("logL1450", "log L1450", r"$\Delta$ log $L_{1450}$ (dex)", 0.01))


def load(*paths):
    out = {}
    for p in paths:
        out.update(json.load(open(p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before", nargs="+")
    ap.add_argument("--after", nargs="+", required=True)
    ap.add_argument("--label-a", default="before")
    ap.add_argument("--label-b", default="after")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A, B = load(*a.before), load(*a.after)
    names = [n for n in sorted(set(A) & set(B))]
    if not names:
        raise SystemExit("no test cases in common")
    short = [n.replace("cartoon_", "~").replace("fos_", "FOS ").replace("stis_", "STIS ")
             .replace("cos_", "COS ").replace("iue_", "IUE ").replace("_", " ")
             for n in names]

    fig, axes = plt.subplots(len(TOL), 1, figsize=(12.0, 9.0), sharex=True)
    x = np.arange(len(names), dtype=float)
    n_flag_total = 0
    for ax, (key, title, ylab, tol) in zip(axes, TOL):
        d, miss = [], []
        for n in names:
            u = (A[n].get("fit") or {}).get(key)
            v = (B[n].get("fit") or {}).get(key)
            if u is None or v is None:
                d.append(np.nan)
                miss.append(n)
            else:
                d.append(v - u)
        d = np.array(d, float)
        lim = max(np.nanmax(np.abs(d)) * 1.35, tol * 3 if tol else 1.0)
        if tol > 0:
            ax.axhspan(-tol, tol, color="#dff0df", zorder=0)
            ax.text(len(names) - 0.4, tol, " no attention needed", fontsize=8,
                    color="darkgreen", va="bottom", ha="right")
        ax.axhline(0.0, color="darkgreen", lw=1.1, zorder=1)
        flag = np.abs(d) > tol
        n_flag_total += int(np.nansum(flag))
        ax.plot(x[~flag & np.isfinite(d)], d[~flag & np.isfinite(d)], "o", ms=7,
                color="0.45", zorder=3)
        ax.plot(x[flag], d[flag], "o", ms=9, color="firebrick", zorder=4)
        for i in np.where(flag)[0]:
            ax.annotate("%+.4g" % d[i], (x[i], d[i]), textcoords="offset points",
                        xytext=(0, 9 if d[i] >= 0 else -16), ha="center",
                        fontsize=8, color="firebrick")
        for i in np.where(~np.isfinite(d))[0]:
            ax.plot(x[i], 0, "x", ms=8, color="0.7", zorder=3)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_ylim(-lim, lim)
        ax.set_title("%s   --   %d of %d test cases changed by more than the tolerance (%g)"
                     % (title, int(np.nansum(flag)), len(names), tol), fontsize=10, loc="left")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(short, rotation=35, ha="right", fontsize=8)
    fig.suptitle("%s  $\\rightarrow$  %s\nred = changed by more than the tolerance, "
                 "so the old value, the new value, or neither is right and somebody must decide"
                 % (a.label_a, a.label_b), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = a.out or os.path.join(HERE, "regression_change.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote %s" % p)
    print("%d measurement(s) changed beyond tolerance across %d test cases"
          % (n_flag_total, len(names)))


if __name__ == "__main__":
    main()
