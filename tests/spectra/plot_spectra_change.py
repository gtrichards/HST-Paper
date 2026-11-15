"""The test-set spectra themselves, before and after a change to the code.

GTR, 2026-11-14: "regression_change.png is good, but we should also be able to
visualize the spectra."

regression_change.png says WHICH test cases moved and by how much.  This says
WHAT moved in the data: the two versions of each product overlaid across the
C IV window, with their difference beneath.  Between them the question "is the
old value, the new value, or neither correct" has something to look at.

Panels whose measured quantities changed beyond tolerance are outlined in red,
so the figure and the summary agree on what needs attention.

THE LOWER PANEL CARRIES TWO TRACES, and the second one exists because the first
was not enough.  Black is the change in flux.  Red is the ratio of the error
arrays, after / before, on the right-hand axis, flat at 1 when they agree.  The
first version of this figure drew flux alone, and against a change that touches
only the error array -- the co-add error question is exactly that -- every
difference panel was flat zero while eleven measurements moved, which reads as
"nothing changed" and is the opposite of the truth.  The fit is error-weighted
and maskNAL thresholds on the errors, so an error-only change moves the answer
without moving a single flux value.

    python plot_spectra_change.py BEFORE_DIR AFTER_DIR \
        --json-a a1.json a2.json --json-b b1.json b2.json
"""
import argparse
import glob
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
TOL_EW, TOL_L = 0.1, 0.01
LO, HI = 1450.0, 1700.0


def load_json(paths):
    out = {}
    for p in paths or []:
        out.update(json.load(open(p)))
    return out


def norm(w, f, m, e):
    """Continuum-normalise on the window's own wings so two versions are comparable.

    Returns the RAW error alongside, not the normalised one, because the lower
    panel wants the honest ratio of the two error arrays."""
    g = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0) & (w >= LO) & (w <= HI)
    if g.sum() < 30:
        return None, None, None
    wing = g & (((w >= LO) & (w <= 1480)) | ((w >= 1640) & (w <= HI)))
    if wing.sum() < 10:
        return None, None, None
    sl, ic = np.polyfit(w[wing], f[wing], 1)
    return w[g], f[g] / (sl * w[g] + ic), e[g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--json-a", nargs="+", default=None)
    ap.add_argument("--json-b", nargs="+", default=None)
    ap.add_argument("--label-a", default="before")
    ap.add_argument("--label-b", default="after")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A, B = load_json(a.json_a), load_json(a.json_b)

    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.before, "*.npz"))
                   if os.path.exists(os.path.join(a.after, os.path.basename(p))))
    ncol = 4
    nblock = int(np.ceil(len(names) / ncol))
    fig = plt.figure(figsize=(4.3 * ncol, 3.6 * nblock))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(nblock * 2, ncol, figure=fig,
                           height_ratios=[3, 1.2] * nblock, wspace=0.52)
    gs.update(hspace=0.62)

    n_changed = 0
    for i, n in enumerate(names):
        r, c = divmod(i, ncol)
        ax = fig.add_subplot(gs[2 * r, c])
        axd = fig.add_subplot(gs[2 * r + 1, c], sharex=ax)
        da = np.load(os.path.join(a.before, n + ".npz"))
        db = np.load(os.path.join(a.after, n + ".npz"))
        wa, fa, ea = norm(da["wave"], da["flux"], da["mask"], da["err"])
        wb, fb, eb = norm(db["wave"], db["flux"], db["mask"], db["err"])
        if wa is None or wb is None:
            ax.text(0.5, 0.5, "no usable C IV window", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="0.4")
            ax.set_title(n.replace("cartoon_", "~"), fontsize=9)
            ax.set_xticks([]); axd.set_xticks([])
            continue
        ax.plot(wa, fa, "-", color="0.55", lw=1.6, zorder=2, label=a.label_a)
        ax.plot(wb, fb, "-k", lw=0.9, zorder=3, label=a.label_b)
        ax.axvspan(1500, 1600, color="#eef3fa", zorder=0)
        ax.axvline(1549.48, color="0.6", lw=0.6, ls=":", zorder=1)

        common = np.intersect1d(np.round(wa, 3), np.round(wb, 3))
        if common.size > 20:
            ia = np.isin(np.round(wa, 3), common)
            ib = np.isin(np.round(wb, 3), common)
            k = min(ia.sum(), ib.sum())
            axd.plot(wa[ia][:k], (fb[ib][:k] - fa[ia][:k]), "-k", lw=0.8)
            # the error array, as a ratio -- see the module docstring
            ra, rb = ea[ia][:k], eb[ib][:k]
            kk = k
            if kk > 20:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rat = rb / ra
                axe = axd.twinx()
                # per-pixel ratio faint behind a running median: the pixel-to-pixel
                # scatter is real and worth seeing, but a talk needs the LEVEL,
                # and the raw trace alone is an unreadable hedge of spikes.
                axe.plot(wa[ia][:kk], rat, "-", color="firebrick", lw=0.4,
                         alpha=0.30, zorder=1)
                nmed = 25
                if kk > nmed:
                    t = rat[: (kk // nmed) * nmed].reshape(-1, nmed)
                    xm = wa[ia][: (kk // nmed) * nmed].reshape(-1, nmed).mean(1)
                    axe.plot(xm, np.nanmedian(t, axis=1), "-", color="firebrick",
                             lw=1.5, zorder=2)
                axe.axhline(1.0, color="firebrick", lw=0.6, ls=":")
                med = float(np.nanmedian(rat))
                axe.text(0.985, 0.08, r"error $\times$%.2f" % med, transform=axe.transAxes,
                         ha="right", va="bottom", fontsize=7, color="firebrick",
                         bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
                lo_r = np.nanpercentile(rat, 2); hi_r = np.nanpercentile(rat, 98)
                pad = max(0.05, 0.15 * (hi_r - lo_r))
                axe.set_ylim(min(lo_r - pad, 0.95), max(hi_r + pad, 1.05))
                axe.tick_params(labelsize=6.5, colors="firebrick")
                axe.set_ylabel("error ratio", fontsize=6.5, color="firebrick",
                               labelpad=1)
        axd.axhline(0.0, color="k", lw=0.5, ls=":")
        axd.set_ylim(-0.25, 0.25)
        axd.set_ylabel("flux: after $-$ before", fontsize=6.5, labelpad=1)
        axd.set_xlabel(r"rest wavelength ($\AA$)", fontsize=8)
        axd.tick_params(labelsize=7)
        ax.tick_params(labelsize=7, labelbottom=False)
        ax.set_ylabel("flux / continuum", fontsize=8)
        ax.set_xlim(LO, HI)

        fa_, fb_ = (A.get(n) or {}).get("fit") or {}, (B.get(n) or {}).get("fit") or {}
        bits, flag = [], False
        for key, lab, tol, fmt in (("blueshift", "v", 0.0, "%+.0f"),
                                   ("ew", "EW", TOL_EW, "%+.2f"),
                                   ("logL1450", "logL", TOL_L, "%+.3f")):
            u, v = fa_.get(key), fb_.get(key)
            if u is None or v is None:
                continue
            d = v - u
            if abs(d) > tol:
                flag = True
                bits.append("%s %s" % (lab, fmt % d))
        if flag:
            n_changed += 1
            for sp in ax.spines.values():
                sp.set_color("firebrick"); sp.set_linewidth(1.8)
        ax.set_title(n.replace("cartoon_", "~").replace("_", " "), fontsize=9,
                     color=("firebrick" if flag else "black"))
        if bits:
            ax.text(0.03, 0.95, "\n".join(bits), transform=ax.transAxes, va="top",
                    ha="left", fontsize=7.5, color="firebrick",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper right")

    fig.suptitle("The test-set spectra, %s $\\rightarrow$ %s\n"
                 "upper: grey = before, black = after, shaded = the C IV window.  "
                 "lower: black = change in flux, RED = error array after/before.  "
                 "red outline = a measured quantity moved beyond tolerance (%d of %d)"
                 % (a.label_a, a.label_b, n_changed, len(names)), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = a.out or os.path.join(HERE, "regression_spectra.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print("wrote %s  (%d of %d panels flagged)" % (p, n_changed, len(names)))


if __name__ == "__main__":
    main()
