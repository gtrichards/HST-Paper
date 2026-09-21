"""Inspection figures for spectra, drawn so C IV is actually visible.

The first attempt at these plotted the full rest-frame range on one axis and
let matplotlib choose the limits.  With optical STIS reaching 9000 A the C IV
window came out a few pixels wide, and a single emission spike set the flux
scale, so the figures showed nothing.  Here the x range is the ICA range
1150-3100 A, a second panel zooms on 1450-1650 A, and the y limits come from
percentiles of the plotted points rather than their extremes.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

M = "/Users/gtr/Dropbox/HST/Pratsos/RebinnedSpec_master/"
N = "data_v23/RebinnedSpec_v23/"
W, F, E, MK = ("Rest-frame Wavelength", "Coadded Flux (Arbitrary Units)",
               "Coadded Flux Errors", "Bad Pixel Mask")
CIV = 1549.48
UV = (1150.0, 3100.0)
ZOOM = (1450.0, 1650.0)


def spectra(D, obj):
    out = []
    if not os.path.isdir(D):
        return out
    for f in sorted(x for x in os.listdir(D)
                    if x.endswith(".fits") and x[:-5].rpartition("_")[0] == obj):
        t = Table.read(D + f)
        w = np.asarray(t[W], float)
        if not len(w):
            continue
        fl = np.asarray(t[F], float)
        good = (np.asarray(t[MK]) == 0) & (np.asarray(t[E], float) > 0) & np.isfinite(fl)
        out.append((f[:-5].rpartition("_")[2], w, fl, good))
    return out


def panel(ax, sets, xlim, title):
    """Draw each (label, colour, spectra) set; scale y to what is in view."""
    vals = []
    for label, colour, specs in sets:
        for inst, w, fl, good in specs:
            sel = good & (w >= xlim[0]) & (w <= xlim[1])
            if sel.sum() < 2:
                continue
            ax.plot(w[sel], fl[sel], lw=0.6, color=colour, alpha=0.85,
                    label="%s %s (%d px)" % (label, inst, int(sel.sum())))
            vals.append(fl[sel])
            # mark where flagged pixels sit, along the bottom
            bad = (~good) & (w >= xlim[0]) & (w <= xlim[1])
            if bad.sum():
                ax.plot(w[bad], np.full(bad.sum(), np.nan), ".")
    ax.axvspan(CIV - 25, CIV + 25, color="tab:red", alpha=0.12, zorder=0)
    ax.axvline(CIV, color="tab:red", lw=0.8, ls="--", zorder=1)
    ax.set_xlim(*xlim)
    if vals:
        v = np.concatenate(vals)
        lo, hi = np.percentile(v, 1.0), np.percentile(v, 99.0)
        pad = 0.15 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
    else:
        ax.text(0.5, 0.5, "nothing in this range", ha="center",
                transform=ax.transAxes, fontsize=9, color="0.4")
    ax.set_title(title, fontsize=9)
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(fontsize=7, loc="upper right")


def figure(obj, header, sets, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    panel(axes[0], sets, UV, header)
    panel(axes[1], sets, ZOOM, "C IV zoom  (1450-1650 A)")
    axes[1].set_xlabel("rest wavelength (A)")
    for a in axes:
        a.set_ylabel("flux")
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


def main():
    d = pd.read_csv("pipeline_output/final/manmask_status_v23.csv")

    out = "pipeline_output/final/reconsider_plots"
    os.makedirs(out, exist_ok=True)
    sel = d[d.status.str.startswith("RECONSIDER")].sort_values("index")
    for _, r in sel.iterrows():
        o = r.common_name
        figure(o,
               "%03d  %s  (%s)   manmask %d, %s   --   grey = master, colour = regenerated"
               % (r["index"], r.name_pub, o, int(r.manmask_now), r.reason_now),
               [("master", "0.65", spectra(M, o)), ("new", "tab:blue", spectra(N, o))],
               os.path.join(out, "%03d_%s.png" % (r["index"], o.replace("/", "_"))))
    print("reconsider: %d figures -> %s" % (len(sel), out))

    out = "pipeline_output/final/nocivid_plots"
    os.makedirs(out, exist_ok=True)
    nr = d[(d.status == "not yet reached") & (d.manmask_expected == 0)].sort_values("index")
    for _, r in nr.iterrows():
        o = r.common_name
        figure(o,
               "%03d  %s  (%s)   C IV usable px = %d across %d spectra"
               % (r["index"], r.name_pub, o, int(r.civ_px), int(r.n_spectra)),
               [("", "tab:blue", spectra(N, o))],
               os.path.join(out, "%03d_%s.png" % (r["index"], o.replace("/", "_"))))
    print("no-C IV:    %d figures -> %s" % (len(nr), out))


if __name__ == "__main__":
    main()
