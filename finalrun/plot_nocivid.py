"""One figure per object that the regenerated data says has no usable C IV.

The verdict is arithmetic -- fewer than 20 unmasked pixels with a positive
error within 25 A of 1549.48 -- but zero pixels has two quite different
causes, and only the picture separates them: the spectrum may simply not
reach C IV, or it may span the line with everything there flagged or in a
detector gap. Each figure shades the C IV window, marks 1549.48, and says
which case it is, so the call can be checked by eye rather than trusted.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

N = "data_v23/RebinnedSpec_v23/"
OUT = "pipeline_output/final/nocivid_plots"
W, F, E, MK = ("Rest-frame Wavelength", "Coadded Flux (Arbitrary Units)",
               "Coadded Flux Errors", "Bad Pixel Mask")
CIV, HALF = 1549.48, 25.0

os.makedirs(OUT, exist_ok=True)
d = pd.read_csv("pipeline_output/final/manmask_status_v23.csv")
sel = d[(d.status == "not yet reached") & (d.manmask_expected == 0)].sort_values("index")

rows = []
for _, r in sel.iterrows():
    obj = r.common_name
    specs = sorted(f for f in os.listdir(N)
                   if f.endswith(".fits") and f[:-5].rpartition("_")[0] == obj)
    fig, ax = plt.subplots(figsize=(11, 4))
    reach = False
    for f in specs:
        t = Table.read(N + f)
        w = np.asarray(t[W], float); fl = np.asarray(t[F], float)
        m = np.asarray(t[MK]); er = np.asarray(t[E], float)
        if not len(w):
            continue
        if w.min() <= CIV <= w.max():
            reach = True
        good = (m == 0) & (er > 0) & np.isfinite(fl)
        ax.plot(w[good], fl[good], lw=0.6, label="%s (%d px)" % (f[:-5].rpartition("_")[2], good.sum()))
        bad = ~good & np.isfinite(w)
        if bad.sum():
            ax.plot(w[bad], np.zeros(bad.sum()), "|", ms=3, color="0.7")

    cause = ("spans C IV but every pixel there is flagged or in a gap"
             if reach else "wavelength coverage does not reach C IV")
    ax.axvspan(CIV - HALF, CIV + HALF, color="tab:red", alpha=0.15, zorder=0)
    ax.axvline(CIV, color="tab:red", lw=0.8, ls="--")
    ax.set_xlabel("rest wavelength (A)"); ax.set_ylabel("flux")
    ax.set_title("%03d  %s  (%s)   C IV usable px = %d   --   %s"
                 % (r['index'], r.name_pub, obj, r.civ_px, cause), fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "%03d_%s.png" % (r['index'], obj.replace("/", "_"))), dpi=90)
    plt.close(fig)
    rows.append(dict(index=int(r['index']), common_name=obj, name_pub=r.name_pub,
                     civ_px=int(r.civ_px), n_spectra=len(specs), cause=cause))

out = pd.DataFrame(rows)
out.to_csv("pipeline_output/final/nocivid_objects.csv", index=False)
print(out.cause.value_counts().to_string())
print("\n%d figures -> %s" % (len(out), OUT))
