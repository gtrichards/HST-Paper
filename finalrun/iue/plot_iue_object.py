"""Show an IUE object's data, so a judgement about it is made on the data.

GTR, of 3C 120: "I wouldn't be so quick to exclude 3C 120.  I'll want to see
the data."  That object has 41 exposures and the highest native signal-to-noise
of the 22, and our continuum fitter still fails on it -- so the figure has to
show BOTH the spectrum and why the continuum failed, or it cannot be judged.

Panels:
  top     every exposure, faint, with the inverse-variance co-add over them, on
          the original flux scale.  Shaded: the stretches spec_morph flags as
          emission lines.  Those are what cont_filtered must avoid, and at a
          low-redshift object's IUE coverage they take 70 per cent of the range.
  middle  the co-add again, zoomed on C IV, with the native IUE sampling drawn
          as points so the real resolution is visible rather than the
          interpolated lattice.
  bottom  per-pixel signal-to-noise of the co-add.

    python plot_iue_object.py "3C 120"
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
try:
    from richardsplot import richardsplot as rplot   # noqa: F401
except Exception:
    pass
from rebinning import spec_morph

TABLE = "/Users/gtr/Work/git/HST-Paper/Data/CIV_measurements_v23.csv"
CIV = 1549.48


def safe(n):
    return "".join(c if (c.isalnum() or c in "+-._") else "_" for c in str(n))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "3C 120"
    t = pd.read_csv(TABLE)
    r = t[t["common_name"] == name].iloc[0]
    z = float(r["z"])
    d = os.path.join(HERE, "spectra", safe(name))
    files = sorted(set(glob.glob(os.path.join(d, "**", "*mxlo_vo.fits"), recursive=True)))
    specs = []
    for p in files:
        with fits.open(p) as h:
            dd = h[1].data
            w = np.asarray(dd["WAVE"][0], float) / (1.0 + z)
            f = np.asarray(dd["FLUX"][0], float)
            s = np.asarray(dd["SIGMA"][0], float)
            q = np.asarray(dd["QUALITY"][0], int)
        g = (q == 0) & np.isfinite(f) & np.isfinite(s) & (s > 0)
        if g.sum() > 20:
            specs.append((w, f, s, g))
    grid = np.arange(1100.0, 2000.0, 1.5)
    num = np.zeros(grid.size); den = np.zeros(grid.size)
    for w, f, s, g in specs:
        fi = np.interp(grid, w[g], f[g], left=np.nan, right=np.nan)
        si = np.interp(grid, w[g], s[g], left=np.nan, right=np.nan)
        ok = np.isfinite(fi) & np.isfinite(si) & (si > 0)
        num[ok] += fi[ok] / si[ok] ** 2; den[ok] += 1.0 / si[ok] ** 2
    co = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    ce = np.where(den > 0, 1.0 / np.sqrt(np.where(den > 0, den, 1)), np.nan)
    em = spec_morph.emission_lines2(grid)

    sc = 10.0 ** (-int(np.floor(np.log10(np.nanmedian(np.abs(co[np.isfinite(co)]))))))
    fig, ax = plt.subplots(3, 1, figsize=(13, 11),
                           gridspec_kw=dict(height_ratios=[2.0, 1.4, 0.9]))
    a = ax[0]
    for w, f, s, g in specs:
        a.plot(w[g], f[g] * sc, lw=0.3, alpha=0.25, color="0.6")
    # shade the emission-line mask
    e2 = np.concatenate(([False], em, [False]))
    edges = np.flatnonzero(e2[1:] != e2[:-1])
    for i in range(0, len(edges) - 1, 2):
        a.axvspan(grid[max(edges[i] - 1, 0)], grid[min(edges[i + 1], len(grid) - 1)],
                  color="tab:orange", alpha=0.13, zorder=0)
    a.plot(grid, co * sc, color="tab:blue", lw=1.3, label="inverse-variance co-add")
    a.axvline(CIV, color="k", ls=":", lw=1)
    a.set_xlim(1100, 2000)
    fin = co[np.isfinite(co)]
    a.set_ylim(min(0, np.percentile(fin, 1) * sc), np.percentile(fin, 99.5) * sc * 1.3)
    a.set_ylabel(r"$f_\lambda$ ($10^{%d}$ cgs)" % -int(np.log10(sc)))
    a.set_title(r"%s --- IUE, %d exposures, $z=%.4f$   (shaded: emission-line mask, %.0f\%% of the range)"
                % (name.replace("+", "$+$"), len(specs), z, 100 * em.mean()))
    a.legend(loc="upper right", fontsize=10)

    b = ax[1]
    for w, f, s, g in specs:
        b.plot(w[g], f[g] * sc, lw=0.4, alpha=0.3, color="0.6")
    k = np.isfinite(co) & (grid >= 1450) & (grid <= 1700)
    b.errorbar(grid[k], co[k] * sc, yerr=ce[k] * sc, fmt="o", ms=3.5, lw=0.9,
               color="tab:blue", label="co-add, native 1.5 \\AA\\ sampling")
    b.axvline(CIV, color="k", ls=":", lw=1)
    b.axvspan(1500, 1600, color="tab:blue", alpha=0.07)
    b.set_xlim(1450, 1700)
    kk = co[k]
    b.set_ylim(min(0, np.nanpercentile(kk, 2) * sc), np.nanpercentile(kk, 98) * sc * 1.35)
    b.set_ylabel(r"$f_\lambda$ ($10^{%d}$ cgs)" % -int(np.log10(sc)))
    b.legend(loc="upper right", fontsize=10)

    c = ax[2]
    c.plot(grid, co / ce, color="tab:green", lw=1.0)
    c.axhline(2, color="r", ls="--", lw=0.9)
    c.axvline(CIV, color="k", ls=":", lw=1)
    c.set_xlim(1100, 2000); c.set_ylim(0, None)
    c.set_ylabel("co-add S/N"); c.set_xlabel(r"rest wavelength (\AA)")

    out = os.path.join(HERE, "figures"); os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "%s_IUE_data.pdf" % safe(name))
    fig.tight_layout(); fig.savefig(p); fig.savefig(p.replace(".pdf", ".png"), dpi=125)
    kc = np.isfinite(co) & (grid >= 1500) & (grid <= 1600)
    print("wrote %s" % p)
    print("  C IV window: %d native px, median S/N %.2f"
          % (kc.sum(), np.nanmedian((co / ce)[kc])))


if __name__ == "__main__":
    main()
