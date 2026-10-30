"""Show the estimated 1450 A point on the adopted fit for the four objects
whose L1450 could not be measured.

GTR: "We should add the predicted 1450 points to the plots of the adopted best
fits for those 4.  I can then look at them."  Left as OBSERVED flux against
REST wavelength, per GTR, which is the frame the rebinned files are in.

Why each one failed, and what is being estimated:

  #51  Mrk 486          continuum starts 1508.8 A, 59 A short of the window.
                        Has BOTH L1700 and L2500, and its own continuum slope
                        is POSITIVE (+0.59, rising to the red), so the sample
                        median slope would be the wrong one to use here.
  #169 Mrk 813          continuum starts 1467.8 A -- 18 A short.
  #356 QSO0020+287      continuum starts 1467.1 A -- 17 A short.
  #220 2MASS J1021+4645 1450 A IS covered, but the Continuum Normalisation is
                        NEGATIVE there (-3.7e-17), which is the real failure.
                        Its two estimates differ by 0.26 dex; least trustworthy.

The estimate is the object's own continuum-normalisation power law, fitted
over 1200-3000 A where it is positive, evaluated at 1450 A and multiplied by
the ICA model's shape value there.  The sample-median slope (-1.064) anchored
at the nearest good pixel is drawn alongside for comparison.
"""
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

HERE = os.path.dirname(os.path.abspath(__file__))
FINALRUN = os.path.dirname(HERE)
sys.path.insert(0, FINALRUN)
try:
    from richardsplot import richardsplot as rplot   # noqa: F401
except Exception:
    pass
REBIN = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "rebin")
FITS_DIR = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "fits")
OUT = os.path.join(FINALRUN, "pipeline_output", "final", "l1450_estimates")
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
SAMPLE_SLOPE = -1.064
TARGETS = [(51, "Mrk 486"), (169, "Mrk 813"),
           (220, "2MASS J10215235+4645158"), (356, "QSO0020+287")]


def one(idx, label, ax):
    g = (glob.glob(os.path.join(FITS_DIR, "%d_*" % idx, "records", "BEST_*.json")) +
         glob.glob(os.path.join(FITS_DIR, "%03d_*" % idx, "records", "BEST_*.json")))
    r = json.load(open(g[0]))
    stem, z = r["name"], r["result"]["z"]
    d = fits.open(os.path.join(REBIN, stem + ".fits"))[1].data
    w = np.asarray(d["Rest-frame Wavelength"], float)
    fl = np.asarray(d["Coadded Flux (Arbitrary Units)"], float)
    cn = np.asarray(d["Continuum Normalisation"], float)
    bp = np.asarray(d["Bad Pixel Mask"], float)
    good = np.isfinite(fl) & (bp == 0)

    # observed f_lambda = arbitrary flux x continuum normalisation
    obs = fl * cn
    k = np.isfinite(cn) & (cn > 0) & (w >= 1200) & (w <= 3000)
    slope, inter = np.polyfit(np.log10(w[k]), np.log10(cn[k]), 1)
    cont_own = 10 ** (inter + slope * np.log10(1450.0))
    j = np.argmin(np.abs(w[k] - 1450)); w0, c0 = w[k][j], cn[k][j]
    cont_med = c0 * (1450.0 / w0) ** SAMPLE_SLOPE

    mono = r["monochromatic"]["1450"]
    f_shape = mono["f_lambda"]
    dl = cosmo.luminosity_distance(z).to("cm").value
    def L(c): return np.log10(4 * np.pi * dl ** 2 * f_shape * c * 1450.0 * (1 + z))

    s = good & (w >= 1250) & (w <= 2200)
    # f_lambda is ~1e-15, which plain tick formatting renders as a column of
    # zeros, so each panel is scaled by its own power of ten and the exponent
    # goes in the axis label.
    pos = obs[s][np.isfinite(obs[s]) & (obs[s] > 0)]
    expo = int(np.floor(np.log10(np.median(pos)))) if pos.size else -15
    sc = 10.0 ** (-expo)
    ax.plot(w[s], obs[s] * sc, color="0.45", lw=0.6, zorder=1)
    # the ICA model, on the same observed scale, via the continuum normalisation
    wi = np.asarray(r.get("wave_ica") or [], float)
    ax.axvspan(1445, 1465, color="tab:blue", alpha=0.12, zorder=0)
    ax.plot(1450, f_shape * cont_own * sc, "o", ms=9, mfc="none", mew=2,
            color="tab:red", zorder=5,
            label=r"own slope %.2f: $\log L=%.2f$" % (slope, L(cont_own)))
    ax.plot(1450, f_shape * cont_med * sc, "s", ms=8, mfc="none", mew=2,
            color="tab:green", zorder=5,
            label=r"median slope: $\log L=%.2f$" % L(cont_med))
    for lab2, wl, col in (("1700", 1700.0, "tab:purple"), ("2500", 2500.0, "tab:orange")):
        v = r["monochromatic"][lab2]
        if v.get("measured") and np.isfinite(v.get("logL", np.nan)):
            ax.plot(wl, v["f_lambda"] * v["nearest_cont"] * sc, "^", ms=8, color=col,
                    zorder=5, label=r"$L_{%s}$ measured: $\log L=%.2f$" % (lab2, v["logL"]))
    ax.set_xlim(1250, 2200)
    if pos.size:
        # 97th percentile, not the maximum: #220 is a field of noise spikes an
        # order of magnitude above its own continuum, and scaling to those
        # flattens the spectrum into the axis.
        ax.set_ylim(0, np.percentile(pos * sc, 97) * 1.6)
    ax.set_title(r"\#%d %s, $z=%.4f$" % (idx, label.replace("+", "$+$"), z), fontsize=13)
    ax.set_xlabel(r"rest wavelength (\AA)")
    ax.set_ylabel(r"observed $f_\lambda$ ($10^{%d}$\,erg\,s$^{-1}$\,cm$^{-2}$\,\AA$^{-1}$)" % expo)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (i, nm) in zip(axes.ravel(), TARGETS):
        one(i, nm, ax)
    fig.suptitle(r"Estimated 1450\,\AA\ continuum point (shaded: the 1445--1465\,\AA\ window)",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT, "l1450_estimates.pdf")
    fig.savefig(p); fig.savefig(p.replace(".pdf", ".png"), dpi=130)
    print("wrote", p)


if __name__ == "__main__":
    main()
