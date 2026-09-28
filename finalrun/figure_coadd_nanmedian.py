"""Before-and-after figure for the NaN-corrupted weighted median in the co-adder.

The co-adder stores every exposure across the full lattice, so at any pixel the
exposures that do not reach it sit in the column as NaN with zero weight.
weightedstats.numpy_weighted_median starts with sorted(zip(data, weights)), and
Python cannot sort tuples whose first element is NaN -- every comparison against
NaN is False -- so the real values are left out of ascending order and the
cumulative-weight walk returns an arbitrary contributing exposure rather than
their median.

This draws the contributing exposures themselves behind both versions of the
co-add so the failure is visible: the broken median hops between exposures pixel
to pixel, while the corrected one tracks their weighted centre.

    python figure_coadd_nanmedian.py [OBJECT] [--inst COS] [--zoom LO HI]
"""
import argparse
import glob
import os
import sys

import numpy as np
import weightedstats as ws
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from rebinning import read_spec_data, spec_morph
from rebinning import lower_res_rebin as LR

HERE = os.path.dirname(os.path.abspath(__file__))
MAST = os.path.join(HERE, "data_v23", "MAST_v23")
SDSS_REF = "/Users/gtr/Work/git/HST-Paper/Data/spec-0266-51630-0080.fits"
CIV = (1500.0, 1600.0)


def place(name, path, z, inst="COS"):
    """Rebin every exposure and lay it on the shared lattice, as coadd.rebin does."""
    w, f, e, m = read_spec_data.read_data_flat(name, path, inst, z)
    hd = fits.open(SDSS_REF)
    c0, c1 = hd[0].header["coeff0"], hd[0].header["coeff1"]
    n0 = (min(np.log10(w[w != 0])) - c0) // c1
    n1 = (max(np.log10(w[w != 0])) - c0) // c1
    lam = 10 ** np.arange(c0 + n0 * c1, c0 + n1 * c1 + c1, c1)
    F = np.full((w.shape[0], len(lam)), np.nan)
    E = np.zeros_like(F)
    M = np.zeros_like(F)
    for i in range(w.shape[0]):
        nw, nf, ne, nm = LR.HSTLowResRebin(w[i, :], f[i, :], e[i, :], m[i, :], name, z)
        cont = spec_morph.cont_filtered(nw, nf, z, name)
        a = max(0, np.argmin(np.abs(nw[0] - lam)) - 1)
        F[i, a:a + len(nw)] = nf / cont
        E[i, a:a + len(nw)] = ne / cont
        M[i, a:a + len(nw)] = nm
    return lam, F, E, M


def weights_of(E, M, lam, z):
    W = np.ones(E.shape)
    for i in range(E.shape[1]):
        if np.nansum(E[:, i] ** 2) != 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                W[:, i] = 1 / E[:, i] ** 2
            W[~np.isfinite(W[:, i]), i] = 0.0
    em = spec_morph.emission_lines2(lam / (1 + z))
    W[(M > 0) & (~em)] = 0.0
    return W


def coadd(F, E, W, drop_nan):
    out = np.full(F.shape[1], np.nan)
    oe = np.zeros(F.shape[1])
    for i in range(F.shape[1]):
        v, er, wt = F[:, i], E[:, i], W[:, i]
        if drop_nan:
            g = np.isfinite(v) & np.isfinite(er) & (er > 0) & np.isfinite(wt) & (wt > 0)
            if not g.any():
                continue
            v, er, wt = v[g], er[g], wt[g]
        mf = ws.numpy_weighted_median(v, weights=wt)
        me = ws.numpy_weighted_median(er, weights=wt)
        out[i] = np.nan if mf is None else mf
        oe[i] = 0.0 if me is None else me
    return out, oe


def exposure_disagreement(F, E, sel, contrib):
    """How far apart the contributing exposures are, in units of their own errors.

    1 means they scatter about each other no more than their errors allow, which
    is the condition under which co-adding them is simply averaging down noise.
    Much above 1 means the exposures disagree about the spectrum -- a variable
    source between epochs, a calibration or normalisation offset between
    gratings, a bad exposure -- and combining them all is then averaging things
    that are not measurements of the same quantity.
    """
    if len(contrib) < 2:
        return np.nan
    f = F[np.ix_(contrib, np.where(sel)[0])]
    e = E[np.ix_(contrib, np.where(sel)[0])]
    good = np.isfinite(f) & np.isfinite(e) & (e > 0)
    n = good.sum(axis=0)
    cols = n >= 2
    if not cols.any():
        return np.nan
    fg = np.where(good, f, np.nan)
    eg = np.where(good, e, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        med = np.nanmedian(fg, axis=0)
        chi2 = np.nansum(((fg - med) / eg) ** 2, axis=0)
    dof = np.maximum(n - 1, 1)
    r = np.sqrt(chi2[cols] / dof[cols])
    return float(np.nanmedian(r))


def band_offsets(F, E, sel, contrib):
    """Per-exposure coherent offset from the other exposures, across the window.

    The per-pixel chi-square misses a difference that is small pixel by pixel
    but points the same way everywhere: on TON S180 the two STIS exposures agree
    to 0.8 sigma per pixel while one sits systematically above the other through
    the whole red wing of C IV.

    The offset is taken as a MEDIAN of the per-pixel differences, not a mean.  A
    mean is not usable here: NGC 6814's STIS has a single exposure that plunges
    to -3300 in continuum-normalised units at 1571 A, and that one pixel alone
    drove the mean offset to 7000 per cent of the flux, which said nothing about
    whether the exposures agree.  Catastrophic pixels are counted separately and
    are a different problem from a level offset.

    Returns (offset in sigma, offset as a fraction of the flux, n catastrophic
    pixels over all exposures).
    """
    if len(contrib) < 2:
        return np.nan, np.nan, 0
    cols = np.where(sel)[0]
    f = F[np.ix_(contrib, cols)]
    e = E[np.ix_(contrib, cols)]
    good = np.isfinite(f) & np.isfinite(e) & (e > 0)
    with np.errstate(invalid="ignore"):
        ref = np.nanmedian(np.where(good, f, np.nan), axis=0)
    worst_sig, worst_frac, n_bad = 0.0, 0.0, 0
    for k in range(len(contrib)):
        g = good[k] & np.isfinite(ref)
        if g.sum() < 20:
            continue
        d = f[k][g] - ref[g]
        n_bad += int((np.abs(d) > 10 * e[k][g]).sum())
        off = np.median(d)
        # uncertainty on a median is ~1.253 times that on a mean of the same n
        sem = 1.2533 * np.sqrt(np.sum(e[k][g] ** 2)) / g.sum()
        scale = np.nanmedian(np.abs(ref[g]))
        sig = np.abs(off) / sem if sem > 0 else np.nan
        frac = np.abs(off) / scale if scale > 0 else np.nan
        if np.isfinite(sig):
            worst_sig = max(worst_sig, float(sig))
        if np.isfinite(frac):
            worst_frac = max(worst_frac, float(frac))
    return worst_sig, worst_frac, n_bad


def roughness(f, e, sel):
    g = sel & np.isfinite(f) & (e > 0)
    if g.sum() < 10:
        return np.nan
    d = np.median(np.abs(np.diff(f[g]))) / np.sqrt(2) / 0.6745
    return d / np.median(e[g])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("object", nargs="?", default="SWIFT J2325.6+2157")
    ap.add_argument("--z", type=float, default=0.12)
    ap.add_argument("--inst", default="COS")
    ap.add_argument("--zoom", type=float, nargs=2, default=(1543.0, 1558.0))
    ap.add_argument("--out", default=os.path.join(HERE, "pipeline_output", "final",
                                                  "figure_coadd_nanmedian.png"))
    a = ap.parse_args()

    path = os.path.join(MAST, a.object, a.inst)
    if not os.path.isdir(path):
        sys.exit("no retrieved exposures at %s" % path)
    lam, F, E, M = place(a.object, path, a.z, a.inst)
    rest = lam / (1 + a.z)
    W = weights_of(E, M, lam, a.z)
    f_bad, e_bad = coadd(F, E, W, drop_nan=False)
    f_ok, e_ok = coadd(F, E, W, drop_nan=True)

    sel = (rest >= CIV[0]) & (rest <= CIV[1])
    contrib = [i for i in range(F.shape[0]) if np.isfinite(F[i, sel]).sum() > 10]
    n_nan = F.shape[0] - len(contrib)
    r_bad, r_ok = roughness(f_bad, e_bad, sel), roughness(f_ok, e_ok, sel)
    disagree = exposure_disagreement(F, E, sel, contrib)
    off_sig, off_frac, n_bad = band_offsets(F, E, sel, contrib)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 7.6), sharey="row")
    panels = [("before  --  NaN rows passed to the median", f_bad, "#c0392b", r_bad),
              ("after  --  non-contributing rows dropped first", f_ok, "#1b6ca8", r_ok)]
    for col, (title, fco, colour, rgh) in enumerate(panels):
        for row, (lo, hi) in enumerate([CIV, tuple(a.zoom)]):
            ax = axes[row, col]
            s = (rest >= lo) & (rest <= hi)
            # With only a handful of exposures, colour them individually: the
            # question of whether they agree well enough to co-add at all is
            # unanswerable from an undifferentiated grey band.
            if len(contrib) <= 8:
                cyc = plt.cm.tab10(np.linspace(0, 1, 10))
                for n, i in enumerate(contrib):
                    ax.plot(rest[s], F[i, s], color=cyc[n % 10], lw=0.9, alpha=0.85,
                            zorder=2, label=("exposure %d" % (n + 1)) if row == 0 and col == 0 else None)
            else:
                for i in contrib:
                    ax.plot(rest[s], F[i, s], color="0.75", lw=0.6, zorder=2)
            # Dashed so an exposure the co-add sits exactly on top of stays visible
            # -- with one contributing exposure the co-add IS that exposure, and a
            # solid line hid it completely.
            ax.plot(rest[s], fco[s], color=colour, lw=1.6, zorder=3,
                    dashes=(5, 2), solid_capstyle="butt")
            ax.set_xlim(lo, hi)
            if row == 0:
                ax.set_title("%s\n%d exposures (+%d NaN rows)    scatter/error %.2f"
                             % (title, len(contrib), n_nan, rgh), fontsize=11)
                ax.axvspan(a.zoom[0], a.zoom[1], color="#f5d76e", alpha=0.35, zorder=0)
            else:
                ax.set_xlabel(r"Rest wavelength ($\AA$)")
            if col == 0:
                ax.set_ylabel("Continuum-normalised flux")
    fig.suptitle("%s %s  --  C IV window; contributing exposures thin, co-add dashed."
                 "   Exposures disagree at %.1f sigma per pixel; "
                 "worst offset %.0f sigma (%.0f%% of the flux); %d catastrophic pixels"
                 % (a.object, a.inst, disagree, off_sig, 100 * off_frac, n_bad), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("contributing exposures: %d   NaN rows in each column: %d" % (len(contrib), n_nan))
    print("scatter/error over C IV   before %.2f   after %.2f" % (r_bad, r_ok))
    print("exposure disagreement     %.2f sigma per pixel" % disagree)
    print("worst coherent offset     %.1f sigma  (%.1f%% of the flux)"
          % (off_sig, 100 * off_frac))
    print("catastrophic pixels       %d (any exposure more than 10 sigma off the rest)" % n_bad)
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
