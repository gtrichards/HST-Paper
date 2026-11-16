"""Figures of what the analysis code does to spectra whose answer we know.

GTR asked for plots that can be shown in a talk, then said of the first attempt
"I don't know how to interpret those plots" -- so these are built to be read
without a caption:

  cartoons_overview.png   per cartoon, the injected line and what the pipeline
                          returned, with a RESIDUAL panel beneath so a departure
                          is visible rather than inferred.  Each panel is
                          labelled PASS or FAIL against a stated criterion.
There WAS a second figure here, cartoons_errors.png, asking whether the reported
error bar was the right size.  GTR, 2026-11-13: "I don't care 'if the pipeline
reports the right size error bar'.  I just care that the pipeline reports the
'correct' EW, blueshift, and L1450."  It was removed rather than kept as an
option, because an error-bar figure invites exactly the argument he ruled out.
What the error array does to a measurement is visible in regression_change.py,
where it belongs: as a change in the three numbers.

    python plot_cartoons.py
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PAPER)
sys.path.insert(0, os.path.join(PAPER, "finalrun"))
CART = os.path.join(HERE, "cartoons")
CIV = 1549.48

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
try:
    from richardsplot import richardsplot as rplot
except Exception:
    pass

#: a cartoon passes if the blueshift is recovered inside one lattice pixel and
#: the EW inside 10 per cent -- stated here so the figure can show it
TOL_V = 69.03
TOL_EW_FRAC = 0.10
#: and no pixel may sit further than this many of its own error bars from the
#: injected truth.  Added after the first version marked zero_error_pixel PASS
#: while the figure plainly showed a spike: velocity and EW are integrals and
#: survive a few bad pixels, so a criterion built only on them cannot see one.
#: 20, not 6.  Each panel holds about 500 pixels, so the largest |z| from pure
#: noise runs to 4-6 sigma and a 6-sigma cut fails healthy cartoons.  The defect
#: this criterion exists to catch sits at 131 sigma, so the two are separated by
#: more than an order of magnitude and the threshold does not need to be fine.
TOL_RESID_SIGMA = 20.0

TITLES = {
    "single_clean": "one exposure\n(the control)",
    "independent_readouts": "20 independent readouts\n(must be averaged)",
    "cumulative_groups": "20 cumulative groups\n(use the last only)",
    "nine_agreeing": "9 identical exposures",
    "nine_disagreeing": "9 exposures, levels differ 20%\n(normalisation removes it)",
    "nine_varying_line": "9 exposures, LINE strength\nEW 30-50 A (a real disagreement)",
    "foreign_target": "8 exposures, 5 of them\nANOTHER QUASAR (TARGNAME ignored)",
    "nine_tilted": "9 exposures, 3 tilted 15%",
    "zero_error_pixel": "4 pixels with no signal\nand a tiny quoted error",
    "two_epochs_offset": "2 gratings, 25% apart\n(different epochs)",
}
#: ORDER, not the glob, decides what is drawn -- so a cartoon missing from this
#: list is silently absent from the figure.  cartoon_nine_varying_line was added
#: on 2026-11-14 and did not appear here until it was noticed by its timestamp.
ORDER = ["single_clean", "independent_readouts", "cumulative_groups",
         "zero_error_pixel", "nine_agreeing", "nine_disagreeing",
         "nine_varying_line", "nine_tilted", "two_epochs_offset",
         "foreign_target"]


def build(name):
    from astropy.table import Table
    from rebinning import coadd as CO
    meta = json.load(open(os.path.join(CART, name, "fixture.json")))
    z = meta["z"]
    work = tempfile.mkdtemp(prefix="cartplot_")
    try:
        stage, reb = os.path.join(work, "s"), os.path.join(work, "r")
        os.makedirs(stage); os.makedirs(reb)
        for p in sorted(glob.glob(os.path.join(CART, name, "exposures", "*"))):
            shutil.copy2(p, stage)
        CO.rebin(meta["obj"], z, "FOS", None, data_path=stage, output_dir=reb, flat=True)
        t = Table.read(sorted(glob.glob(os.path.join(reb, "*.fits")))[0])
        w = np.asarray(t["Rest-frame Wavelength"], float)
        f = np.asarray(t["Coadded Flux (Arbitrary Units)"], float)
        e = np.asarray(t["Coadded Flux Errors"], float)
        m = np.asarray(t["Bad Pixel Mask"], float)
        g = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)
        return meta, (w, f, e, g)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def measure(w, f, g):
    sel = g & (w >= 1400) & (w <= 1700)
    wing = sel & (((w >= 1400) & (w <= 1480)) | ((w >= 1640) & (w <= 1700)))
    sl, ic = np.polyfit(w[wing], f[wing], 1)
    k = sel & (w >= 1500) & (w <= 1600)
    cont = sl * w[k] + ic
    dl = np.gradient(w[k])
    ew = float(np.sum((f[k] - cont) / cont * dl))
    line = np.clip(f[k] - cont, 0, None)
    cum = np.cumsum(line * dl)
    lam = float(np.interp(cum[-1] / 2.0, cum, w[k]))
    return (CIV - lam) / CIV * 299792.458, ew, (sl, ic)


def overview():
    names = [n for n in ORDER if os.path.isdir(os.path.join(CART, "cartoon_" + n))]
    ncol = 4
    nblock = int(np.ceil(len(names) / ncol))
    fig = plt.figure(figsize=(4.3 * ncol, 4.3 * nblock))
    gs = gridspec.GridSpec(nblock * 2, ncol, figure=fig,
                           height_ratios=[3, 1.25] * nblock, wspace=0.30)
    # blocks need real space between them or the lower row's titles land inside
    # the upper row's residual panels
    gs.update(hspace=0.75)
    for i, short in enumerate(names):
        r, c = divmod(i, ncol)
        ax = fig.add_subplot(gs[2 * r, c])
        axr = fig.add_subplot(gs[2 * r + 1, c], sharex=ax)
        meta, (w, f, e, g) = build("cartoon_" + short)
        tr = meta["truth"]
        b, ew, (sl, ic) = measure(w, f, g)

        k = g & (w >= 1460) & (w <= 1640)
        cont = sl * w[k] + ic
        obs = f[k] / cont
        err = e[k] / cont
        lam0 = CIV * (1.0 - tr["blueshift_kms"] / 299792.458)
        sig = lam0 * (tr["fwhm_kms"] / 299792.458) / 2.3548
        peak = tr["ew_A"] / (sig * np.sqrt(2.0 * np.pi))
        inj = 1.0 + peak * np.exp(-0.5 * ((w[k] - lam0) / sig) ** 2)

        ax.fill_between(w[k], obs - err, obs + err, color="0.82", lw=0, zorder=1)
        ax.plot(w[k], inj, "-", color="firebrick", lw=2.0, zorder=4)
        ax.plot(w[k], obs, "-k", lw=0.9, zorder=5)
        ok_v = abs(b - tr["blueshift_kms"]) <= TOL_V
        ok_e = abs(ew - tr["ew_A"]) <= TOL_EW_FRAC * tr["ew_A"]
        worst = float(np.nanmax(np.abs((obs - inj) / np.where(err > 0, err, np.nan))))
        ok_r = worst <= TOL_RESID_SIGMA
        good = ok_v and ok_e and ok_r
        ax.set_title(TITLES[short], fontsize=9.5)
        ax.text(0.03, 0.96, "PASS" if good else "FAIL",
                transform=ax.transAxes, va="top", ha="left", fontsize=11,
                color=("darkgreen" if good else "firebrick"), fontweight="bold")
        ax.text(0.97, 0.97,
                "velocity %+.0f km/s   %s\nEW %.1f vs 40 $\\AA$   %s\n"
                "worst pixel %.1f$\\sigma$   %s"
                % (b - tr["blueshift_kms"], "ok" if ok_v else "OFF",
                   ew, "ok" if ok_e else "OFF",
                   worst, "ok" if ok_r else "OFF"),
                transform=ax.transAxes, va="top", ha="right", fontsize=8,
                bbox=dict(fc="white", ec="0.8", lw=0.5, pad=2.5))
        ax.set_ylabel("flux / continuum", fontsize=8.5)
        ax.tick_params(labelsize=7.5, labelbottom=False)
        ax.set_ylim(0.4, 4.2)

        axr.axhline(0.0, color="firebrick", lw=1.2, zorder=3)
        axr.fill_between(w[k], -err, err, color="0.82", lw=0, zorder=1)
        axr.plot(w[k], obs - inj, "-k", lw=0.9, zorder=4)
        axr.set_ylim(-0.6, 0.6)
        axr.set_ylabel("out $-$ in", fontsize=8.5)
        axr.set_xlabel(r"rest wavelength ($\AA$)", fontsize=8.5)
        axr.tick_params(labelsize=7.5)
        axr.set_xlim(1465, 1635)

    fig.suptitle("Synthetic spectra: a C IV line of known blueshift (+1000 km/s) and known "
                 "strength (EW 40 $\\AA$) is put in, and the pipeline's answer compared\n"
                 "red = what went in     black = what came out     grey = the error bar the "
                 "pipeline reports\nPASS = blueshift within one 69 km/s pixel, EW within 10%, "
                 "and no pixel more than 20$\\sigma$ from the injected curve",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(HERE, "cartoons_overview.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote %s" % p)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    overview()
