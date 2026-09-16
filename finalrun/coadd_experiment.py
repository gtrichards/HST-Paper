"""
Does combining an object's HST spectra give a better C IV reconstruction than
using any one of them?

The ICA fits its components across 1260-3000 A, so C IV's shape is constrained by
the whole UV spectrum, not just the 1500-1600 A window. A spectrum that misses
C IV entirely can still improve the reconstruction by constraining the component
weights -- NGC 4051's COS spectrum covers 79% of the ICA range but only 26% of
C IV, while its STIS spectrum covers 36% of the range and 99% of C IV. Choosing
one discards most of the information either way.

So for each object this fits every spectrum on its own, then fits an
inverse-variance combination of all of them, and reports C IV blueshift, EW and
f2500 for each so the variants can be compared directly.

Combination is exact, not interpolated. The rebinned spectra are log-spaced at
1e-4 dex (69.03 km/s), and every one of the 159 within-object spectrum pairs in
this sample shares the same lattice zero-point -- measured phase offset 0.000
pixels in all cases. Pixels therefore align by index and can be summed without
resampling, which matters because interpolation would correlate neighbouring
noise and blur the very line profile being measured. The zero-point is per
object, not absolute, so it is taken from the object's first spectrum.

Only usable pixels contribute: finite flux and error, strictly positive error,
bad-pixel mask clear. Everything else is left out of the weighted sum rather than
being given a large error, so a padded or flagged region contributes nothing
instead of contributing noise.

Spectra are cross-normalised first. Instruments disagree in flux scale by several
per cent -- measured 0.99 for NGC 4151 but 0.93 for NGC 3516 and 0.87 for
NGC 4051 -- and inverse-variance averaging spectra at different flux levels
distorts the continuum, which showed up as NGC 3516's combined f2500 landing at
1.11 against 0.78 and 0.68 for its individual fits. Each spectrum therefore gets a
single multiplicative factor, the median flux ratio against the highest-S/N
spectrum over their common good pixels, applied to flux and error alike.

Part of that offset is real variability rather than calibration -- these
observations are years apart and the targets vary. That is deliberately not
corrected for: if mixing epochs damages the reconstruction, the merged spectra
will show it.
"""

import argparse
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from astropy.io import fits

sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")
WORKDIR = os.path.join(OUTDIR, "coadd_work")
PLOTS = os.path.join(OUTDIR, "coadd_plots")
QUEUE = os.path.join(OUTDIR, "work_queue.csv")
REBIN = "/Users/gtr/Dropbox/HST/Pratsos/RebinnedSpec_master"

STEP = np.log(10) / 10000.0      # 1e-4 dex == 69.03 km/s

C_WAVE = "Rest-frame Wavelength"
C_FLUX = "Coadded Flux (Arbitrary Units)"
C_ERRS = "Coadded Flux Errors"
C_MASK = "Bad Pixel Mask"
C_ZED = "Redshift"


def read_spectrum(stem):
    d = fits.getdata(os.path.join(REBIN, stem + ".fits"), 1)
    w = np.asarray(d[C_WAVE], float)
    f = np.asarray(d[C_FLUX], float)
    e = np.asarray(d[C_ERRS], float)
    try:
        b = np.asarray(d[C_MASK], float)
    except KeyError:
        b = np.zeros_like(w)
    try:
        z = float(np.asarray(d[C_ZED], float)[0])
    except (KeyError, IndexError):
        z = float("nan")
    good = np.isfinite(f) & np.isfinite(e) & (e > 0) & (b == 0)
    return w, f, e, b, z, good


def _snr(w, f, e, good):
    """Median S/N over usable pixels inside the ICA fitting range."""
    sel = good & (w >= 1260) & (w <= 3000)
    return float(np.median(f[sel] / e[sel])) if sel.sum() else 0.0


def coadd(stems):
    """Inverse-variance combine on the shared lattice. No interpolation."""
    specs = [read_spectrum(s) for s in stems]
    ref = np.log(specs[0][0][0])

    # Cross-normalise to the highest-S/N spectrum before combining.
    keys = [np.rint((np.log(sp[0]) - ref) / STEP).astype(int) for sp in specs]
    snrs = [_snr(sp[0], sp[1], sp[2], sp[5]) for sp in specs]
    iref = int(np.argmax(snrs))
    dref = {int(k): v for k, v, g in zip(keys[iref], specs[iref][1], specs[iref][5]) if g}
    scales = []
    for i, sp in enumerate(specs):
        if i == iref:
            scales.append(1.0)
            continue
        d = {int(k): v for k, v, g in zip(keys[i], sp[1], sp[5]) if g}
        r = np.array([dref[k] / d[k] for k in set(dref) & set(d)
                      if d[k] != 0 and np.isfinite(dref[k] / d[k])])
        r = r[r > 0]
        scales.append(float(np.median(r)) if r.size >= 20 else 1.0)
    specs = [(w, f * sc, e * sc, b, z, g)
             for (w, f, e, b, z, g), sc in zip(specs, scales)]
    coadd.last_scales = dict(zip(stems, [round(s, 4) for s in scales]))
    coadd.last_ref = stems[iref]

    idx, contributions = {}, []
    for (w, f, e, b, z, good) in specs:
        k = np.rint((np.log(w) - ref) / STEP).astype(int)
        # Confirm the assumption rather than trusting it: a spectrum off this
        # lattice would silently smear if we aligned it by index.
        resid = np.abs((np.log(w) - ref) / STEP - k)
        if resid.max() > 0.05:
            raise ValueError("spectrum is off the shared lattice by %.3f px"
                             % resid.max())
        contributions.append((k, f, e, good))
        for kk in k[good]:
            idx[int(kk)] = True

    if not idx:
        raise ValueError("no usable pixels in any spectrum")
    ks = np.array(sorted(idx))
    pos = {int(k): i for i, k in enumerate(ks)}

    wsum = np.zeros(ks.size)
    fsum = np.zeros(ks.size)
    nspec = np.zeros(ks.size, int)
    for (k, f, e, good) in contributions:
        sel = np.flatnonzero(good)
        if sel.size == 0:
            continue
        j = np.array([pos[int(k[i])] for i in sel])
        wt = 1.0 / (e[sel] ** 2)
        np.add.at(wsum, j, wt)
        np.add.at(fsum, j, wt * f[sel])
        np.add.at(nspec, j, 1)

    flux = np.where(wsum > 0, fsum / np.where(wsum > 0, wsum, 1.0), np.nan)
    errs = np.where(wsum > 0, 1.0 / np.sqrt(np.where(wsum > 0, wsum, 1.0)), np.nan)
    wave = np.exp(ref + ks * STEP)
    mask = np.where(wsum > 0, 0.0, 1.0)
    return wave, flux, errs, mask, specs[0][4], nspec


def write_fits(path, wave, flux, errs, mask, z):
    cols = fits.ColDefs([
        fits.Column(name=C_WAVE, format="D", array=wave),
        fits.Column(name=C_FLUX, format="D", array=flux),
        fits.Column(name=C_ERRS, format="D", array=errs),
        fits.Column(name=C_MASK, format="D", array=mask),
        fits.Column(name=C_ZED, format="D", array=np.full(wave.size, z)),
    ])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
        path, overwrite=True)


def fit(rebin_path, name, plots_dir=None):
    """Fit, and optionally render the same diagnostic figure the GUI produces.

    The plot is the point of the exercise as much as the numbers: whether merging
    epochs damages the reconstruction is something to be seen in the spectrum, not
    inferred from three scalars.
    """
    import matplotlib
    matplotlib.use("Agg")
    from ica.manual_fix import ICAManualFixProcessor
    proc = ICAManualFixProcessor(rebin_path=rebin_path, master_mode=True)
    res = proc.fit_with_overrides(name)
    if plots_dir:
        import matplotlib.pyplot as plt
        os.makedirs(plots_dir, exist_ok=True)
        proc.create_diagnostic_plot(
            res["wave_arb"], res["flux_arb"], res["errs_arb"], res["mask_arb"],
            res["wave_ica"], res["flux_ica"], name, res["spec_name"],
            res["civ_blue"], res["civ_ew"], show=False, save=False)
        safe = "".join(c if (c.isalnum() or c in "+.-_") else "_" for c in name)
        plt.savefig(os.path.join(plots_dir, safe + ".png"), dpi=110,
                    bbox_inches="tight")
        plt.close("all")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Required on purpose: an earlier default of [1, 2, 3, 9] meant running this
    # with no arguments silently refitted those four objects and overwrote the
    # results file, and that stale list is how NGC 3516 ended up in the first
    # batch in place of NGC 3227.
    ap.add_argument("--orders", type=int, nargs="+", required=True,
                    help="work_queue order numbers to run")
    ap.add_argument("--out", default=os.path.join(OUTDIR, "coadd_experiment.csv"))
    args = ap.parse_args()

    queue = {int(r["order"]): r for r in csv.DictReader(open(QUEUE))}
    rows = []
    for o in args.orders:
        r = queue[o]
        stems = [s for s in r["stems"].split(";") if s]
        print("\n=== order %d  %s  (z=%s, %d spectra) ==="
              % (o, r["name_pub"], str(r["z"])[:8], len(stems)), flush=True)

        for st in stems:
            try:
                res = fit(REBIN, st, plots_dir=PLOTS)
                rows.append(dict(order=o, object=r["name_pub"], variant=st,
                                 kind="individual", civ_blue=res["civ_blue"],
                                 civ_ew=res["civ_ew"], f2500=res["f2500"],
                                 z=res["z"], status="ok"))
                print("   %-30s blue=%9.1f  ew=%8.2f  f2500=%.4f"
                      % (st, res["civ_blue"], res["civ_ew"], res["f2500"]), flush=True)
            except Exception as exc:
                rows.append(dict(order=o, object=r["name_pub"], variant=st,
                                 kind="individual", status="FAIL: %s" % type(exc).__name__))
                print("   %-30s FAIL %s: %s" % (st, type(exc).__name__, exc), flush=True)

        if len(stems) < 2:
            continue
        cname = "%s_COMBINED" % r["name_mast_key"]
        try:
            wave, flux, errs, mask, z, nspec = coadd(stems)
            write_fits(os.path.join(WORKDIR, cname + ".fits"), wave, flux, errs, mask, z)
            overlap = int((nspec > 1).sum())
            print("   [combined %d px, %d with >1 spectrum contributing; "
                  "scaled to %s: %s]"
                  % (wave.size, overlap, coadd.last_ref, coadd.last_scales), flush=True)
            res = fit(WORKDIR, cname, plots_dir=PLOTS)
            rows.append(dict(order=o, object=r["name_pub"], variant=cname,
                             kind="combined", civ_blue=res["civ_blue"],
                             civ_ew=res["civ_ew"], f2500=res["f2500"],
                             z=res["z"], n_overlap_px=overlap,
                             scales=str(coadd.last_scales), status="ok"))
            print("   %-30s blue=%9.1f  ew=%8.2f  f2500=%.4f"
                  % (cname, res["civ_blue"], res["civ_ew"], res["f2500"]), flush=True)
        except Exception as exc:
            rows.append(dict(order=o, object=r["name_pub"], variant=cname,
                             kind="combined", status="FAIL: %s" % type(exc).__name__))
            print("   %-30s FAIL %s: %s" % (cname, type(exc).__name__, exc), flush=True)

    cols = ["order", "object", "variant", "kind", "civ_blue", "civ_ew", "f2500",
            "z", "n_overlap_px", "scales", "status"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\n-> %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
