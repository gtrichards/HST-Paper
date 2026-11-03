"""Divide the morph coefficient out of the Continuum Normalisation column.

WHY.  The column is documented -- and consumed everywhere, including
check_l1450_anchor.py, which calls `cn x arb` "the de-normalised flux" -- as the
number that turns the stored arbitrary flux back into erg/s/cm2/A:

    Coadded Flux (Arbitrary Units) x Continuum Normalisation = f_lambda

It was not that number.  spec_morph.morph2 MULTIPLIES the flux by its morph
coefficient (the reference quasar continuum over the continuum of the
already-normalised spectrum, so about 3) and nothing divided that coefficient
back out of the continuum, so `cn x arb` over-counted the continuum once.  Every
luminosity built on it was high by the morph coefficient: a median 0.495 dex over
the 399 chosen-visit products with 1450 A coverage, redshift-dependent because
the coefficient is the reference continuum sampled at the observed wavelength
(0.497 dex at z < 0.5 against 0.432 dex at z = 1-1.5), so a tilt in L-z rather
than a constant offset.  GTR said repeatedly that the HST luminosities looked too
high against SDSS; they were.

coadd.py now divides it out at all three final_cont sites (see _descale there).
The coefficient is not recoverable from a written product -- it would need the
normalised flux, which is the quantity it was multiplied into -- so each product
has to be rebinned again to recover it.

WHAT THIS DOES NOT CHANGE.  The morph is Rankine's prescription and is what the
ICA components are fitted against, so the flux is untouched: this script REFUSES
to write unless the regenerated wavelength, flux, error and mask columns come
back bit-identical to the ones already on disk, and it then replaces ONLY the
Continuum Normalisation column.  Every adopted fit, blueshift and equivalent
width therefore stands; only the absolute scale moves.

Redshift is taken from each product's own Redshift column rather than from the
ledger, so the regeneration reproduces exactly what is on disk and the
bit-identical assertion is a real check.  (Guessing NGC 3783's z as 0.00973 when
the product was built with 0.00950 made every column differ, which is what that
assertion is there to catch.)

By default the corrected products are written to a SEPARATE tree beside each
original -- rebin_morphfix/ next to rebin/ -- and nothing on disk is overwritten.
That is the same shape as the MASTER/V23 pair rebin_path.py already describes:
the corrected tree can be fitted against with

    export HSTICA_REBIN=.../pipeline_output/chosen_visit/rebin_morphfix

and compared against the original before anything is promoted.  Promotion is a
deliberate, separate step (--in-place), not a side effect of running this.

    python sweeps/fix_luminosity_morph.py --dry-run      # report, write nothing
    python sweeps/fix_luminosity_morph.py                # write corrected tree
    python sweeps/fix_luminosity_morph.py --in-place     # overwrite originals
"""
import argparse, csv, glob, io, contextlib, os, sys
import numpy as np
from astropy.table import Table

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from rebinning import coadd as CO

STAGE = os.path.join(HERE, "pipeline_output", "chosen_visit", "stage")
REBIN = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
IUE_STAGE = os.path.join(HERE, "iue", "chosen_run_work")
IUE_REBIN = os.path.join(HERE, "iue", "rebin")
WORK = os.path.join(HERE, "pipeline_output", "chosen_visit", "morph_fix_work")

FLUXCOLS = ["Rest-frame Wavelength", "Coadded Flux (Arbitrary Units)",
            "Coadded Flux Errors", "Bad Pixel Mask"]


def ident(a, b, col):
    x, y = np.asarray(a[col], float), np.asarray(b[col], float)
    if len(x) != len(y):
        return False
    gx, gy = np.isfinite(x), np.isfinite(y)
    return np.array_equal(gx, gy) and np.array_equal(x[gx], y[gy])


def jobs():
    """(product path, Identifier, instrument, staged exposure dir) for every product."""
    out = []
    queue = {int(r["order"]): r for r in csv.DictReader(
        open(os.path.join(HERE, "pipeline_output", "work_queue.csv")))}
    for i, r in sorted(queue.items()):
        nm = (r.get("name_mast_key") or "").strip()
        if not nm:
            continue
        for inst in ("FOS", "STIS", "COS"):
            d = os.path.join(STAGE, "%03d_%s" % (i, inst))
            p = os.path.join(REBIN, "%s_%s.fits" % (nm, inst))
            if os.path.isdir(d) and os.path.exists(p):
                out.append((p, nm, inst, d))
    for p in sorted(glob.glob(os.path.join(IUE_REBIN, "*_IUE.fits"))):
        nm = os.path.basename(p)[:-len("_IUE.fits")]
        d = os.path.join(IUE_STAGE, nm.replace(" ", "_"))
        if os.path.isdir(d):
            out.append((p, nm, "IUE", d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the original products instead of writing "
                         "a rebin_morphfix/ tree beside them")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    todo = jobs()
    if a.limit:
        todo = todo[:a.limit]
    print("products to regenerate: %d" % len(todo), flush=True)

    rows, fixed, refused, failed = [], 0, 0, 0
    for p, nm, inst, d in todo:
        old = Table.read(p)
        z = float(np.nanmedian(np.asarray(old["Redshift"], float)))
        if not np.isfinite(z):
            print("  %-34s REFUSED: no redshift in product" % nm[:34], flush=True)
            refused += 1
            continue
        for stale in glob.glob(os.path.join(WORK, "*.fits")):
            os.remove(stale)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                CO.rebin(nm, z, inst, None, data_path=d, output_dir=WORK, flat=True)
        except Exception as exc:
            print("  %-34s REBIN FAILED: %s: %s"
                  % (nm[:34], type(exc).__name__, str(exc)[:70]), flush=True)
            failed += 1
            continue
        got = glob.glob(os.path.join(WORK, "*.fits"))
        if not got:
            print("  %-34s REBIN produced no file" % nm[:34], flush=True)
            failed += 1
            continue
        new = Table.read(got[0])

        bad = [c for c in FLUXCOLS if not ident(new, old, c)]
        if bad:
            print("  %-34s REFUSED: %s differ -- not written"
                  % (nm[:34], ", ".join(bad)), flush=True)
            refused += 1
            rows.append(dict(product=os.path.basename(p), name=nm, inst=inst, z=z,
                             status="refused", note="columns differ: %s" % "|".join(bad)))
            continue

        cn_old = np.asarray(old["Continuum Normalisation"], float)
        cn_new = np.asarray(new["Continuum Normalisation"], float)
        w = np.asarray(old["Rest-frame Wavelength"], float)
        m = (w > 1445) & (w < 1465) & np.isfinite(cn_old) & np.isfinite(cn_new)
        fac = float(np.nanmedian(cn_new[m] / cn_old[m])) if m.sum() > 3 else np.nan
        dex = float(np.log10(fac)) if np.isfinite(fac) and fac > 0 else np.nan

        if not a.dry_run:
            old["Continuum Normalisation"] = cn_new
            if a.in_place:
                dest = p
            else:
                dd = os.path.join(os.path.dirname(os.path.dirname(p)),
                                  os.path.basename(os.path.dirname(p)) + "_morphfix")
                os.makedirs(dd, exist_ok=True)
                dest = os.path.join(dd, os.path.basename(p))
            old.write(dest, overwrite=True)
        fixed += 1
        rows.append(dict(product=os.path.basename(p), name=nm, inst=inst, z=z,
                         status="fixed", cn_factor_1450=fac, dlogL_1450=dex,
                         note="cn divided by morph coefficient"))
        if fixed % 50 == 0:
            print("  ... %d done" % fixed, flush=True)

    log = os.path.join(HERE, "pipeline_output", "final", "morph_fix_log.csv")
    if rows:
        keys = ["product", "name", "inst", "z", "status", "cn_factor_1450",
                "dlogL_1450", "note"]
        with open(log, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            for r in rows:
                wr.writerow({k: r.get(k, "") for k in keys})
    d = np.array([r["dlogL_1450"] for r in rows
                  if r.get("status") == "fixed" and np.isfinite(r.get("dlogL_1450", np.nan))])
    print("\n%s" % ("DRY RUN -- nothing written" if a.dry_run
                     else ("originals OVERWRITTEN" if a.in_place
                           else "corrected tree written beside each original "
                                "(rebin_morphfix/); originals untouched")))
    print("  fixed %d   refused %d   failed %d" % (fixed, refused, failed))
    if len(d):
        print("  logL(1450) correction, dex: median %+.3f   16-84%% %+.3f to %+.3f   n=%d"
              % (np.median(d), np.percentile(d, 16), np.percentile(d, 84), len(d)))
    print("  log: %s" % log)


if __name__ == "__main__":
    main()
