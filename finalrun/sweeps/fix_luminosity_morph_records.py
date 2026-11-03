"""Apply the morph-coefficient correction to the luminosities in the fit records.

Companion to fix_luminosity_morph.py, which corrects the Continuum Normalisation
column in the rebinned products.  This propagates that correction into the
records the ledger and the paper table quote, WITHOUT refitting anything.

It is safe not to refit because the defect is purely in the recovery of the
absolute scale.  iterate_fit.monochromatic computes

    f_real = f0 * scale          f0    = median(flux_ica) in the window, fit units
                                 scale = median(Continuum Normalisation) there
    logL   = log10(4 pi D_L^2 * f_real * w0 * (1+z))

`f0` comes from the fit and does not change -- the flux column is bit-identical
before and after, which fix_luminosity_morph.py asserts before it writes
anything.  Only `scale` changes, so the whole correction is one multiplication
per window:

    f_lambda_cgs *= scale_new / scale_old
    logL         += log10(scale_new / scale_old)

Doing it this way rather than re-deriving the formula keeps the cosmology, the
window edges and the (1+z) factor exactly as the adopted records already have
them -- in particular it cannot re-apply or drop the (1+z) that
fix_luminosity_1pz.py put in on 2026-10-30, which re-deriving would risk.

The ratio is taken per window, not once per object: the morph coefficient is the
reference continuum sampled at the observed wavelength, so it differs between
1450, 1700 and 2500 A.

Every record carrying a monochromatic block is corrected, not only BEST_, so that
promoting a different iteration later cannot resurrect an uncorrected luminosity.
Records already marked logL_morph_applied are skipped, so this is idempotent.

    python sweeps/fix_luminosity_morph_records.py --dry-run
    python sweeps/fix_luminosity_morph_records.py
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from astropy.table import Table

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = {"1450": (1445.0, 1465.0), "1700": (1700.0, 1705.0),
           "2500": (2490.0, 2510.0)}
TREES = [(os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin"),
          os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin_morphfix"),
          os.path.join(HERE, "pipeline_output", "chosen_visit", "fits")),
         (os.path.join(HERE, "iue", "rebin"),
          os.path.join(HERE, "iue", "rebin_morphfix"),
          os.path.join(HERE, "iue", "fits"))]

NOTE = ("logL corrected by log10(cn_new/cn_old) on 2026-11-03: spec_morph.morph2 "
        "multiplied the flux by its morph coefficient and the Continuum "
        "Normalisation column never divided it out, so f_lambda_cgs over-counted "
        "the continuum once; see sweeps/fix_luminosity_morph.py")


def factors(old_p, new_p):
    """Per-window scale_new/scale_old, plus the ratio at the calibration anchors."""
    a, b = Table.read(old_p), Table.read(new_p)
    w = np.asarray(a["Rest-frame Wavelength"], float)
    co = np.asarray(a["Continuum Normalisation"], float)
    cn = np.asarray(b["Continuum Normalisation"], float)
    out = {}
    for k, (lo, hi) in WINDOWS.items():
        m = (w >= lo) & (w <= hi) & np.isfinite(co) & np.isfinite(cn)
        # iterate_fit requires m.sum() > 3 before it sets a scale at all
        if m.sum() > 3:
            so, sn = np.nanmedian(co[m]), np.nanmedian(cn[m])
            out[k] = float(sn / so) if (np.isfinite(so) and so != 0) else np.nan
        else:
            out[k] = np.nan
    g = np.isfinite(w) & np.isfinite(co) & (co != 0) & np.isfinite(cn)
    anch = {}
    if g.any():
        for tag, j in (("min", int(np.argmin(w[g]))), ("max", int(np.argmax(w[g])))):
            anch[tag] = float(cn[g][j] / co[g][j])
    return out, anch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows, nrec, nfix, nskip, nmiss = [], 0, 0, 0, 0
    missing = set()
    cache = {}
    for old_t, new_t, fitdir in TREES:
        for rec in sorted(glob.glob(os.path.join(fitdir, "*", "records", "*.json"))):
            nrec += 1
            try:
                d = json.load(open(rec))
            except Exception:
                continue
            mono = d.get("monochromatic")
            if not isinstance(mono, dict):
                continue
            if d.get("logL_morph_applied"):
                nskip += 1
                continue
            nm = d.get("name")
            if not isinstance(nm, str) or not nm.strip():
                nmiss += 1
                continue
            key = (old_t, nm)
            if key not in cache:
                op = os.path.join(old_t, "%s.fits" % nm)
                npth = os.path.join(new_t, "%s.fits" % nm)
                cache[key] = (factors(op, npth)
                              if (os.path.exists(op) and os.path.exists(npth)) else None)
            f = cache[key]
            if f is None:
                nmiss += 1
                missing.add(nm)
                continue
            fac, anch = f

            changed = []
            for k in WINDOWS:
                blk = mono.get(k)
                if not isinstance(blk, dict):
                    continue
                r = fac.get(k, np.nan)
                if not (np.isfinite(r) and r > 0):
                    continue
                if (isinstance(blk.get("f_lambda_cgs"), (int, float))
                        and np.isfinite(blk["f_lambda_cgs"])):
                    blk["f_lambda_cgs"] = float(blk["f_lambda_cgs"]) * r
                if (isinstance(blk.get("logL"), (int, float))
                        and np.isfinite(blk["logL"])):
                    blk["logL"] = float(blk["logL"]) + float(np.log10(r))
                    changed.append((k, float(np.log10(r))))
            cal = mono.get("_calibration")
            if isinstance(cal, dict):
                for tag in ("min", "max"):
                    kk = "cont_at_%s" % tag
                    if (tag in anch and isinstance(cal.get(kk), (int, float))
                            and np.isfinite(cal[kk])):
                        cal[kk] = float(cal[kk]) * anch[tag]
            d["logL_morph_applied"] = True
            d["logL_morph_note"] = NOTE
            if not a.dry_run:
                with open(rec, "w") as fh:
                    json.dump(d, fh, indent=1)
            nfix += 1
            if changed:
                rows.append(dict(record=os.path.relpath(rec, HERE), name=nm,
                                 is_best=int(os.path.basename(rec).startswith("BEST_")),
                                 **{"dlogL_%s" % k: "%.4f" % v for k, v in changed}))

    log = os.path.join(HERE, "pipeline_output", "final", "morph_fix_records_log.csv")
    if rows and not a.dry_run:
        keys = ["record", "name", "is_best", "dlogL_1450", "dlogL_1700", "dlogL_2500"]
        with open(log, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            for r in rows:
                wr.writerow({k: r.get(k, "") for k in keys})
    print("%s" % ("DRY RUN -- nothing written" if a.dry_run else "written"))
    print("  records seen %d   corrected %d   already done %d   no product %d"
          % (nrec, nfix, nskip, nmiss))
    if missing:
        print("  products missing for: %s" % ", ".join(sorted(missing)[:8]))
    d1 = np.array([float(r["dlogL_1450"]) for r in rows if r.get("dlogL_1450")])
    b1 = np.array([float(r["dlogL_1450"]) for r in rows
                   if r.get("dlogL_1450") and r["is_best"]])
    for tag, arr in (("all records", d1), ("BEST only", b1)):
        if len(arr):
            print("  dlogL(1450) %s: median %+.3f  16-84%% %+.3f to %+.3f  n=%d"
                  % (tag, np.median(arr), np.percentile(arr, 16),
                     np.percentile(arr, 84), len(arr)))
    if not a.dry_run:
        print("  log: %s" % log)


if __name__ == "__main__":
    main()
