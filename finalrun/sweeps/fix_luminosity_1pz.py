"""Apply the missing (1+z) to every stored monochromatic luminosity.

rebinning/coadd.py:526 writes the rebinned table as

    Table([final_wave/(1+z), final_flux, ...])

-- only the WAVELENGTH is divided.  The flux column, and the Continuum
Normalisation that rescales it to erg/s/cm2/A, are written untouched, so the
calibrated flux is OBSERVED-frame flux density on a rest-frame wavelength axis.
For an observed f_lambda the monochromatic luminosity is

    nu_rest L_nu(rest) = 4 pi D_L^2 lambda_obs f_lambda(obs)
                       = 4 pi D_L^2 lambda_rest (1+z) f_lambda(obs)

iterate_fit.py and ica/manual_fix_gui.py both omitted the (1+z) until
2026-10-30, so every logL they wrote is low by log10(1+z): median 0.116 dex
over the adopted sample, 0.347 dex at the highest redshift, and monotonic in z.

Nothing is refitted.  Each record already stores the redshift, so the fix is
logL -> logL + log10(1+z), applied once and recorded in the record itself so a
second run cannot double-apply it.  Blueshifts and equivalent widths are
untouched: they are positions and ratios, independent of the flux calibration.
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FITS_DIR = os.path.join(os.path.dirname(HERE), "pipeline_output", "chosen_visit", "fits")
STAMP = "logL_1pz_applied"

def main():
    dry = "--apply" not in sys.argv
    recs = sorted(glob.glob(os.path.join(FITS_DIR, "*", "records", "*.json")))
    touched = skipped = already = 0
    shifts = []
    for p in recs:
        r = json.load(open(p))
        mono = r.get("monochromatic")
        if not isinstance(mono, dict):
            skipped += 1; continue
        if r.get(STAMP):
            already += 1; continue
        z = (r.get("result") or {}).get("z")
        if z is None or not np.isfinite(z):
            skipped += 1; continue
        d = np.log10(1.0 + float(z))
        changed = False
        for k, v in mono.items():
            if k == "_calibration" or not isinstance(v, dict):
                continue
            L = v.get("logL")
            if L is not None and np.isfinite(L):
                v["logL"] = float(L) + d
                changed = True
                shifts.append(d)
        if changed:
            r[STAMP] = True
            r["logL_1pz_note"] = ("logL corrected by +log10(1+z) = %+.4f on "
                                  "2026-10-30; see sweeps/fix_luminosity_1pz.py" % d)
            touched += 1
            if not dry:
                json.dump(r, open(p, "w"), indent=1, default=str)
    print("%s" % ("DRY RUN (pass --apply to write)" if dry else "WRITTEN"))
    print("  records scanned      : %d" % len(recs))
    print("  records with a logL  : %d" % touched)
    print("  already corrected    : %d" % already)
    print("  no logL / no z       : %d" % skipped)
    if shifts:
        print("  individual logL values shifted: %d" % len(shifts))
        print("  shift: median +%.3f dex, max +%.3f dex" % (np.median(shifts), max(shifts)))

if __name__ == "__main__":
    main()
