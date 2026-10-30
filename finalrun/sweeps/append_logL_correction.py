"""Append the (1+z) luminosity correction to the affected ledger notes.

The monochromatic luminosities in the BEST records were corrected on
2026-10-30 (sweeps/fix_luminosity_1pz.py).  The justification prose in
Data/decisions.csv quotes luminosities as literal numbers -- 318 rows, 697
values -- and regenerating the records does not touch that text.

GTR chose to APPEND a correction rather than rewrite the numbers in place.
That matches how this ledger already works: several notes open with "This note
is an append-only history and OPENS WITH A SUPERSEDED VERDICT", and rewriting
697 numbers inside scientific prose by pattern match would risk altering
differences, comparisons and values belonging to other objects.

Each affected row gets one paragraph giving the object's own correction and the
corrected values, so no arithmetic is left to the reader.
"""
import os, sys, csv, json, glob, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINALRUN = os.path.dirname(HERE)
FITS_DIR = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "fits")
LEDGER = "/Users/gtr/Work/git/HST-Paper/Data/decisions.csv"
MARK = "*** LUMINOSITY CORRECTION, 2026-10-30"
QUOTES = re.compile(r"(?:log\s*)?L\s*_?(?:1450|1700|2500)\s*(?:=|of|is|:)?\s*\d{2}\.\d{1,4}", re.I)


def main():
    apply = "--apply" in sys.argv
    d = pd.read_csv(LEDGER)
    d["index"] = d["folder"].map(lambda f: int(str(f).split("_")[0]))
    done = skipped = already = 0
    for i, r in d.iterrows():
        note = str(r["note"])
        if MARK in note:
            already += 1; continue
        if not QUOTES.search(note):
            skipped += 1; continue
        idx = int(r["index"])
        g = (glob.glob(os.path.join(FITS_DIR, "%d_*" % idx, "records", "BEST_*.json")) +
             glob.glob(os.path.join(FITS_DIR, "%03d_*" % idx, "records", "BEST_*.json")))
        if not g:
            # manmask 0/1 rows can quote a luminosity in the prose that argued
            # for exclusion; they have no adopted record, so give the rule only.
            para = (" %s. Every monochromatic luminosity quoted ABOVE in this note "
                    "predates a correction and is LOW by log10(1+z). "
                    "rebinning/coadd.py writes the rebinned table as "
                    "Table([final_wave/(1+z), final_flux, ...]) -- only the WAVELENGTH "
                    "is divided -- so the calibrated flux is OBSERVED-frame flux density "
                    "on a rest-frame wavelength axis, and nu_rest L_nu = 4 pi D_L^2 "
                    "lambda_rest (1+z) f_lambda(obs). iterate_fit.py and "
                    "ica/manual_fix_gui.py both omitted the (1+z) until 2026-10-30. "
                    "This object has no adopted record; add log10(1+z) to any "
                    "luminosity quoted above. Blueshift and EW are UNAFFECTED." % MARK)
        else:
            rec = json.load(open(g[0]))
            z = rec["result"]["z"]; mono = rec["monochromatic"]
            dex = np.log10(1.0 + z)
            vals = []
            for k in ("1450", "1700", "2500"):
                v = mono.get(k) or {}
                if v.get("measured") and np.isfinite(v.get("logL", np.nan)):
                    vals.append("log L%s = %.3f (was %.3f)" % (k, v["logL"], v["logL"] - dex))
            para = (" %s. Every monochromatic luminosity quoted ABOVE in this note is "
                    "LOW by log10(1+z) = %.4f dex and is SUPERSEDED by the values here. "
                    "CORRECTED: %s. CAUSE: rebinning/coadd.py writes the rebinned table as "
                    "Table([final_wave/(1+z), final_flux, ...]) -- only the WAVELENGTH is "
                    "divided, so the calibrated flux is OBSERVED-frame flux density on a "
                    "rest-frame wavelength axis, and nu_rest L_nu = 4 pi D_L^2 lambda_rest "
                    "(1+z) f_lambda(obs). iterate_fit.py line 397 and ica/manual_fix_gui.py "
                    "both omitted the factor until 2026-10-30, when GTR asked why Figure 1 "
                    "disagreed with the ledger; the figure was right. Across the adopted "
                    "sample the shift is a median of 0.116 dex and up to 0.347 dex, and it "
                    "is monotonic in z, so it tilts the L-z relation rather than offsetting "
                    "it. THE BLUESHIFT AND EW ON THIS ROW ARE UNAFFECTED -- they are "
                    "positions and ratios, independent of the flux calibration."
                    % (MARK, dex, "; ".join(vals) if vals else "no luminosity is measured for this object"))
        d.at[i, "note"] = note + para
        done += 1
    print("%s" % ("DRY RUN (pass --apply to write)" if not apply else "WRITTEN"))
    print("  rows quoting a luminosity, corrected : %d" % done)
    print("  rows already carrying the correction : %d" % already)
    print("  rows with no luminosity quoted       : %d" % skipped)
    if apply:
        d.drop(columns=["index"]).to_csv(LEDGER, index=False, quoting=csv.QUOTE_MINIMAL)
        print("  -> %s" % LEDGER)


if __name__ == "__main__":
    main()
