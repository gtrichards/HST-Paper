"""Append the morph-coefficient luminosity correction to the affected ledger notes.

Second of two luminosity corrections to the same prose.  The first
(sweeps/append_logL_correction.py, 2026-10-30) appended the missing log10(1+z);
this one appends the morph coefficient, found on 2026-11-03.  The paragraph says
so explicitly, because a reader hitting two correction paragraphs on one row
needs to know the later one supersedes the values quoted in the earlier.

Same method as the first, for the same reason GTR gave then: APPEND rather than
rewrite.  The notes quote luminosities as literal numbers inside scientific
prose, and rewriting them by pattern match would risk altering differences,
comparisons and values belonging to other objects.  Each affected row gets one
paragraph carrying its own per-window correction and the corrected values, so no
arithmetic is left to the reader.

The correction is per window, not per object: the morph coefficient is the
reference quasar continuum sampled at the observed wavelength, so it falls from
about 3.9 at 1150-1300 A to about 1.4 at 2500-3000 A.  Each row therefore quotes
its own dex at 1450, 1700 and 2500 A, read from morph_fix_records_log.csv.

    python sweeps/append_logL_morph_correction.py            # dry run
    python sweeps/append_logL_morph_correction.py --apply
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINALRUN = os.path.dirname(HERE)
FITS_DIR = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "fits")
LOG = os.path.join(FINALRUN, "pipeline_output", "final", "morph_fix_records_log.csv")
LEDGER = "/Users/gtr/Work/git/HST-Paper/Data/decisions.csv"
MARK = "*** LUMINOSITY CORRECTION, 2026-11-03 (MORPH COEFFICIENT)"
QUOTES = re.compile(r"(?:log\s*)?L\s*_?(?:1450|1700|2500)\s*(?:=|of|is|:)?\s*\d{2}\.\d{1,4}", re.I)

#: Objects whose adopted product could not be regenerated, and whose luminosity
#: is therefore NOT corrected.  Empty as of 2026-11-03.
#:
#: Index 237 was the only entry.  fix_luminosity_morph.py refuses to write a
#: product whose flux, errors or mask do not come back bit-identical, and 237's
#: did not: 353 pixels changed their finite/NaN pattern, the co-add running to
#: 1979 A rest where the stored product stopped at 1578.  Chasing the cause
#: through the history found nothing -- the co-add is deterministic, the staged
#: exposures predate the product, there is no exposure override and no BAL
#: trough, and rebuilding at every commit that touched rebinning/ reproduced the
#: stored product every time.  What resolved it was GTR rebuilding the object
#: with fit_chosen_visit.py --index 237 and confirming the fit in the
#: interactive fitter: the rebuilt product reproduces exactly across repeat
#: runs, so the exact factor applied and no estimate was needed.  The likeliest
#: explanation is that that run re-staged the exposures and the earlier staging
#: was what differed, not the code.
#:
#: Cross-check that the arithmetic rescaling is sound: iter07_chosen_visit,
#: computed from scratch against the corrected product rather than rescaled,
#: gives log L1450 = 44.430 where the rescaled BEST gives 44.411 -- 0.019 dex
#: apart, the residual being GTR's mask against the automatic one.
UNCORRECTED = set()

CAUSE = (
    "CAUSE: spec_morph.morph2 MULTIPLIES the flux by its morph coefficient "
    "-- cont_morph / cont_spec, the reference quasar continuum over the continuum "
    "of the already-normalised spectrum, so about 3 -- and rebinning/coadd.py "
    "never divided that coefficient back out of the Continuum Normalisation "
    "column it writes alongside. The errors WERE scaled by it (coadd.py "
    "multiplies them by varweighted_morph_factor on the line above), so the flux "
    "being morphed was understood; the continuum simply did not get the matching "
    "division. iterate_fit.py then formed f_real = median(flux_ica) x "
    "median(Continuum Normalisation), which over-counts the continuum once, and "
    "every luminosity built on it was high by the morph coefficient. The morph "
    "itself is correct and deliberate: it is Rankine's prescription and it is "
    "what the ICA components are fitted against. Only the recovery of the "
    "absolute scale was wrong. The coefficient falls with wavelength (about 3.9 "
    "at 1150-1300 A to about 1.4 at 2500-3000 A), so the error was larger at "
    "1450 than at 2500 and was a tilt rather than an offset. VERIFIED against "
    "two independent archives: NGC 3783's raw FOS exposures give f_lambda(1450) "
    "= 6.86e-14 where the record read 2.05e-13 and now reads 6.10e-14; 3C 120's "
    "native IUE MXLO co-add gives 1.04e-14 where the product read 2.91e-14 and "
    "now reads 9.87e-15. Against SDSS, whose flux never passes through this "
    "pipeline, the HST/SDSS ratio inside the fitted range went from 1.84 to 1.10 "
    "and the scatter from 0.322 to 0.205 dex. GTR said repeatedly that the HST "
    "luminosities looked too high against SDSS; they were. The project's own "
    "earlier check reported agreement because it averaged over 3000-7200 A rest, "
    "outside the range the ICA components and the morph reference cover. "
    "THE BLUESHIFT AND EW ON THIS ROW ARE UNAFFECTED -- they are a position and a "
    "ratio measured on the morphed spectrum, independent of the absolute scale, "
    "and the flux column is bit-identical before and after, which "
    "sweeps/fix_luminosity_morph.py asserts before writing any product. "
    "See sweeps/fix_luminosity_morph.py and coadd._descale.")


def main():
    apply = "--apply" in sys.argv
    lg = pd.read_csv(LOG)
    lg = lg[lg.is_best == 1]
    by_index = {}
    for _, r in lg.iterrows():
        m = re.search(r"fits/(\d+)_", str(r["record"]))
        if m:
            by_index[int(m.group(1))] = r

    d = pd.read_csv(LEDGER)
    d["index"] = d["folder"].map(lambda f: int(str(f).split("_")[0]))
    done = skipped = already = nouncorr = 0
    for i, r in d.iterrows():
        note = str(r["note"])
        if MARK in note:
            already += 1
            continue
        if not QUOTES.search(note):
            skipped += 1
            continue
        idx = int(r["index"])

        if idx in UNCORRECTED:
            para = (" %s. This row is an EXCEPTION: its luminosities are NOT "
                    "corrected and every monochromatic luminosity quoted ABOVE is "
                    "STILL HIGH by the morph coefficient, roughly 0.5 dex at 1450 A. "
                    "The correction could not be applied because the adopted "
                    "product is not reproducible from the current code and staged "
                    "exposures -- on regeneration 353 pixels change their "
                    "finite/NaN pattern and the flux differs by a median 14 per "
                    "cent -- and sweeps/fix_luminosity_morph.py refuses to write a "
                    "product whose flux is not bit-identical, rather than silently "
                    "change the data a fit was made against. That irreproducibility "
                    "is pre-existing and unrelated to this correction. DO NOT USE "
                    "THIS OBJECT'S LUMINOSITY until the product is rebuilt and "
                    "refitted. %s" % (MARK, CAUSE))
            nouncorr += 1
        else:
            g = (glob.glob(os.path.join(FITS_DIR, "%d_*" % idx, "records", "BEST_*.json")) +
                 glob.glob(os.path.join(FITS_DIR, "%03d_*" % idx, "records", "BEST_*.json")))
            row = by_index.get(idx)
            if not g or row is None:
                para = (" %s. Every monochromatic luminosity quoted ABOVE in this "
                        "note predates a correction and is HIGH by log10 of the "
                        "morph coefficient -- about 0.50 dex at 1450 A, 0.44 at "
                        "1700 A and 0.21 at 2500 A. This object has no adopted "
                        "record, so no corrected values are quoted here; subtract "
                        "the figure for the relevant window from any luminosity "
                        "above. %s" % (MARK, CAUSE))
            else:
                mono = json.load(open(g[0]))["monochromatic"]
                vals, dexes = [], []
                for k in ("1450", "1700", "2500"):
                    v = mono.get(k) or {}
                    dv = row.get("dlogL_%s" % k)
                    if (v.get("measured") and np.isfinite(v.get("logL", np.nan))
                            and dv is not None and np.isfinite(dv)):
                        vals.append("log L%s = %.3f (was %.3f)"
                                    % (k, v["logL"], v["logL"] - float(dv)))
                        dexes.append("%s A %+.3f" % (k, float(dv)))
                para = (" %s. Every monochromatic luminosity quoted ABOVE in this "
                        "note is HIGH by log10 of the morph coefficient and is "
                        "SUPERSEDED by the values here -- including any value given "
                        "in an earlier LUMINOSITY CORRECTION paragraph on this row, "
                        "which corrected a different error ((1+z), 2026-10-30) and "
                        "did not touch this one. This object's correction, by "
                        "window: %s. CORRECTED: %s. %s"
                        % (MARK, "; ".join(dexes) if dexes else "none applicable",
                           "; ".join(vals) if vals
                           else "no luminosity is measured for this object", CAUSE))
        d.at[i, "note"] = note + para
        done += 1

    print("%s" % ("DRY RUN (pass --apply to write)" if not apply else "WRITTEN"))
    print("  rows quoting a luminosity, corrected : %d" % done)
    print("    of which flagged UNCORRECTED       : %d" % nouncorr)
    print("  rows already carrying the correction : %d" % already)
    print("  rows with no luminosity quoted       : %d" % skipped)
    if apply:
        d.drop(columns=["index"]).to_csv(LEDGER, index=False, quoting=csv.QUOTE_MINIMAL)
        print("  -> %s" % LEDGER)


if __name__ == "__main__":
    main()
