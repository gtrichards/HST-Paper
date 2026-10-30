"""
Extract the 1450 A continuum flux for every adopted object (manmask 2 or 3).

Output: Data/HST_F1450_v23.csv with columns
    row_num   -- row in CIV_measurements_v23.csv (the v22 spine)
    F1450     -- observed-frame f_lambda at rest-frame 1450 A, erg/s/cm2/A
    F1450_est -- 1 where F1450 is ESTIMATED rather than measured, 0 otherwise.
                 Plot estimated points with an open symbol.

Where the number comes from
---------------------------
Each adopted fit has one BEST_*.json record.  Its "monochromatic" block holds,
for the 1445-1465 A window:
    f_lambda      median of the ICA reconstruction in the window (fit units)
    continuum_scale  the co-adder's "Continuum Normalisation" in the window,
                  i.e. the archive flux in erg/s/cm2/A that the spectrum was
                  divided by before co-adding
    f_lambda_cgs  = f_lambda * continuum_scale    <- this is what we take
    measured      True only if the data (not just the model) cover the window

The flux is OBSERVED-frame f_lambda (archive units); only the wavelength is
rest-frame.  The (1+z) needed to make a luminosity is applied in make_fig1.py,
exactly as in Trevor's original make_fig1.py.  We deliberately do NOT use the
"logL" stored in the record.

How objects are matched to row_num
----------------------------------
The record folders are named by an index (e.g. 092_...), which is NOT row_num.
The join is the same one finalrun/build_paper_table.py uses:
    decisions.csv folder  ->  work_queue name_mast_key  ->  row_num
and the record is found by the 3-digit index prefix of the decisions folder.
As a check, the record's redshift must equal the z in CIV_measurements_v23.csv.

Four objects cannot be measured and are ESTIMATED
-------------------------------------------------
The ICA model gives the SHAPE at 1450 A for every object (f_lambda).  What can
be missing is the arbitrary-to-physical conversion, the co-adder's Continuum
Normalisation, which is why the fit alone cannot supply the answer: the model
is fitted to continuum-NORMALISED flux.  For four adopted objects that
conversion is unavailable at 1450 A:

    #51  Mrk 486              normalisation starts at 1508.8 A, 59 A short
    #169 Mrk 813              starts at 1467.8 A, 18 A short
    #356 QSO0020+287          starts at 1467.1 A, 17 A short
    #220 2MASS J10215235+4645158   1450 A is reached but the normalisation is
                             NEGATIVE there (-3.7e-17); the FIT is fine

For these the normalisation is extrapolated to 1450 A along the object's OWN
power law, fitted to its Continuum Normalisation over 1200-3000 A wherever that
is positive, and multiplied by the model's f_lambda.  The object's own slope is
used rather than the sample median (-1.064) because the data support it: Mrk
486's own slope is +0.59, RISING to the red, which its own L1700 < L2500
confirms, so the sample median would push it the wrong way.

Cross-checked against two other estimators -- the sample-median slope, and
scaling the ICA model to the observed flux across the overlap.  All three agree
to within 0.08 dex for #51, #169 and #356.  For #220 the own-slope and
ICA-scaled values agree to 0.03 dex while the sample-median one sits 0.26 dex
low, which is why the median is not used.  See
finalrun/sweeps/plot_l1450_estimates.py and
Figures/L1450_estimates_4objects.pdf.

This script needs the fit records, which live only on GTR's machine; the CSV it
writes is committed so that make_fig1.py runs anywhere.
"""

import csv
import glob
import json
import os

import numpy as np
from astropy.io import fits

# ---- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Data")
FINALRUN = "/Users/gtr/Work/projects/hstica/finalrun"      # GTR's working tree
FITS = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "fits")
QUEUE = os.path.join(FINALRUN, "pipeline_output", "work_queue.csv")
DECISIONS = os.path.join(DATA, "decisions.csv")
TABLE = os.path.join(DATA, "CIV_measurements_v23.csv")
OUT = os.path.join(DATA, "HST_F1450_v23.csv")
REBIN = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "rebin")


def estimate_f1450(rec):
    """Observed f_lambda at 1450 A when the Continuum Normalisation does not
    reach it: the object's own normalisation power law, times the model shape.

    Returns (value, slope, fit_range) or (None, ...) if it cannot be done.
    """
    path = os.path.join(REBIN, rec["name"] + ".fits")
    if not os.path.exists(path):
        return None, float("nan"), None
    d = fits.open(path)[1].data
    w = np.asarray(d["Rest-frame Wavelength"], float)
    cn = np.asarray(d["Continuum Normalisation"], float)
    k = np.isfinite(cn) & (cn > 0) & (w >= 1200) & (w <= 3000)
    if k.sum() < 50:
        return None, float("nan"), None
    slope, inter = np.polyfit(np.log10(w[k]), np.log10(cn[k]), 1)
    cont = 10.0 ** (inter + slope * np.log10(1450.0))
    f_shape = rec["monochromatic"]["1450"]["f_lambda"]
    if not np.isfinite(f_shape) or not np.isfinite(cont):
        return None, float("nan"), None
    return float(f_shape * cont), float(slope), (float(w[k].min()), float(w[k].max()))


def folder_key(name):
    """Folder names as object_paths.folder_for writes them (spaces etc. -> _)."""
    import sys
    sys.path.insert(0, FINALRUN)
    from object_paths import folder_for
    return folder_for(name)


# ---- 1. map each decisions folder to a row_num via the work queue ----------
# A queue entry can appear in the ledger as NAME_COMBINED, NAME_SPLICE, or as
# one of its single-instrument stems (NAME_COS, NAME_FOS, ...).
folder_to_row = {}
for q in csv.DictReader(open(QUEUE)):
    names = [q["name_mast_key"] + "_COMBINED", q["name_mast_key"] + "_SPLICE"]
    names += [s for s in q["stems"].split(";") if s]
    for n in names:
        folder_to_row[folder_key(n)] = int(q["row_num"])

# ---- 2. find the BEST record for each fit index ----------------------------
best_by_index = {}
for path in glob.glob(os.path.join(FITS, "*", "records", "BEST_*.json")):
    index = int(os.path.basename(os.path.dirname(os.path.dirname(path)))[:3])
    best_by_index[index] = path

# ---- 3. read z from the paper table, for the check -------------------------
z_table = {int(r["row_num"]): float(r["z"]) for r in csv.DictReader(open(TABLE))}

# ---- 4. loop over adopted objects ------------------------------------------
rows = []
estimated = []
for dec in csv.DictReader(open(DECISIONS)):
    if int(float(dec["manmask"])) not in (2, 3):
        continue                                   # not in the adopted sample
    row_num = folder_to_row[dec["folder"]]
    rec = json.load(open(best_by_index[int(dec["folder"][:3])]))

    # check: same object as the paper table (redshift must agree)
    assert abs(rec["result"]["z"] - z_table[row_num]) < 1e-6, dec["folder"]

    m1450 = rec["monochromatic"]["1450"]
    if m1450["measured"]:
        rows.append({"row_num": row_num, "F1450": m1450["f_lambda_cgs"],
                     "F1450_est": 0})
    else:
        est, slope, rng = estimate_f1450(rec)
        if est is None:
            rows.append({"row_num": row_num, "F1450": "", "F1450_est": ""})
        else:
            rows.append({"row_num": row_num, "F1450": est, "F1450_est": 1})
            estimated.append((row_num, dec["folder"], est, slope, rng))

rows.sort(key=lambda r: r["row_num"])
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["row_num", "F1450", "F1450_est"])
    w.writeheader()
    w.writerows(rows)

n_meas = sum(1 for r in rows if r["F1450"] != "" and r["F1450_est"] == 0)
n_est = sum(1 for r in rows if r["F1450_est"] == 1)
n_none = sum(1 for r in rows if r["F1450"] == "")
print("wrote %s: %d adopted objects, %d measured, %d estimated, %d neither"
      % (OUT, len(rows), n_meas, n_est, n_none))
for row_num, folder, est, slope, rng in estimated:
    print("   ESTIMATED  row %-4s %-38s F1450=%.4g  own slope %+.2f over %.0f-%.0f A"
          % (row_num, folder, est, slope, rng[0], rng[1]))
