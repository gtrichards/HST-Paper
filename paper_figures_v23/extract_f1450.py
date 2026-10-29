"""
Extract the 1450 A continuum flux for every adopted object (manmask 2 or 3).

Output: Data/HST_F1450_v23.csv with columns
    row_num   -- row in CIV_measurements_v23.csv (the v22 spine)
    F1450     -- observed-frame f_lambda at rest-frame 1450 A, erg/s/cm2/A
                 (blank where the spectrum does not cover 1445-1465 A)

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

This script needs the fit records, which live only on GTR's machine; the CSV it
writes is committed so that make_fig1.py runs anywhere.
"""

import csv
import glob
import json
import os

# ---- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Data")
FINALRUN = "/Users/gtr/Work/projects/hstica/finalrun"      # GTR's working tree
FITS = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "fits")
QUEUE = os.path.join(FINALRUN, "pipeline_output", "work_queue.csv")
DECISIONS = os.path.join(DATA, "decisions.csv")
TABLE = os.path.join(DATA, "CIV_measurements_v23.csv")
OUT = os.path.join(DATA, "HST_F1450_v23.csv")


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
for dec in csv.DictReader(open(DECISIONS)):
    if int(float(dec["manmask"])) not in (2, 3):
        continue                                   # not in the adopted sample
    row_num = folder_to_row[dec["folder"]]
    rec = json.load(open(best_by_index[int(dec["folder"][:3])]))

    # check: same object as the paper table (redshift must agree)
    assert abs(rec["result"]["z"] - z_table[row_num]) < 1e-6, dec["folder"]

    m1450 = rec["monochromatic"]["1450"]
    f1450 = m1450["f_lambda_cgs"] if m1450["measured"] else ""
    rows.append({"row_num": row_num, "F1450": f1450})

rows.sort(key=lambda r: r["row_num"])
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["row_num", "F1450"])
    w.writeheader()
    w.writerows(rows)

n_meas = sum(1 for r in rows if r["F1450"] != "")
print("wrote %s: %d adopted objects, %d with F1450" % (OUT, len(rows), n_meas))
