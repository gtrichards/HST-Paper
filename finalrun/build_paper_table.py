"""
The paper table, in the xlsx's own layout.

One file, one row per object in CIV_measurements_v22.xlsx, with exactly the
columns the paper table needs:

    row_num  designation  common_name  z  instrument  adopted  sdss_flag
    sul_flag  cat_flags  CIV_blue  CIV_EW  CIVgood_manmask

`instrument` is every spectrum that exists for the object, carried from the xlsx;
`adopted` is the one the measurement was actually made from, which is not the
same thing. Several objects fit better on a single spectrum than on the merge of
all of them -- NGC 3783 and NGC 5548 on FOS alone, NGC 7469 on STIS alone -- and
without this column the table would read as though all three instruments went
into those numbers. It is blank for an excluded object, which has no measurement,
and for an object not yet worked.

The identity columns come straight from the xlsx. The measurement columns are
overwritten for every object that has been through the definitive pass -- taken
from the adopted BEST_ record, never typed in -- and carried forward unchanged
from the xlsx for objects not yet worked. So the file is always the current
best table: the spine of the v22 list, iterated in place, with nothing dropped.

An object excluded on the pass gets manmask 0 and blank measurements; the
reason for the exclusion is in final/pass_decisions.csv, which is the working
ledger this table is distilled from.

Redshifts are the xlsx values unless the adopted fit ran at a different one, in
which case the adopted value is written and the change is reported.
"""

import csv
import glob
import json
import os
import re
import sys

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")
from rebin_path import ITERDIR   # env HSTICA_ITERDIR overrides

#: Where the definitive pass writes its fits.  The decisions ledger is keyed on
#: a folder name that predates the single-visit policy -- 092's row is still
#: 092_2MASS_J08105865+7602424_COMBINED while its fit lives in
#: ..._COS -- so reading ITERDIR/<decisions folder> found the PRE-PASS fit, or
#: nothing, for 20 of the 73 objects that carry a measurement.  The table came
#: out with stale numbers and nothing said so: PG 0804+761 read EW 34.78, the
#: old merge value, against the 28.47 adopted that morning.  The index prefix is
#: the stable identifier -- both names begin "092_" -- so the fit is found by
#: index and the ledger's folder name is left alone as the historical key it is.
CHOSEN = os.path.join(OUTDIR, "chosen_visit", "fits")

#: Where to look for an adopted fit, in order.  Three trees exist and each was
#: the working one at some point, so "the adopted fit" is whichever the most
#: recent tree holds:
#:
#:   chosen_visit/fits    this pass, under the single-visit policy
#:   fit_iterations_v23   the re-review on the regenerated data set; this is the
#:                        tree the decisions notes cite ("BEST_iter03_allmasks in
#:                        fit_iterations_v23/097_Ton_S_210_COMBINED")
#:   fit_iterations       the first pass, and the default ITERDIR
#:
#: Reading only ITERDIR missed both of the others: Ton S 210 and 3C 273 are
#: graded 3 with their adopted fits sitting in fit_iterations_v23, and the table
#: wrote empty measurements for them without complaint.  The fallback is what
#: keeps the file "always the current best" for objects the pass has not reached.
FIT_TREES = [CHOSEN,
             os.path.join(OUTDIR, "fit_iterations_v23"),
             ITERDIR]
FINAL = os.path.join(OUTDIR, "final")
DECISIONS = os.path.join(FINAL, "decisions.csv")
QUEUE = os.path.join(OUTDIR, "work_queue.csv")
XLSX = "/Users/gtr/Dropbox/HST/Summer2026/CIV_measurements_v22.xlsx"
OUT = os.path.join(FINAL, "CIV_measurements_v23.csv")

#: Objects whose catalogue name is not the name an astronomer would look for.
#: The xlsx `common_name` and the queue's own precedence (Sulentic, then SIMBAD,
#: then MAST) both produce survey designations for a handful of famous objects:
#: row 22 is I Zw 1, the prototype narrow-line Seyfert 1, carried as Mrk 1502
#: and as UGC 00545; row 94 is PG 0804+761, a standard reverberation-mapping
#: target, carried as 2MASS J08105865+7602424.  A reader scanning the paper's
#: table will not find either.  One row per object with the reason and the date,
#: the same discipline as every other hand decision in final/.
NAME_OVERRIDES = os.path.join(FINAL, "name_overrides.csv")


def _fit_folders_by_index():
    """Adopted fits by object index: (tree, folder), most recent tree first.

    Keyed on the index rather than the folder name because the decisions ledger
    predates the single-visit policy and its folder names no longer match: 092's
    row is still ..._COMBINED while its fit is in ..._COS.  Both begin "092_".
    """
    out = {}
    for tree in FIT_TREES:
        if not os.path.isdir(tree):
            continue
        for d in sorted(os.listdir(tree)):
            m = re.match(r"^(\d{3})_", d)
            if not m or not os.path.isdir(os.path.join(tree, d)):
                continue
            i = int(m.group(1))
            if i in out:
                continue          # an earlier tree in the list already has it
            if glob.glob(os.path.join(tree, d, "records", "BEST_*.json")):
                out[i] = (tree, d)
    return out


def _name_overrides():
    out = {}
    if not os.path.exists(NAME_OVERRIDES):
        return out
    for r in csv.DictReader(open(NAME_OVERRIDES)):
        if str(r.get("row_num", "")).strip() and str(r.get("name", "")).strip():
            out[int(r["row_num"])] = r["name"].strip()
    return out

XLSX_COLS = ["row_num", "designation", "common_name", "z", "instrument",
             "sdss_flag", "sul_flag", "cat_flags", "CIV_blue", "CIV_EW",
             "CIVgood_manmask"]
COLS = XLSX_COLS[:5] + ["adopted"] + XLSX_COLS[5:]


def main():
    ws = openpyxl.load_workbook(XLSX)["CIV_measurements_v22"]
    hdr = [c.value for c in ws[1]]
    idx = {h: n for n, h in enumerate(hdr)}
    rows = []
    for row in ws.iter_rows(min_row=2):
        v = [c.value for c in row]
        if not isinstance(v[idx["row_num"]], (int, float)) or not v[idx["designation"]]:
            continue          # blank row and the totals row of formulas
        r = {c: v[idx[c]] for c in XLSX_COLS}
        r["adopted"] = None
        rows.append(r)

    # Applied after the xlsx is read and before anything is keyed on the name,
    # so the override is the name everywhere downstream, not a late relabel.
    _nov = _name_overrides()
    for r in rows:
        new_name = _nov.get(int(r["row_num"]))
        if new_name and new_name != r["common_name"]:
            print("name override: row %d  %r -> %r"
                  % (int(r["row_num"]), r["common_name"], new_name))
            r["common_name"] = new_name

    # Map each decision folder back to its xlsx row via the queue.
    from object_paths import folder_for
    queue = {r["name_mast_key"]: r for r in csv.DictReader(open(QUEUE))}
    by_folder = {}
    for r in queue.values():
        by_folder[folder_for("%s_COMBINED" % r["name_mast_key"])] = r
        by_folder[folder_for("%s_SPLICE" % r["name_mast_key"])] = r
        for st in r["stems"].split(";"):
            if st:
                by_folder[folder_for(st)] = r
    by_rownum = {int(r["row_num"]): r for r in rows}

    updated, z_changed = [], []
    fits_by_index = _fit_folders_by_index()
    from_pass, from_legacy, no_record = 0, 0, []
    if os.path.exists(DECISIONS):
        for dec in csv.DictReader(open(DECISIONS)):
            q = by_folder.get(dec["folder"])
            if q is None:
                print("no queue entry for %s" % dec["folder"])
                continue
            target = by_rownum[int(q["row_num"])]
            manmask = int(float(dec["manmask"]))
            # Prefer this pass's fit, found by the index prefix; fall back to the
            # pre-pass tree for objects the pass has not reached yet, which is
            # what keeps the table "always the current best" rather than empty.
            m = re.match(r"^(\d{3})_", dec["folder"])
            found = fits_by_index.get(int(m.group(1))) if m else None
            if found:
                tree, fit_folder = found
                rec_dir = os.path.join(tree, fit_folder, "records")
                if tree == CHOSEN:
                    from_pass += 1
                elif manmask in (2, 3):
                    from_legacy += 1
            else:
                fit_folder = dec["folder"]
                rec_dir = os.path.join(ITERDIR, fit_folder, "records")
            best = glob.glob(os.path.join(rec_dir, "BEST_*.json"))
            adopted = json.load(open(best[0])) if best else None
            if manmask in (2, 3) and adopted is None:
                no_record.append(dec["folder"])
            target["CIVgood_manmask"] = manmask
            # Named for the spectrum the fit was actually run on -- the FIT
            # folder, not the ledger's -- so a re-chosen visit or a splice shows
            # up here instead of the instrument the merge era recorded.
            # COMBINED means every spectrum the object has.
            spec = fit_folder.split("_")[-1]
            # 3 = final; 2 = accepted for this paper with a reservation stated in
            # the ledger note (first used on 2MASX-J00391586-5117013, a BAL whose
            # best fit still sits below the continuum in both wings). Both carry
            # their measurement; 0 and 1 do not.
            target["adopted"] = (target["instrument"] if spec == "COMBINED" else spec) \
                if manmask in (2, 3) else None
            if manmask in (2, 3) and adopted:
                target["CIV_blue"] = adopted["result"]["civ_blue"]
                target["CIV_EW"] = adopted["result"]["civ_ew"]
                z_fit = adopted["result"]["z"]
                try:
                    if abs(float(z_fit) - float(target["z"])) > 1e-9:
                        z_changed.append((target["common_name"], target["z"], z_fit))
                        target["z"] = z_fit
                except (TypeError, ValueError):
                    pass
            else:
                target["CIV_blue"] = None
                target["CIV_EW"] = None
            updated.append((target["common_name"], manmask))

    os.makedirs(FINAL, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})

    n3 = sum(1 for r in rows if r["CIVgood_manmask"] in (3, 3.0, "3"))
    n0 = sum(1 for r in rows if r["CIVgood_manmask"] in (0, 0.0, "0"))
    print("wrote %d rows -> %s" % (len(rows), OUT))
    print("measurements read from: this pass %d, pre-pass tree %d"
          % (from_pass, from_legacy))
    if no_record:
        print("NO adopted record for %d object(s) graded 2 or 3: %s"
              % (len(no_record), ", ".join(no_record)))
    print("updated on this pass: %d  (%s)" % (len(updated),
          ", ".join("%s=%d" % u for u in updated)))
    print("manmask now: 3 -> %d, 0 -> %d (of which %d from this pass)"
          % (n3, n0, sum(1 for _, m in updated if m == 0)))
    for name, old, new in z_changed:
        print("redshift changed for %s: %s -> %s" % (name, old, new))
    return 0


if __name__ == "__main__":
    sys.exit(main())
