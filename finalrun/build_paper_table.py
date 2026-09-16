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
import sys

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")
ITERDIR = os.path.join(OUTDIR, "fit_iterations")
FINAL = os.path.join(OUTDIR, "final")
DECISIONS = os.path.join(FINAL, "decisions.csv")
QUEUE = os.path.join(OUTDIR, "work_queue.csv")
XLSX = "/Users/gtr/Dropbox/HST/Summer2026/CIV_measurements_v22.xlsx"
OUT = os.path.join(FINAL, "CIV_measurements_v23.csv")

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
    if os.path.exists(DECISIONS):
        for dec in csv.DictReader(open(DECISIONS)):
            q = by_folder.get(dec["folder"])
            if q is None:
                print("no queue entry for %s" % dec["folder"])
                continue
            target = by_rownum[int(q["row_num"])]
            manmask = int(float(dec["manmask"]))
            rec_dir = os.path.join(ITERDIR, dec["folder"], "records")
            best = glob.glob(os.path.join(rec_dir, "BEST_*.json"))
            adopted = json.load(open(best[0])) if best else None
            target["CIVgood_manmask"] = manmask
            # The folder is named for the spectrum the fit was run on; COMBINED
            # means every spectrum the object has, which is the instrument column.
            spec = dec["folder"].split("_")[-1]
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
    print("updated on this pass: %d  (%s)" % (len(updated),
          ", ".join("%s=%d" % u for u in updated)))
    print("manmask now: 3 -> %d, 0 -> %d (of which %d from this pass)"
          % (n3, n0, sum(1 for _, m in updated if m == 0)))
    for name, old, new in z_changed:
        print("redshift changed for %s: %s -> %s" % (name, old, new))
    return 0


if __name__ == "__main__":
    sys.exit(main())
