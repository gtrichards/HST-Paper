"""Add any object that has a fit on disk but no row in chosen_visit_summary.csv.

Nothing writes that file.  It was produced once, on 2026-09-27, and three
separate readers key off it:

    iterate_fit._sdss_file_for_stem   the SDSS overlay on every diagnostic figure
    manual_fix_gui._sdss_overlay      the Mg II inset in the interactive fitter
    check_oiii                        the redshift-override lookup

Each resolves a spectrum stem to an object index through this file, and each
returns None -- silently -- when the stem is absent.  So an object added to the
tree after that date loses its SDSS overlay and its Mg II inset with no error,
no warning and nothing in the log.  GTR found it on index 382: "There is no
MgII inset for the interactive fitter.  What's the problem here?  Why wasn't
this block prepared exactly like the last one?"  The block HAD been prepared
identically; 382 simply postdated the file, as did 216, 365 and 374.

Only `index` and `stem` are read, so this fills those and leaves the other
columns blank rather than recomputing numbers it has no business inventing.
Existing rows are never touched.

    python reconcile_chosen_visit_summary.py            # report only
    python reconcile_chosen_visit_summary.py --write
"""
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "pipeline_output", "final", "chosen_visit_summary.csv")
sys.path.insert(0, HERE)
from rebin_path import ITERDIR


def reconcile(write=False, quiet=False):
    """Add missing (index, stem) rows.  Returns how many were added.

    Importable so fit_chosen_visit can call it at the end of a run, which is
    when new stems appear; see the note there.
    """
    with open(SUMMARY, newline="") as f:
        rows = list(csv.DictReader(f))
        cols = list(rows[0].keys()) if rows else ["index", "stem"]
    have = {str(r.get("stem", "")).strip() for r in rows}

    found = []
    for d in sorted(glob.glob(os.path.join(ITERDIR, "*"))):
        m = re.match(r"^(\d{3})_", os.path.basename(d))
        if not m:
            continue
        recs = sorted(glob.glob(os.path.join(d, "records", "*.json")))
        if not recs:
            continue
        try:
            stem = json.load(open(recs[0]))["name"]
        except Exception:
            continue
        if stem not in have:
            found.append((int(m.group(1)), stem))

    if not found:
        if not quiet:
            print("chosen_visit_summary.csv is complete: every fit on disk has a row")
        return 0
    if not quiet:
        print("%d object(s) with a fit but no summary row:" % len(found))
        for i, s in found:
            print("   %3d  %s" % (i, s))
    if not write:
        if not quiet:
            print("\nreport only -- re-run with --write to add them")
        return 0
    for i, s in found:
        rows.append({c: "" for c in cols} | {"index": str(i), "stem": s})
    rows.sort(key=lambda r: (int(r["index"]) if str(r.get("index", "")).strip()
                             .lstrip("-").isdigit() else 10**6))
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    if not quiet:
        print("\nadded %d row(s); chosen_visit_summary.csv now has %d" % (len(found), len(rows)))
    return len(found)


def main():
    reconcile(write="--write" in sys.argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
