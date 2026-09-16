"""
The one place to check the definitive pass: a table of adopted values and one
final figure per object.

    finalrun/pipeline_output/final/
        decisions.csv        the verdicts, hand-edited (see below)
        pass_decisions.csv   the table: one row per object worked, verdict,
                             reject reason, adopted numbers, which record
        plots/               one figure per object -- the adopted fit, or the
                             least-bad one prefixed REJECT_ for exclusions

On this pass an object ends as manmask 3 (final) or 0 (excluded), nothing in
between. Every exclusion carries a reason, from a controlled vocabulary so the
excluded objects group cleanly at the end -- and in particular so that objects
that passed the "is there data" query but still could not be measured are
separable from those that never had data:

    insufficient_coverage   usable data too narrow to constrain the fit
                            (e.g. no pixels in either anchor window)
    low_snr                 data present across the range but too noisy
    civ_unphysical          every fit variant puts the reconstruction below
                            the continuum inside C IV (the negative-EW veto)
    fit_unstable            the C IV blueshift moves by several quanta under
                            reasonable mask changes
    no_civ_data             no usable pixels in the C IV window at all

A fit can also be accepted with a stated reservation as manmask 2 ("OK for this
paper"): it carries its measurement into the table like a 3, and the note must
spell out what the reservation is. First used on 2MASX-J00391586-5117013, whose
best fit still sits 11.7% below the continuum across both wings of C IV.
    iue_only                no HST spectrum exists; the object is IUE-only and
                            is carried as manmask 1 -- the one value other
                            than 3 or 0 this pass assigns -- as the user's
                            reminder that it still has to be dealt with in
                            the IUE grouping. A placeholder, not a verdict.

Verdicts live in decisions.csv, one row per object folder:

    folder,manmask,reject_reason,note

Edit that file to change a verdict; the reason column takes several reasons
separated by semicolons. This script joins it with the BEST_ (or REJECT_ALL_)
record in each iteration folder, so the numbers always come from the fit that
was actually adopted, never from a hand-typed value.
"""

import csv
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")
ITERDIR = os.path.join(OUTDIR, "fit_iterations")
QUEUE = os.path.join(OUTDIR, "work_queue.csv")
FINAL = os.path.join(OUTDIR, "final")
DECISIONS = os.path.join(FINAL, "decisions.csv")
TABLE = os.path.join(FINAL, "pass_decisions.csv")
ZCHECK = os.path.join(FINAL, "redshift_check.csv")
PLOTS = os.path.join(FINAL, "plots")

# Seed for decisions.csv the first time it is built.
_INITIAL = [
    ("NGC_3516_COMBINED", 3, "",
     "blueshift identical across all mask sets; anchors aligned; trial-z test "
     "shows the anchor offsets are model shape, not redshift"),
    ("NGC4151_COMBINED", 3, "",
     "one quantum between mask sets; C III] is 3 sigma high in the adopted fit "
     "but the component set that fixes it breaks Si IV and Mg II at 12 sigma"),
    ("NGC4395_COMBINED", 3, "",
     "every variant passing the C IV veto gives -78.8 km/s; the variants that "
     "improve Si IV all fail the veto; 2300-2500 A non-AGN emission masked; "
     "user confirmed after trying the GUI"),
    ("NGC-4051_COMBINED", 0, "insufficient_coverage;civ_unphysical;fit_unstable",
     "1545 usable px all below 1800 A, no anchor coverage; all four variants "
     "fail the sub-continuum veto (77-123 px); blueshift swings 14 quanta"),
]


def safe_name(s):
    return "".join(c if (c.isalnum() or c in "+.-_ ") else "_" for c in s).strip()


def main():
    os.makedirs(PLOTS, exist_ok=True)
    for old in glob.glob(os.path.join(PLOTS, "*.png")):
        os.remove(old)          # regenerated below; stale names must not linger
    if not os.path.exists(DECISIONS):
        with open(DECISIONS, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["folder", "manmask", "reject_reason", "note"])
            w.writerows(_INITIAL)

    decisions = list(csv.DictReader(open(DECISIONS)))
    zcheck = ({r["folder"]: r for r in csv.DictReader(open(ZCHECK))}
              if os.path.exists(ZCHECK) else {})
    from object_paths import folder_for
    queue = {r["name_mast_key"]: r for r in csv.DictReader(open(QUEUE))}
    by_folder = {}
    for r in queue.values():
        by_folder[folder_for("%s_COMBINED" % r["name_mast_key"])] = r
        by_folder[folder_for("%s_SPLICE" % r["name_mast_key"])] = r
        # Single-spectrum objects are fit under their own stem.
        for st in r["stems"].split(";"):
            if st:
                by_folder[folder_for(st)] = r

    rows = []
    for dec in decisions:
        folder = dec["folder"]
        q = by_folder.get(folder, {})
        rec_dir = os.path.join(ITERDIR, folder, "records")
        best = glob.glob(os.path.join(rec_dir, "BEST_*.json"))
        rej = glob.glob(os.path.join(rec_dir, "REJECT_ALL_*.json"))
        chosen = best[0] if best else (rej[0] if rej else None)
        adopted = json.load(open(chosen)) if chosen else None
        manmask = int(float(dec["manmask"]))

        # One figure per object in final/plots. Named by the xlsx common_name,
        # which is the MAST key that also names the spectrum stems and hence
        # the fit_iterations folders -- so the same string finds the row in
        # the spreadsheet, the figure here, and the iterations. name_pub (the
        # SIMBAD-preferred name) is kept as an alias column only: the user
        # could not match "MCG-05-48-003" to a folder called ESO_462-9.
        name_pub = q.get("name_pub") or folder
        common_name = q.get("name_mast_key") or folder
        if chosen:
            src = os.path.join(ITERDIR, folder,
                               os.path.basename(chosen)[:-len(".json")] + ".png")
            if os.path.exists(src):
                # Prefixed with the queue index so the folder sorts into
                # working order and an object stays findable among hundreds.
                dst = os.path.join(PLOTS, "%s%03d_%s.png" % (
                    "REJECT_" if manmask == 0 else "",
                    int(q.get("order") or 0), safe_name(common_name)))
                shutil.copy2(src, dst)

        zc = zcheck.get(folder, {})
        rows.append({
            "row_num": q.get("row_num", ""),
            "index": q.get("order", ""),
            "common_name": common_name,
            "name_pub": name_pub,
            "name_mast_key": q.get("name_mast_key", ""),
            "name_simbad": q.get("name_simbad", ""),
            "designation": q.get("designation", ""),
            "category": q.get("category", ""),
            "manmask": manmask,
            "reject_reason": dec["reject_reason"],
            "had_data": "yes" if q.get("work_status") == "fit" else q.get("work_status", ""),
            "civ_blue": adopted["result"]["civ_blue"] if adopted and manmask in (2, 3) else "",
            "civ_ew": adopted["result"]["civ_ew"] if adopted and manmask in (2, 3) else "",
            "f2500": adopted["result"]["f2500"] if adopted and manmask in (2, 3) else "",
            "z_catalog": q.get("z", ""),
            "z_adopted": adopted["result"]["z"] if adopted else "",
            "z_ned": zc.get("z_ned", ""),
            "dv_ned_kms": zc.get("dv_ned_kms", ""),
            "z_simbad": zc.get("z_simbad", ""),
            "dv_simbad_kms": zc.get("dv_simbad_kms", ""),
            "z_flag": zc.get("flag", ""),
            "adopted_record": os.path.basename(chosen) if chosen else "none",
            "sulentic": q.get("sulentic", ""),
            "note": dec["note"],
        })

    rows.sort(key=lambda r: int(r["index"] or 0))
    with open(TABLE, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("%-5s %-4s %-16s %-3s %-40s %9s %s" % ("index", "row", "common_name", "mm", "reject_reason", "blue", "record"))
    for r in rows:
        print("%-5s %-4s %-16s %-3s %-40s %9s %s" % (
            r["index"], r["row_num"], r["common_name"][:16], r["manmask"], r["reject_reason"] or "-",
            ("%.1f" % r["civ_blue"]) if r["civ_blue"] != "" else "-", r["adopted_record"]))
    print("\ntable : %s\nplots : %s/  (%d figures)\nedit  : %s\nz     : %s%s"
          % (TABLE, PLOTS, len(os.listdir(PLOTS)), DECISIONS, ZCHECK,
             "" if zcheck else "   (missing -- run check_redshift.py)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
