"""
Refit every recorded iteration in place, after a change to the fit code.

Each iteration record holds the override that produced it, and the fit path is
deterministic, so replaying a record reproduces its fit exactly -- which means
that after a change to the measurement code, replaying every record brings the
whole tree onto the new arithmetic without any judgement being revisited. The
BEST_ and REJECT_ALL_ prefixes are preserved, since which variant was chosen is
a decision, not a computation.

Reports what moved, so the effect of the change is visible rather than silently
absorbed.
"""

import csv
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from rebin_path import ITERDIR   # env HSTICA_ITERDIR overrides
NOISE = re.compile(r"^(File column|    name =|\)|Debug|Loaded|Using|Median)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--adopted-only", action="store_true",
                    help="replay only the BEST_ / REJECT_ALL_ records (the ones the tables are built "
                         "from); the rest of the tree can follow later")
    ap.add_argument("--out", default=None, help="CSV to write (default pipeline_output/refit_<label>.csv)")
    args = ap.parse_args()
    records = sorted(
        f for f in glob.glob(os.path.join(ITERDIR, "*", "records", "*.json"))
        if os.path.basename(f) not in ("trial_z.csv", "gui_overrides.json")
        and "trial_z" not in os.path.basename(f)
        and (not args.adopted_only or os.path.basename(f).startswith(("BEST_", "REJECT_ALL_"))))

    rows, failed = [], []
    for i, rec_path in enumerate(records, 1):
        try:
            rec = json.load(open(rec_path))
            name = rec["name"]
            before = (rec["result"]["civ_blue"], rec["result"]["civ_ew"])
        except (KeyError, ValueError):
            continue
        r = subprocess.run([sys.executable, os.path.join(HERE, "iterate_fit.py"),
                            name, "--replot", rec_path],
                           capture_output=True, text=True)
        if r.returncode:
            failed.append((os.path.basename(rec_path),
                           " | ".join(l for l in r.stderr.splitlines()[-2:])))
            continue
        after_rec = json.load(open(rec_path))
        after = (after_rec["result"]["civ_blue"], after_rec["result"]["civ_ew"])
        tag = os.path.basename(rec_path)[:-5]
        rows.append(dict(folder=os.path.basename(os.path.dirname(os.path.dirname(rec_path))),
                         record=tag, name=name,
                         blue_before=before[0], blue_after=after[0],
                         ew_before=before[1], ew_after=after[1],
                         adopted=tag.startswith(("BEST_", "REJECT_ALL_"))))
        if tag.startswith("BEST_"):
            print("  %-26s %-22s blue %8.1f -> %8.1f   ew %7.2f -> %7.2f  (%+.1f%%)"
                  % (rows[-1]["folder"], tag[:22], before[0], after[0], before[1], after[1],
                     100 * (after[1] - before[1]) / before[1] if before[1] else 0), flush=True)
        if i % 10 == 0:
            print("     [%d/%d]" % (i, len(records)), flush=True)

    out = args.out or os.path.join(HERE, "pipeline_output", "refit_after_morphfix%s.csv" % ("_adopted" if args.adopted_only else ""))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    moved = sum(1 for r in rows if abs(r["blue_after"] - r["blue_before"]) > 1)
    print("\nrefit %d records; blueshift moved in %d of them" % (len(rows), moved))
    if failed:
        print("FAILED %d:" % len(failed))
        for n, e in failed:
            print("   %-40s %s" % (n, e))
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
