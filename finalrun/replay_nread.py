"""Replay each adopted fit against its NREAD-rebuilt product, as a NEW iteration.

NOT in place.  refit_all.py rewrites a record under its own filename, which is
right after a pure code change but wrong here: these 108 objects each need a
judgement, so the new numbers must sit BESIDE the adopted fit as a candidate.
GTR, when I did it the other way: "Wait, so you overwrote instead of making a
new one?"

Each object's adopted override is re-run through iterate_fit --base, which
writes iterNN_nread.json and .png into the same fit folder and leaves BEST_
untouched.

    python replay_nread.py --snap pipeline_output/archive_pre_nread_...
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FITS = os.path.join(HERE, "pipeline_output", "chosen_visit", "fits")
PY = "/opt/homebrew/Caskroom/miniforge/base/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    a = ap.parse_args()
    idx = [int(x) for x in open(os.path.join(a.snap, "INDICES.txt")).read().split()]

    env = dict(os.environ)
    env["PYTHONPATH"] = "/Users/gtr/Work/git/HST-Paper"
    env["HSTICA_REBIN"] = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
    env["HSTICA_ITERDIR"] = FITS

    done = skipped = failed = 0
    for i in idx:
        dirs = glob.glob(os.path.join(FITS, "%03d_*" % i))
        if not dirs:
            skipped += 1
            continue
        d = dirs[0]
        rec = (sorted(glob.glob(os.path.join(d, "records", "BEST_*.json")))
               or sorted(glob.glob(os.path.join(d, "records", "REJECT_ALL_*.json"))))
        if not rec:
            print("  %03d  no adopted record (manmask 0) -- skipped" % i, flush=True)
            skipped += 1
            continue
        name = json.load(open(rec[0]))["name"]
        r = subprocess.run([PY, os.path.join(HERE, "iterate_fit.py"), name,
                            "--base", rec[0], "--label", "nread"],
                           cwd=HERE, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print("  %03d  %-28s FAILED\n%s" % (i, name[:28], r.stderr[-400:]), flush=True)
            failed += 1
            continue
        line = [l for l in r.stdout.splitlines() if "blue=" in l]
        print("  %03d  %-28s %s" % (i, name[:28], line[0].strip() if line else "ok"), flush=True)
        done += 1
    print()
    print("replayed %d, skipped %d, failed %d" % (done, skipped, failed))


if __name__ == "__main__":
    main()
