"""Replay each adopted fit against its rebuilt product, as a NEW iteration.

Generalised from replay_nread.py (2026-11-12), which had its label hard-coded.
Same rule, learned the hard way: NOT in place.  GTR, when a replay overwrote
adopted records: "Wait, so you overwrote instead of making a new one?"  Each
object's adopted override (masks, pixels, prior) is re-run through
iterate_fit --base, which writes iterNN_<label>.json and .png beside the
adopted fit and leaves BEST_ untouched.  The adopted record is read from the
SNAPSHOT, so a replay cannot chase its own tail.

    python replay_fits.py --snap pipeline_output/archive_pre_sqrtn_20261117_0904 --label sqrtn
"""
import argparse, glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FITS = os.path.join(HERE, "pipeline_output", "chosen_visit", "fits")
PY = "/opt/homebrew/Caskroom/miniforge/base/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    idx = [int(x) for x in open(os.path.join(a.snap, "INDICES.txt")).read().split()]
    env = dict(os.environ)
    env["PYTHONPATH"] = "/Users/gtr/Work/git/HST-Paper"
    env["HSTICA_REBIN"] = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
    env["HSTICA_ITERDIR"] = FITS
    done = skipped = failed = 0
    for i in idx:
        dirs = glob.glob(os.path.join(a.snap, "fits", "%03d_*" % i))
        rec = []
        for d in dirs:
            rec += sorted(glob.glob(os.path.join(d, "records", "BEST_*.json"))) or \
                   sorted(glob.glob(os.path.join(d, "records", "REJECT_ALL_*.json")))
        if not rec:
            print("  %03d  no adopted record in snapshot -- skipped" % i, flush=True); skipped += 1; continue
        name = json.load(open(rec[0]))["name"]
        r = subprocess.run([PY, os.path.join(HERE, "iterate_fit.py"), name,
                            "--base", rec[0], "--label", a.label],
                           cwd=HERE, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print("  %03d  %-28s FAILED\n%s" % (i, name[:28], r.stderr[-300:]), flush=True); failed += 1; continue
        line = [l for l in r.stdout.splitlines() if l.startswith(name)]
        print("  %03d  %-28s %s" % (i, name[:28], line[0][len(name):].strip() if line else "ok"), flush=True)
        done += 1
    print("\nreplayed %d   skipped %d   failed %d" % (done, skipped, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
