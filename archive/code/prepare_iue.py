"""Generate the fit variants for IUE objects so each is ready to judge.

The IUE sibling of finalrun/prepare_objects.py.  That script cannot be reused:
it is built on the HST work queue and the visit staging, neither of which an IUE
object has.  What carries over is the purpose -- an object should be ready the
moment GTR comes to it, not half-prepared while he waits -- and the variants.

Per object it writes, into iue/fits/000_<name>_IUE:

    iter*_nomask_auto            no masks, components chosen automatically
    iter*_nomask_{low,mod,high}  no masks, each component family forced

and prints, per variant, the blueshift, EW, usable C IV pixel count and the
sub-continuum veto verdict, so the choice between families is visible before the
GUI is opened.  GTR then saves his own mask, which becomes iter*_gtrmask.

WHAT TO DISTRUST, AND WHY IT IS PRINTED.  IUE per-pixel C IV S/N across this set
spans 0.3 to 11 (see triage_iue.py).  Below about 1.5 the error bars swallow any
smooth curve: the residual panel looks fine and the reconstruction is
unconstrained -- the hollow-residual trap the HST pass hit repeatedly.  The
per-pixel S/N is therefore printed beside every object, and for anything under
1.5 the agreement between component families matters more than the residuals,
because families that disagree by hundreds of km/s on the same data are telling
you the data do not constrain the answer.

A second IUE-specific caution.  The 69.03 km/s lattice oversamples IUE's 1.5 A
pixels by about 4.7, so neighbouring pixels are correlated and a residual rms
computed per pixel is optimistic.  The native numbers in iue_assessment.csv are
the honest ones; the HST judging thresholds do not transfer.

    python prepare_iue.py "QSO 1629+372" "OJ 49"
    python prepare_iue.py --batch 1        # the next four by triage order
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINALRUN = os.path.dirname(HERE)
REBIN = os.path.join(HERE, "rebin")
FITS = os.path.join(HERE, "fits")
PY = sys.executable

VARIANTS = [("auto", None), ("low", "low"), ("mod", "mod"), ("high", "high")]


def env():
    e = dict(os.environ)
    e["HSTICA_REBIN"] = REBIN
    e["HSTICA_ITERDIR"] = FITS
    e["HSTICA_OVERRIDES"] = os.path.join(HERE, "iue_overrides.json")
    e["PYTHONPATH"] = "/Users/gtr/Work/git/HST-Paper:%s" % FINALRUN
    return e


def run(stem, label, comps):
    cmd = [PY, os.path.join(FINALRUN, "iterate_fit.py"), stem,
           "--label", "nomask_%s" % label]
    if comps:
        cmd += ["--comps", comps]
    p = subprocess.run(cmd, cwd=FINALRUN, env=env(), capture_output=True, text=True)
    return p.stdout + p.stderr


def parse(out):
    d = {}
    m = re.search(r"blue=\s*(-?[\d.]+)\s+ew=\s*(-?[\d.]+)", out)
    if m:
        d["blue"], d["ew"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"C IV window coverage:\s*(\d+) usable px", out)
    if m:
        d["civ_px"] = int(m.group(1))
    m = re.search(r"C IV sub-continuum: (\d+) px, worst (-?[\d.]+)%", out)
    if m:
        d["veto_px"], d["veto_worst"] = int(m.group(1)), float(m.group(2))
    d["reject"] = "*** REJECT ***" in out
    m = re.search(r"->\s*(\S+\.png)", out)
    if m:
        d["png"] = m.group(1)
    m = re.search(r"CIV: med (-?[\d.+]+) rms ([\d.]+)", out)
    if m:
        d["civ_rms"] = float(m.group(2))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--batch", type=int, default=0,
                    help="take the next four unjudged objects in triage order")
    a = ap.parse_args()

    tri = pd.read_csv(os.path.join(HERE, "iue_triage.csv"))
    if a.batch:
        todo = [str(r["common_name"]) for _, r in tri.iterrows()
                if not str(r.get("state") or "").strip()][(a.batch - 1) * 4:a.batch * 4]
    else:
        todo = a.names
    if not todo:
        print("nothing to prepare")
        return

    sn = {str(r["common_name"]): float(r["sn_civ"]) for _, r in tri.iterrows()}
    for nm in todo:
        stem = "%s_IUE" % nm
        if not os.path.exists(os.path.join(REBIN, "%s.fits" % stem)):
            print("%-16s no rebinned product" % nm)
            continue
        s = sn.get(nm, float("nan"))
        warn = "   *** per-pixel S/N %.2f: HOLLOW-RESIDUAL TRAP, judge on family agreement" % s \
            if np.isfinite(s) and s < 1.5 else ""
        print("\n%s   (per-pixel C IV S/N %.2f)%s" % (nm, s, warn))
        print("   %-6s %9s %8s %7s %7s %s"
              % ("comps", "blue", "ew", "civpx", "rms", "veto"))
        got = {}
        for label, comps in VARIANTS:
            r = parse(run(stem, label, comps))
            got[label] = r
            if "blue" not in r:
                print("   %-6s   FIT FAILED" % label)
                continue
            v = ("%d px %.1f%%%s" % (r.get("veto_px", 0), r.get("veto_worst", 0.0),
                                     "  REJECT" if r["reject"] else "")
                 if r.get("veto_px") else "clean")
            print("   %-6s %+9.1f %8.2f %7d %7s %s"
                  % (label, r["blue"], r["ew"], r.get("civ_px", -1),
                     ("%.2f" % r["civ_rms"]) if "civ_rms" in r else "--", v))
        bs = [r["blue"] for r in got.values() if "blue" in r]
        if len(bs) > 1:
            spread = max(bs) - min(bs)
            print("   family spread: %.1f km/s = %.1f quanta%s"
                  % (spread, spread / 69.03,
                     "   *** families disagree; the data may not constrain this"
                     if spread > 4 * 69.03 else ""))
        print("   open with:  ./open_fitter_iue.sh \"%s\"" % stem)


if __name__ == "__main__":
    main()
