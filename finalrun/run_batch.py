"""
Run the first pass on a batch of queue objects, end to end.

For each object, in order:

  1. Build the combined spectrum if it has more than one -- every HST spectrum
     of an object goes into the fit, since C IV's shape is constrained by the
     whole UV range -- or use the single spectrum as is.
  2. Copy in whatever prior fits exist (the user's ManualFix figures, Alex's
     _nomask baselines) so the comparison is one folder.
  3. Propose masks: absorption and cosmic-ray spikes, grown generously, with the
     C IV candidates held separately.
  4. Fit three variants through iterate_fit.py: baseline, the non-C IV masks,
     and everything including the C IV candidates. Every variant is screened
     for sub-continuum pixels in C IV, which is a veto.
  5. Mark BEST_ by a stated rule: the most fully masked variant that passes the
     veto. If none passes, mark the least-bad one REJECT_ALL_. The rule is a
     starting point for review, not a verdict -- the figures are what decide.

Writes a summary table for the batch. Verdicts are entered afterwards in
final/decisions.csv by hand, and finalize.py rebuilds the paper table.
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUTDIR = os.path.join(HERE, "pipeline_output")
from rebin_path import ITERDIR   # env HSTICA_ITERDIR overrides
WORKDIR = os.path.join(OUTDIR, "coadd_work")
QUEUE = os.path.join(OUTDIR, "work_queue.csv")
from rebin_path import REBIN as _REBIN_DEFAULT
REBIN = _REBIN_DEFAULT
GTR_PLOTS = "/Users/gtr/Work/git/HST-Paper/ICA_Plots_Rebin_master/ManualFix"
AP_PLOTS = "/Users/gtr/Dropbox/HST/Pratsos/ICA_Plots_Rebin_master"

NOISE = re.compile(r"^(File column|    name =|\)|Debug|Loaded|Using|Median)")


def norm(s):
    return re.sub(r"[^a-z0-9+.-]", "", s.lower())


from object_paths import folder_for  # order-prefixed, shared by every script


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = [l for l in (r.stdout + r.stderr).splitlines() if not NOISE.match(l)]
    return r.returncode, out


def copy_priors(stems, folder):
    d = os.path.join(ITERDIR, folder)
    os.makedirs(d, exist_ok=True)
    got = []
    keys = {norm(s) for s in stems}
    for f in os.listdir(GTR_PLOTS):
        if f.endswith(".png") and norm(f[:-4]) in keys:
            shutil.copy2(os.path.join(GTR_PLOTS, f), os.path.join(d, "prior_gtr_" + f))
            got.append("gtr:" + f)
    for f in os.listdir(AP_PLOTS):
        if f.endswith("_nomask.png") and norm(f[:-len("_nomask.png")]) in keys:
            shutil.copy2(os.path.join(AP_PLOTS, f), os.path.join(d, "prior_ap_" + f))
            got.append("ap:" + f)
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", type=int, nargs="+", required=True)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    from coadd_experiment import coadd, write_fits

    queue = {int(r["order"]): r for r in csv.DictReader(open(QUEUE))}
    summary = []
    for o in args.orders:
        q = queue[o]
        stems = [s for s in q["stems"].split(";") if s]
        if not stems:
            # No HST spectrum at all (the IUE-only objects). Nothing to fit, but
            # the object still gets its look and its decision row (manmask 1,
            # iue_only) -- entered by hand in decisions.csv, not here.
            print("[%d] %s: no HST spectra (IUE-only) -- skipped, needs an iue_only row"
                  % (o, q["name_pub"]), flush=True)
            summary.append(dict(order=o, name=q["name_pub"], folder="", nspec=0, pick="iue_only"))
            continue
        if len(stems) > 1:
            name = "%s_COMBINED" % q["name_mast_key"]
            fits_path = os.path.join(WORKDIR, name + ".fits")
            if not os.path.exists(fits_path):
                wave, flux, errs, mask, z, nspec = coadd(stems)
                write_fits(fits_path, wave, flux, errs, mask, z)
                print("[%d] %s: combined %d spectra -> %d px (%d overlap); scales %s"
                      % (o, q["name_pub"], len(stems), wave.size, int((nspec > 1).sum()),
                         coadd.last_scales), flush=True)
        else:
            name = stems[0]
        folder = folder_for(name)
        priors = copy_priors(stems, folder)
        print("[%d] %s -> %s   priors: %s" % (o, q["name_pub"], name, ", ".join(priors) or "none"), flush=True)

        rc, out = run([PY, os.path.join(HERE, "mask_proposer.py"), "--orders", str(o)])
        for l in out:
            if l.startswith("order"):
                print("     " + l.strip(), flush=True)

        variants = [("baseline", []),
                    ("noncivmasks", ["--from-proposals", "--max-width", "8", "--grow-red", "1"]),
                    ("allmasks", ["--from-proposals", "--include-civ", "--max-width", "8", "--grow-red", "1"])]
        results = {}
        for label, extra in variants:
            rec_dir = os.path.join(ITERDIR, folder, "records")
            before = set(os.listdir(rec_dir)) if os.path.isdir(rec_dir) else set()
            rc, out = run([PY, os.path.join(HERE, "iterate_fit.py"), name, "--label", label] + extra)
            after = set(os.listdir(rec_dir)) if os.path.isdir(rec_dir) else set()
            new = [f for f in after - before if f.endswith(".json")]
            if rc or not new:
                print("     %-12s FAILED: %s" % (label, " | ".join(out[-3:])), flush=True)
                results[label] = None
                continue
            rec = json.load(open(os.path.join(rec_dir, new[0])))
            sub = rec.get("civ_subcontinuum", {})
            results[label] = dict(file=new[0][:-5], blue=rec["result"]["civ_blue"],
                                  ew=rec["result"]["civ_ew"], veto=sub.get("ok"),
                                  sub_px=sub.get("n_px", 0), windows=rec.get("windows", ""),
                                  zchk=rec.get("redshift_check_kms", {}))
            r = results[label]
            print("     %-12s blue=%8.1f ew=%7.2f  veto=%s  %s"
                  % (label, r["blue"], r["ew"], "pass" if r["veto"] else "REJECT(%d)" % r["sub_px"],
                     r["windows"][:70]), flush=True)

        # BEST_ by rule: fullest mask set that passes the veto.
        pick = None
        for label in ("allmasks", "noncivmasks", "baseline"):
            r = results.get(label)
            if r and r["veto"] and r["ew"] > 0:
                pick = (label, "BEST_")
                break
        if pick is None:
            cands = [(r["sub_px"], l) for l, r in results.items() if r]
            if cands:
                pick = (min(cands)[1], "REJECT_ALL_")
        if pick:
            label, prefix = pick
            f = results[label]["file"]
            d = os.path.join(ITERDIR, folder)
            os.rename(os.path.join(d, f + ".png"), os.path.join(d, prefix + f + ".png"))
            os.rename(os.path.join(d, "records", f + ".json"), os.path.join(d, "records", prefix + f + ".json"))
            print("     -> %s%s" % (prefix, f), flush=True)
        summary.append(dict(order=o, name=q["name_pub"], folder=folder, nspec=len(stems),
                            pick=("%s%s" % pick[::-1]) if pick else "none",
                            **{"%s_%s" % (l, k): (results[l][k] if results.get(l) else "")
                               for l in ("baseline", "noncivmasks", "allmasks")
                               for k in ("blue", "ew", "veto", "sub_px")}))

    out = args.summary or os.path.join(OUTDIR, "batch_summary.csv")
    with open(out, "w", newline="") as fh:
        keys = []
        for row in summary:
            keys += [k for k in row if k not in keys]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    print("\n-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
