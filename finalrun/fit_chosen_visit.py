"""Build and fit an object from the single visit the selection policy chooses.

Stages only that visit's surviving exposures into a scratch directory, rebins
them through the unchanged pipeline, and runs the fit, so the result is what the
policy would actually produce rather than an estimate of it.

    python fit_chosen_visit.py --index 40 11 19
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from exposure_decisions import (read_exposures, per_file, assign_epochs, recommend,
                                alignment, choose_visit)
from rebinning import coadd as CO

MAST = os.path.join(HERE, "data_v23", "MAST_v23")
STAGE = os.path.join(HERE, "pipeline_output", "chosen_visit", "stage")
REBIN = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
ITER = os.path.join(HERE, "pipeline_output", "chosen_visit", "fits")


def select(index, name, z):
    base = os.path.join(MAST, name)
    per_inst, rows_all, recs_all = {}, {}, []
    for inst in sorted(os.listdir(base)):
        d = os.path.join(base, inst)
        if not os.path.isdir(d):
            continue
        raw = read_exposures(d, inst, z)
        recs = per_file(raw)
        if not recs:
            continue
        for q in raw:
            rows_all.setdefault(q["file"], []).append(q)
        per_inst[inst] = assign_epochs(recs)
        recs_all += per_inst[inst]
    al = alignment(recs_all, rows_all)
    for inst, recs in per_inst.items():
        for r in recs:
            r["align_rms"] = al.get(r["file"], np.nan)
        per_inst[inst] = recommend(recs)[0]
    cands = choose_visit(per_inst, rows_all, index=index)
    return per_inst, cands


def stage(name, inst, keep_files, dest):
    """Copy the chosen visit's files (and their FOS partners) into a clean directory."""
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    src = os.path.join(MAST, name, inst)
    n = 0
    for fn in keep_files:
        root = fn.replace("_c0f.fits", "").replace(".fits", "")
        for p in glob.glob(os.path.join(glob.escape(src), root + "*")):
            shutil.copy2(p, dest)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, nargs="+", required=True)
    a = ap.parse_args()
    queue = {int(r["order"]): r for r in csv.DictReader(
        open(os.path.join(HERE, "pipeline_output", "work_queue.csv")))}
    dec = pd.read_csv(os.path.join(HERE, "pipeline_output", "final",
                                   "pass_decisions.csv")).set_index("index")
    _cat = pd.read_csv("/Users/gtr/Work/git/HST-Paper/Data/master_catalog_v22.csv")
    global ZCAT
    ZCAT = {}
    for _c in ("common_name", "name_mast_key", "name_pub"):
        if _c in _cat.columns and "best_z" in _cat.columns:
            for _k, _v in zip(_cat[_c], _cat["best_z"]):
                if isinstance(_k, str):
                    ZCAT.setdefault(_k.strip(), _v)
    os.makedirs(REBIN, exist_ok=True)
    for i in a.index:
        name = queue[i]["name_mast_key"]
        # pass_decisions.csv only carries the objects worked so far -- 174 of 474 --
        # so keying the redshift on it crashed the sample run at index 121.  Fall
        # back to the catalogue best_z, the same one rebin_v23.py uses.
        z = np.nan
        if i in dec.index:
            try:
                z = float(dec.loc[i, "z_adopted"])
            except Exception:
                z = np.nan
        if not np.isfinite(z):
            for k in (name, queue[i].get("common_name"), queue[i].get("name_pub")):
                if isinstance(k, str) and k.strip() in ZCAT:
                    try:
                        z = float(ZCAT[k.strip()])
                    except Exception:
                        z = np.nan
                    if np.isfinite(z):
                        break
        if not np.isfinite(z):
            print("%03d %s: no redshift" % (i, name), flush=True)
            continue
        nm = dec.loc[i, "name_pub"] if i in dec.index else name
        per_inst, cands = select(i, name, z)
        if not cands:
            print("%03d %s: no candidate visit" % (i, nm)); continue
        b = cands[0]
        keep = [r["file"] for r in per_inst[b["inst"]]
                if r["epoch"] == b["epoch"] and str(r["action"]).startswith("keep")]
        d = os.path.join(STAGE, "%03d_%s" % (i, b["inst"]))
        nfiles = stage(name, b["inst"], keep, d)
        print("\n%03d  row %s  %s  ->  %s visit %d: %d exposures (%d files), S/N ~%.0f, %s"
              % (i, dec.loc[i, "row_num"] if i in dec.index else "?", nm, b["inst"],
                 b["epoch"], len(keep), nfiles, b["snr_combined"], b["why"]), flush=True)
        try:
            CO.rebin(name, z, b["inst"], None, data_path=d, output_dir=REBIN, flat=True)
        except Exception as exc:
            print("   rebin failed: %s" % exc, flush=True); continue
        stem = None
        for p in sorted(glob.glob(os.path.join(REBIN, "*.fits")),
                        key=os.path.getmtime, reverse=True):
            stem = os.path.basename(p)[:-5]; break
        if not stem:
            print("   no rebinned output"); continue
        env = dict(os.environ, HSTICA_REBIN=REBIN, HSTICA_ITERDIR=ITER)
        r = subprocess.run([PY, os.path.join(HERE, "iterate_fit.py"), stem,
                            "--label", "chosen_visit"],
                           capture_output=True, text=True, env=env)
        recs = sorted(glob.glob(os.path.join(ITER, "*", "records", "*chosen_visit*.json")),
                      key=os.path.getmtime)
        if recs:
            j = json.load(open(recs[-1]))
            sc = j["civ_subcontinuum"]
            print("   %-28s blue=%8.1f  ew=%7.2f  usable C IV px %d  veto %s"
                  % (stem, j["result"]["civ_blue"], j["result"]["civ_ew"],
                     sc.get("n_civ_px", 0), "pass" if sc["ok"] else "REJECT %d" % sc["n_px"]),
                  flush=True)
            print("   figure: %s" % recs[-1].replace("/records/", "/").replace(".json", ".png"))
        else:
            print("   fit produced no record: %s" % (r.stderr.strip().splitlines()[-1:] or ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
