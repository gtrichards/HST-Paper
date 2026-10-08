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
                                alignment, choose_visit, _overrides as _OV)
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
        # An override may name several epochs, where a second exists only to fill
        # a wavelength hole the first cannot cover (Mrk 841, index 48).
        _want = _OV().get(i, {}).get("epochs") or (b["epoch"],)
        _force = _OV().get(i, {}).get("keep_files", ())
        kept = [r for r in per_inst[b["inst"]]
                if r["epoch"] in _want
                and (str(r["action"]).startswith("keep")
                     or any(r["file"].startswith(x) for x in _force))]
        if _force:
            got = [r["file"] for r in kept
                   if not str(r["action"]).startswith("keep")]
            print("   override: reinstating %d exposure(s) set aside for review: %s"
                  % (len(got), ", ".join(sorted(got)) or "none matched"), flush=True)
        if len(_want) > 1:
            print("   override: combining epochs %s"
                  % " + ".join(str(x) for x in sorted(_want)), flush=True)
        # A hand override may exclude gratings within the chosen visit: mixing
        # resolutions that differ by an order of magnitude puts a flux step
        # across the line rather than adding signal (Mrk 1044, G140L + G140M).
        drop = _OV().get(i, {}).get("drop_gratings", ())
        if drop:
            before = len(kept)
            kept = [r for r in kept if str(r["grating"]).upper() not in drop]
            print("   override: dropping grating(s) %s -- %d of %d exposures excluded"
                  % (" ".join(drop), before - len(kept), before), flush=True)
        # ...and individual exposures, where no instrument/epoch/grating cut
        # separates the good from the bad (Mrk 205, index 81).
        dropf = _OV().get(i, {}).get("drop_files", ())
        if dropf:
            before = len(kept)
            kept = [r for r in kept
                    if not any(r["file"].startswith(x) for x in dropf)]
            print("   override: dropping exposure(s) %s -- %d of %d excluded"
                  % (" ".join(dropf), before - len(kept), before), flush=True)
        keep = [r["file"] for r in kept]
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
        # Where a donor instrument is named, splice it on HERE, immediately after
        # the chosen visit is rebinned, and fit the spliced product instead.  The
        # splice was a second manual step for one day and that is one day too
        # long: re-running this script regenerated the host and left the spliced
        # file beside it, stale, with nothing to say so.  One entry point means
        # the two cannot drift apart.
        if _OV().get(i, {}).get("donor_inst"):
            try:
                import splice_spectra
                sp = splice_spectra.build(i)
            except SystemExit as exc:
                print("   splice failed: %s" % exc); sp = None
            if sp:
                stem = os.path.basename(sp)[:-5]
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
    # The interactive fitter orders its object list from index_order.csv in the
    # rebin directory.  Nothing wrote that file, so an override that changed an
    # object's instrument left the old stem mapped and the new one unlisted:
    # H1821+643 appeared at the end of the fitter with no index number at all.
    # Refreshing it here means it cannot drift from what is actually on disk.
    try:
        import write_index_order
        write_index_order.write(quiet=True)
    except Exception as exc:
        print("   could not refresh index_order.csv: %s" % exc, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
