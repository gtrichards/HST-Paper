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
SDSS_DIR = os.path.join(HERE, "data_v23", "SDSS_spec", "lite")
SDSS_TABLE = os.path.join(HERE, "pipeline_output", "final", "sdss_spectra.csv")

_SDSS_CACHE = {}


#: Whether to splice SDSS onto the chosen-visit co-add.  OFF, by GTR's decision
#: of 2026-10-14 after the splice was tested on index 286: it moved C IV by 11
#: quanta and bought only ~300 fittable pixels, the ICA components ending at
#: 2999.9 A.  The cost is larger than that test showed.  Splicing extends the
#: co-add to ~6600 A rest, which changes the continuum fit over the whole range
#: and therefore which pixels maskNAL rejects INSIDE C IV: on index 296 the
#: usable C IV pixel count fell from 274 to 148 and the blueshift moved from
#: +887.0 to +955.9 km/s, with the C IV data themselves untouched.  Enable per
#: run with --splice-sdss only for a deliberate test, never for an adopted fit.
SPLICE_SDSS = False


def sdss_for(index):
    """The SDSS spectrum to splice onto this object's HST data, or None.

    Returns (None, None) unless SPLICE_SDSS is set; see that flag.

    coadd.rebin takes fn_sdss as its fourth argument and has always been able
    to splice SDSS onto the HST co-add, but this driver passed None
    unconditionally, so the branch never ran. sdss_spectra.csv records which
    objects have a spectrum and how far blueward it reaches; only those whose
    coverage overlaps the ICA range are worth splicing, which is what
    sdss_flag = 2 means and begins at index 286 (z ~ 0.359, where SDSS first
    reaches Mg II).

    Returns (fn_sdss, sdss_spec_dir) or (None, None).
    """
    if not SPLICE_SDSS:
        return None, None
    if not _SDSS_CACHE:
        try:
            t = pd.read_csv(SDSS_TABLE)
        except Exception:
            _SDSS_CACHE["_"] = None
            return None, None
        for _, r in t.iterrows():
            try:
                k = int(r["index"])
            except Exception:
                continue
            if not bool(r.get("covers_mgii", False)):
                continue
            fn = str(r.get("fn_sdss") or "").strip()
            if not fn or not os.path.exists(os.path.join(SDSS_DIR, fn)):
                continue
            _SDSS_CACHE[k] = fn
    fn = _SDSS_CACHE.get(int(index))
    return (fn, SDSS_DIR) if fn else (None, None)


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
    ap.add_argument("--splice-sdss", action="store_true",
                    help="splice SDSS onto the chosen-visit co-add (off by default; "
                         "changes the continuum fit and the C IV masking -- test only)")
    a = ap.parse_args()
    global SPLICE_SDSS
    SPLICE_SDSS = bool(a.splice_sdss)
    if SPLICE_SDSS:
        print("   SDSS SPLICE ENABLED -- not for an adopted fit", flush=True)
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
            _fn_sdss, _sdss_dir = sdss_for(i)
            if _fn_sdss:
                print("   splicing SDSS spectrum %s" % _fn_sdss, flush=True)
            CO.rebin(name, z, b["inst"], _fn_sdss, data_path=d, output_dir=REBIN,
                     sdss_spec_dir=_sdss_dir, flat=True)
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
        # Glob THIS object's folder, not the whole tree.  Globbing ITER/* and
        # taking the newest record means that when an object's fit produces no
        # record at all, recs[-1] is the PREVIOUS object's -- so the log reports
        # one object's blueshift, EW and figure path under another object's
        # index.  Index 374 (QSO J0943+0531, a visit at S/N 0.0 that produced
        # nothing) was reported with index 372's numbers and 372's figure, and
        # a variant prepared from that log would have carried 372's measurement
        # against 374's row.  The index prefix is the stable key.
        recs = sorted(glob.glob(os.path.join(ITER, "%03d_*" % i, "records",
                                             "*chosen_visit*.json")),
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

    # chosen_visit_summary.csv maps spectrum stems to object indices, and three
    # readers resolve through it: the SDSS overlay on the diagnostic figures
    # (iterate_fit._sdss_file_for_stem), the Mg II inset in the interactive
    # fitter (manual_fix_gui._sdss_overlay) and check_oiii's redshift-override
    # lookup.  Each returns None SILENTLY for a stem it cannot find.  Nothing
    # wrote that file -- it was produced once, on 2026-09-27 -- so every object
    # that entered the tree afterwards lost its overlay and its inset with no
    # error and nothing in the log.  Fifteen had, four of them already recorded.
    # GTR found it on index 382: "There is no MgII inset for the interactive
    # fitter.  What's the problem here?  Why wasn't this block prepared exactly
    # like the last one?"  It had been; the file was simply stale.  This is the
    # step that creates new stems, so this is where the file is kept current.
    try:
        import reconcile_chosen_visit_summary as _rec
        _added = _rec.reconcile(write=True, quiet=True)
        if _added:
            print("   chosen_visit_summary.csv: added %d stem(s)" % _added, flush=True)
    except Exception as exc:
        print("   could not refresh chosen_visit_summary.csv: %s" % exc, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
