"""Generate every fit variant for a range of objects, and flag bad data assembly.

Past index 120 no prior hand fit exists, so nothing warns us when a spectrum has
been assembled badly -- a mixed-up co-add simply looks like a badly behaved
object.  This runs the checks that used to be unnecessary, per object, and then
generates the variants GTR chooses between, so an object is ready to judge the
moment it comes up rather than being half-prepared while GTR waits.

Per object it writes, into the chosen-visit fit folder:

    iter*_nomask_{low,mod,high}     no masks, the component scan
    iter*_automask[_{low,mod,high}] mask_proposer.py's masks, where it found any
    iter*_gtrmask[_{low,mod,high}]  GTR's saved GUI override, where one exists

Exact duplicates are not written twice: a variant whose override and result both
match an earlier one carries no information and only makes the list harder to
read.

and appends a row to pipeline_output/final/prepared.csv with the numbers, plus
a line to prepared_flags.csv for anything the assembly checks dislike:

    grating_disagreement   two gratings covering C IV differ by >1.5x in a band
    red_edge_collapse      the grating carrying C IV has <0.6 at 1700-1705 A
    level_outlier          one exposure's level is >5x the median of its visit
                           (the index-119 pattern: the screening may have kept
                           the wrong one)
    no_anchor              neither C III] nor Mg II is covered
    low_snr_civ            under 3 per pixel in 1500-1600 A

    python prepare_objects.py 125 199
"""
import csv, glob, json, os, re, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from exposure_decisions import (read_exposures, per_file, assign_epochs,
                                recommend, alignment, _overrides as OV)
import fit_chosen_visit as F

#: The systemic-redshift table, consulted by run() when deciding whether an
#: existing record is still current.
REDSHIFT_CSV = "/Users/gtr/Work/git/HST-Paper/ica/redshift_overrides.csv"

MAST = os.path.join(HERE, "data_v23", "MAST_v23")
FITS = os.path.join(HERE, "pipeline_output", "chosen_visit", "fits")
REBIN = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
FINAL = os.path.join(HERE, "pipeline_output", "final")
STORE = "/Users/gtr/Work/git/HST-Paper/ica/manual_fix_overrides.json"
ECH = {"E140M", "E230M", "E140H", "E230H"}

#: work_status == "civ_partial" objects are fittable but partially covered, so
#: they are held back from the default run and prepared deliberately with
#: --include-partial.  Without it they reached GTR carrying only the two
#: chosen_visit rows that fit_chosen_visit.py writes -- no component scan, no
#: mask variants -- which is how index 300 sat in prepared.csv at a blueshift of
#: -3973.2 km/s with 107 vetoed pixels and nothing to compare it against.
_argv = [a for a in sys.argv[1:] if a not in ("--include-partial", "--force")]
INCLUDE_PARTIAL = "--include-partial" in sys.argv
#: Prepare an object even though pass_decisions.csv already carries a verdict on
#: it.  That file holds decisions from EARLIER passes, some of them made in bulk
#: without individual review, and a few have since been overtaken by fixes to
#: the data.  Index 382 is the case: its row reads "0 unmasked pixels with a
#: positive error within 25 A of 1549.48", and the current tree has 140 of 140
#: usable there and 275 of 280 across the window.  Without this flag such an
#: object cannot be re-prepared at all.
FORCE = "--force" in sys.argv
lo_i, hi_i = int(_argv[0]), int(_argv[1])
queue = {int(r["order"]): r for r in csv.DictReader(
    open(os.path.join(HERE, "pipeline_output", "work_queue.csv")))}
dec = pd.read_csv(os.path.join(FINAL, "pass_decisions.csv")).set_index("index")
cat = pd.read_csv("/Users/gtr/Work/git/HST-Paper/Data/master_catalog_v22.csv")
ZC = {}
for c in ("common_name", "name_mast_key", "name_pub"):
    if c in cat.columns and "best_z" in cat.columns:
        for k, v in zip(cat[c], cat["best_z"]):
            if isinstance(k, str):
                ZC.setdefault(k.strip(), v)
store = json.load(open(STORE)) if os.path.exists(STORE) else {}


def checks(index, name, z):
    """Assembly checks. Returns (flags, notes, chosen-visit description)."""
    flags, notes = [], []
    base = os.path.join(MAST, name)
    if not os.path.isdir(base):
        return ["no_archive"], [], ""
    try:
        per_inst, cands = F.select(index, name, z)
    except Exception as exc:
        return ["select_failed:%s" % type(exc).__name__], [], ""
    if not cands:
        return ["no_candidate_visit"], [], ""
    b = cands[0]
    ov = OV().get(index, {})
    want = ov.get("epochs") or (b["epoch"],)
    kept = [r for r in per_inst[b["inst"]]
            if r["epoch"] in want and str(r["action"]).startswith("keep")]
    dg = ov.get("drop_gratings", ())
    kept = [r for r in kept if str(r["grating"]).upper() not in dg]
    desc = "%s ep%d, %d exp, %s" % (
        b["inst"], b["epoch"], len(kept),
        " ".join(sorted({str(r["grating"]).upper() for r in kept})))

    # one exposure far above its fellows -- the index-119 pattern
    lv = np.array([r["level"] for r in kept if np.isfinite(r["level"])], float)
    if lv.size >= 2:
        med = float(np.median(lv))
        if med > 0 and float(np.max(lv)) > 5 * med:
            flags.append("level_outlier")
            notes.append("one exposure %.0fx the visit median" % (np.max(lv) / med))

    # per-grating behaviour across C IV and at the red anchor
    d = os.path.join(base, b["inst"])
    try:
        rs = read_exposures(d, b["inst"], z)
    except Exception:
        rs = []
    bands = {}
    for g in sorted({str(r.get("grating")).upper() for r in rs}):
        W, FL = [], []
        for r in rs:
            if str(r.get("grating")).upper() != g:
                continue
            w = np.asarray(r["w"]); f = np.asarray(r["f"])
            ok = np.isfinite(f) & (f != 0)
            W.append(w[ok]); FL.append(f[ok])
        if not any(len(x) for x in W):
            continue
        W = np.concatenate(W); FL = np.concatenate(FL)
        o = np.argsort(W); W, FL = W[o], FL[o]
        m = (W >= 1500) & (W <= 1600)
        a = (W >= 1445) & (W <= 1465)
        if m.sum() < 20 or a.sum() < 5:
            continue
        c0 = np.median(FL[a])
        if not np.isfinite(c0) or c0 == 0:
            continue
        def band(x, y):
            s = (W >= x) & (W <= y)
            return float(np.median(FL[s]) / c0) if s.sum() > 4 else np.nan
        bands[g] = (band(1530, 1560), band(1560, 1600), band(1700, 1705))
    live = {g: v for g, v in bands.items()
            if g in {str(r["grating"]).upper() for r in kept}}
    if len(live) >= 2:
        for k in range(2):
            vals = [v[k] for v in live.values() if np.isfinite(v[k])]
            if len(vals) >= 2 and max(vals) > 1.5 * min(vals) and min(vals) > 0:
                flags.append("grating_disagreement")
                notes.append("gratings differ %.1fx in band %d: %s"
                             % (max(vals) / min(vals), k,
                                " ".join("%s=%.2f" % (g, v[k])
                                         for g, v in live.items())))
                break
    for g, v in live.items():
        if np.isfinite(v[2]) and v[2] < 0.6:
            flags.append("red_edge_collapse")
            notes.append("%s reads %.2f at 1700-1705 A" % (g, v[2]))
            break
    return flags, notes, desc


def run(stem, label, extra):
    """Fit one variant, unless an up-to-date record for that label exists.

    iterate_fit.py takes the next free iterNN number, so a second preparation
    of the same object wrote a second copy of every variant rather than
    refreshing them: re-preparing index 299 inside a 298-300 range turned its 9
    records into 16, iter10-16 being byte-equal twins of iter03-09.  The same
    thing had to be cleaned up by hand after re-rebinning 254, 260 and 286.
    Skip a label that is already on disk -- unless the spectrum is NEWER than
    the record, which is exactly the re-rebin case and must re-fit.
    """
    try:
        from object_paths import folder_for
        rec_dir = os.path.join(FITS, folder_for(stem), "records")
        have = sorted(glob.glob(os.path.join(rec_dir, "iter*_%s.json" % label)))
        # Match the label exactly: "gtrmask" must not be satisfied by
        # "gtrmask_low", whose file name also ends in a label that contains it.
        have = [h for h in have
                if re.match(r"^iter\d+_%s$" % re.escape(label),
                            os.path.basename(h)[:-5])]
        if have:
            #: A record is stale if EITHER the spectrum or the redshift override
            #: table is newer than it.  Keying on the spectrum alone was not
            #: enough: a systemic-redshift override changes the rest-frame grid
            #: at fit time without touching the FITS, so index 312's variants
            #: would have been skipped as up to date while still carrying the
            #: old frame.  Both inputs decide the fit, so both invalidate it.
            newer_than = [os.path.join(REBIN, stem + ".fits"), REDSHIFT_CSV]
            rec_t = os.path.getmtime(have[-1])
            if not any(os.path.exists(q) and os.path.getmtime(q) > rec_t
                       for q in newer_than):
                return "skipped"
    except Exception:
        pass
    env = dict(os.environ, HSTICA_REBIN=REBIN, HSTICA_ITERDIR=FITS)
    subprocess.run([PY, os.path.join(HERE, "iterate_fit.py"), stem,
                    "--label", label] + extra,
                   capture_output=True, text=True, env=env)
    return "fitted"


rows, flagrows = [], []
todo = [i for i in range(lo_i, hi_i + 1)
        if i in queue and (FORCE or i not in dec.index)
        and (queue[i].get("work_status") == "fit"
             or (INCLUDE_PARTIAL
                 and queue[i].get("work_status") == "civ_partial"))]

#: Everything in the range this script will NOT prepare, and why.  The filter
#: above takes only work_status == "fit", and used to drop the rest in silence:
#: no printed line, no flag row, nothing to say an object had been passed over.
#: Indices 298 and 300 were skipped that way inside a 297-306 batch whose log
#: read "preparing 8 objects", and only a hand count of the output found them.
#: Both are work_status "civ_partial" and both are fittable -- 83 and 114 usable
#: pixels in 1500-1600 A -- so this is exactly the case that must not vanish.
#: 24 objects carry that status; 16 had already been picked up by individual
#: review, which is to say the silence was survived rather than safe.
#: no_civ_coverage (53) and iue_only (19) are not fittable and are expected
#: here, but they are still named, because "expected" is a judgement for GTR to
#: make from a list rather than one this script should make by omission.
#: An object already carrying a pass_decisions verdict was skipped in SILENCE --
#: absent from the "preparing N objects" list and from the skip list below, so a
#: run covering it printed nothing at all about it.  That is the same defect the
#: skip reporting was added to cure for civ_partial objects, and it hid index
#: 382 inside a 381-400 batch whose log read "preparing 18 objects".  Name them.
_skipped = [(i, queue[i].get("work_status"))
            for i in range(lo_i, hi_i + 1)
            if i in queue and i not in dec.index and i not in todo]
_already = [(i, "already in pass_decisions (manmask %s) -- re-run with --force to prepare anyway"
             % dec.loc[i].get("manmask"))
            for i in range(lo_i, hi_i + 1)
            if i in queue and i in dec.index and i not in todo]
_skipped += _already
if not todo:
    print("preparing 0 objects in %s..%s" % (lo_i, hi_i), flush=True)
else:
    print("preparing %d objects: %s..%s" % (len(todo), todo[0], todo[-1]), flush=True)
if _skipped:
    print("  NOT prepared, %d object(s) in this range:" % len(_skipped), flush=True)
    for i, st in _skipped:
        note = ("fittable -- re-run with --include-partial" if st == "civ_partial"
                else "not fittable from this status")
        print("    %03d  %-28s work_status=%-16s %s"
              % (i, (queue[i].get("name_pub") or "")[:28], st, note), flush=True)
        flagrows.append(dict(index=i, name_pub=queue[i].get("name_pub"),
                             flag="skipped_%s" % st, note=note))

for n, i in enumerate(todo, 1):
    q = queue[i]
    name = q["name_mast_key"]
    z = ZC.get(name, np.nan)
    if not np.isfinite(z):
        for k in (q.get("common_name"), q.get("name_pub")):
            if isinstance(k, str) and k.strip() in ZC:
                z = ZC[k.strip()]; break
    if not np.isfinite(z):
        flagrows.append(dict(index=i, name_pub=q.get("name_pub"), flag="no_redshift", note=""))
        continue
    z = float(z)
    fl, notes, desc = checks(i, name, z)

    folds = glob.glob(os.path.join(FITS, "%03d_*" % i))
    if not folds:
        flagrows.append(dict(index=i, name_pub=q.get("name_pub"),
                             flag="no_chosen_visit_fit", note=desc))
        continue
    fold = folds[0]
    recs = glob.glob(os.path.join(fold, "records", "*.json"))
    if not recs:
        flagrows.append(dict(index=i, name_pub=q.get("name_pub"),
                             flag="no_records", note=desc))
        continue
    stem = json.load(open(recs[0]))["name"]

    # The automatic mask proposals, which is what catches a cosmic-ray spike or
    # an absorption trough that nobody has looked at yet.  Omitted from the
    # first version of this script, and index 129 showed why that was wrong: a
    # 59-sigma spike at 2735-2740 A, unflagged by the pipeline's bad-pixel mask,
    # sat inside the continuum-fitting range and dragged the morph continuum up,
    # biasing EVERY residual window negative by 1 to 3 sigma and moving the
    # blueshift from -148 to -632.  The proposer finds it unaided.  GTR had to
    # point it out: "why would you not have masked the spike at 2738A without me
    # having to tell you?"
    # Labels say what the MASKS are, not how the variant was produced: "probe"
    # and "proposed" were too alike to tell apart in a list, and GTR had to ask
    # which was which.  nomask / automask / gtrmask, each with its component
    # scan.
    subprocess.run([PY, os.path.join(HERE, "mask_proposer.py"), "--names", stem],
                   capture_output=True, text=True,
                   env=dict(os.environ, HSTICA_REBIN=REBIN))
    # Only fit the automatic masks if the proposer actually found something --
    # on most objects it finds nothing and the variants came out byte-identical
    # to the unmasked ones, which is clutter rather than information.
    n_prop = 0
    try:
        _pj = json.load(open(os.path.join(HERE, "pipeline_output",
                                          "mask_proposals.json"))).get(stem, {})
        n_prop = len(_pj.get("mask_ranges") or _pj.get("ranges") or [])
    except Exception:
        pass
    if n_prop:
        run(stem, "automask", ["--from-proposals"])
        for c in ("low", "mod", "high"):
            run(stem, "automask_%s" % c, ["--from-proposals", "--comps", c])
    for c in ("low", "mod", "high"):
        run(stem, "nomask_%s" % c, ["--comps", c])
    if stem in store:
        run(stem, "gtrmask", ["--from-store", stem])
        for c in ("low", "mod", "high"):
            run(stem, "gtrmask_%s" % c, ["--from-store", stem, "--comps", c])

    # The [O III] systemic-redshift figure.  check_oiii.py --plot writes
    # OIII_redshift.png into the object's folder and GTR reads it for every
    # object, but this script never called it: the figure appeared only where it
    # had been run by hand afterwards, so the 401-409 block reached GTR without
    # one -- "There's no OIII for the next set again."  The same shape of defect
    # as the missing Mg II inset at index 382, and the same fix: make the
    # diagnostic part of preparing an object rather than a separate step someone
    # has to remember.  Flagged when no figure appears, so an object without one
    # is visible in prepared_flags.csv instead of silently lacking it.
    _oiii = subprocess.run([PY, os.path.join(HERE, "check_oiii.py"), "--plot", str(i)],
                           capture_output=True, text=True,
                           env=dict(os.environ, HSTICA_REBIN=REBIN))
    if not os.path.exists(os.path.join(fold, "OIII_redshift.png")):
        # Skip check_oiii.py's own "figure -> not drawn" line: the useful note is
        # the reason above it, which unavailable_reason() now spells out.
        _out = [ln.strip() for ln in (_oiii.stdout or "").splitlines()
                if ln.strip() and not ln.strip().startswith("figure ->")]
        _why = (_out[-1] if _out else "check_oiii.py wrote no figure")
        #: Two different things, and only one of them is a failure.  Above
        #: z ~ 0.84 [O III] 5007 sits past the red end of an SDSS spectrum, so
        #: every object from index 424 onward -- 51 of them -- cannot have the
        #: figure at all.  Flagging that the same way as a retrieval or fitting
        #: failure would bury the real cases under physics.
        fl.append("oiii_out_of_range" if "outside the SDSS spectrum" in _why
                  or "blueward of the SDSS coverage" in _why else "no_oiii_figure")
        while len(notes) < len(fl) - 1:
            notes.append("")
        notes.append(_why[:200])

    best = None
    for p in sorted(glob.glob(os.path.join(fold, "records", "*.json"))):
        j = json.load(open(p)); r = j["result"]; sc = j["civ_subcontinuum"]
        o = j.get("override", {}) or {}
        mo = j.get("monochromatic", {}) or {}
        meas = "".join("Y" if (isinstance(v, dict) and v.get("measured")) else "-"
                       for k, v in sorted(mo.items()) if k != "_calibration")
        rows.append(dict(index=i, row_num=q.get("row_num"), name_pub=q.get("name_pub"),
                         chosen=desc, iteration=os.path.basename(p)[:-5],
                         comps=str(o.get("comps_use") or "auto"),
                         n_mask_ranges=len(o.get("mask_ranges") or []),
                         n_mask_px=len(o.get("mask_pixels") or []),
                         civ_blue=round(r["civ_blue"], 1), civ_ew=round(r["civ_ew"], 2),
                         veto="pass" if sc["ok"] else "REJECT %d" % sc["n_px"],
                         n_civ_px=sc.get("n_civ_px"), lum=meas,
                         windows=j.get("windows", "")))
        best = j
    if best is not None:
        w = str(best.get("windows", ""))
        if "CIII]: --" in w and "MgII: --" in w:
            fl.append("no_anchor")
    blues = [r["civ_blue"] for r in rows if r["index"] == i]
    spread = (max(blues) - min(blues)) if blues else 0.0
    for f, nt in zip(fl, notes + [""] * len(fl)):
        flagrows.append(dict(index=i, name_pub=q.get("name_pub"), flag=f, note=nt))
    print("%3d/%d  %03d  %-28s %-34s %2d variants, blue spread %6.0f km/s  %s"
          % (n, len(todo), i, str(q.get("name_pub"))[:28], desc,
             len(blues), spread, ",".join(sorted(set(fl))) or "-"), flush=True)

#: MERGE rather than overwrite.  These tables accumulate across runs -- the
#: docstring above says "appends" -- but the write was an unconditional
#: to_csv, so running the script on a range that produced nothing (index 216,
#: whose chosen visit had no rebin product yet) emptied the file for every
#: object already prepared.  Rows are keyed on (index, iteration) for
#: prepared.csv and (index, flag) for the flags, with the new run winning so a
#: re-preparation refreshes its own numbers.
def _merge(new_rows, path, keys):
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(path) and os.path.getsize(path) > 1:
        try:
            old_df = pd.read_csv(path)
        except Exception:
            old_df = None
        if old_df is not None and len(old_df):
            if new_df.empty:
                return old_df
            keep = [c for c in keys if c in old_df.columns and c in new_df.columns]
            if keep:
                merged = pd.concat([old_df, new_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=keep, keep="last")
                return merged.sort_values(keys[0], kind="stable")
    return new_df

_merge(rows, os.path.join(FINAL, "prepared.csv"), ["index", "iteration"]) \
    .to_csv(os.path.join(FINAL, "prepared.csv"), index=False)
_merge(flagrows, os.path.join(FINAL, "prepared_flags.csv"), ["index", "flag"]) \
    .to_csv(os.path.join(FINAL, "prepared_flags.csv"), index=False)
print("\n%d variant rows -> final/prepared.csv" % len(rows))
print("%d flags -> final/prepared_flags.csv" % len(flagrows))
