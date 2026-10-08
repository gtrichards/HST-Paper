"""Generate every fit variant for a range of objects, and flag bad data assembly.

Past index 120 no prior hand fit exists, so nothing warns us when a spectrum has
been assembled badly -- a mixed-up co-add simply looks like a badly behaved
object.  This runs the checks that used to be unnecessary, per object, and then
generates the variants GTR chooses between, so an object is ready to judge the
moment it comes up rather than being half-prepared while GTR waits.

Per object it writes, into the chosen-visit fit folder:

    iter*_proposed[_{low,mod,high}] mask_proposer.py's automatic proposals
    iter*_probe_{low,mod,high}      the unmasked component scan
    iter*_gtr_hand                  GTR's saved GUI override replayed, if any
    iter*_gtrmask_{low,mod,high}    that override across the component sets

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
import csv, glob, json, os, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from exposure_decisions import (read_exposures, per_file, assign_epochs,
                                recommend, alignment, _overrides as OV)
import fit_chosen_visit as F

MAST = os.path.join(HERE, "data_v23", "MAST_v23")
FITS = os.path.join(HERE, "pipeline_output", "chosen_visit", "fits")
REBIN = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
FINAL = os.path.join(HERE, "pipeline_output", "final")
STORE = "/Users/gtr/Work/git/HST-Paper/ica/manual_fix_overrides.json"
ECH = {"E140M", "E230M", "E140H", "E230H"}

lo_i, hi_i = int(sys.argv[1]), int(sys.argv[2])
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
    env = dict(os.environ, HSTICA_REBIN=REBIN, HSTICA_ITERDIR=FITS)
    subprocess.run([PY, os.path.join(HERE, "iterate_fit.py"), stem,
                    "--label", label] + extra,
                   capture_output=True, text=True, env=env)


rows, flagrows = [], []
todo = [i for i in range(lo_i, hi_i + 1)
        if i in queue and i not in dec.index
        and queue[i].get("work_status") == "fit"]
print("preparing %d objects: %s..%s" % (len(todo), todo[0], todo[-1]), flush=True)

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
    subprocess.run([PY, os.path.join(HERE, "mask_proposer.py"), "--names", stem],
                   capture_output=True, text=True,
                   env=dict(os.environ, HSTICA_REBIN=REBIN))
    run(stem, "proposed", ["--from-proposals"])
    for c in ("low", "mod", "high"):
        run(stem, "proposed_%s" % c, ["--from-proposals", "--comps", c])
    for c in ("low", "mod", "high"):
        run(stem, "probe_%s" % c, ["--comps", c])
    if stem in store:
        run(stem, "gtr_hand", ["--from-store", stem])
        for c in ("low", "mod", "high"):
            run(stem, "gtrmask_%s" % c, ["--from-store", stem, "--comps", c])

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

pd.DataFrame(rows).to_csv(os.path.join(FINAL, "prepared.csv"), index=False)
pd.DataFrame(flagrows).to_csv(os.path.join(FINAL, "prepared_flags.csv"), index=False)
print("\n%d variant rows -> final/prepared.csv" % len(rows))
print("%d flags -> final/prepared_flags.csv" % len(flagrows))
