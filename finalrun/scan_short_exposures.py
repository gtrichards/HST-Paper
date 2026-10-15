"""Find very short exposures that survived into an object's chosen visit.

Written after the same defect was found twice by eye.  On index 201 six G430L
frames of 1.5 s each, and on index 303 two more, were acquisition or peak-up
images whose flux scale is wrong by orders of magnitude -- a factor of 90 and
about 320 at their respective grating joins.  Co-adding them does not merely
add noise.  They carry the SMALLEST fractional errors in their visit (index
303: median S/N 39.5 against 8.2-10.9 for the real science exposures), which is
impossible for a 1.5 s frame and gives them the LARGEST weight in a
variance-weighted co-add, so past the join the co-add is effectively those
files alone.  On 303 that put log L2500 three dex above log L1450, tilted the
continuum enough to swing the blueshift over 2143 km/s on masking choice
alone, and failed the sub-continuum veto on every prepared variant.

Both were caught because GTR looked at a figure and said the red end looked
like a different object.  That is not a process that scales to 474 objects, so
this names every case directly.

An exposure is reported when it is SHORT (below --max-exptime, default 10 s)
and it is in the file list the chosen visit actually co-adds -- after epochs,
keep_files, drop_gratings and drop_files from spectrum_overrides.csv have been
applied.  Exposures already excluded by an override are reported separately as
confirmation that the fix is in place, not as outstanding problems.

    python scan_short_exposures.py
    python scan_short_exposures.py --max-exptime 30 --csv out.csv
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fit_chosen_visit as F
from exposure_decisions import _overrides as _OV


def kept_files(index, name, z):
    """The files the chosen visit co-adds, and the ones an override removed."""
    per_inst, cands = F.select(index, name, z)
    if not cands:
        return None, None, None
    b = cands[0]
    ov = _OV().get(index, {}) or {}
    want = ov.get("epochs") or (b["epoch"],)
    force = ov.get("keep_files", ())
    kept = [r for r in per_inst[b["inst"]]
            if r["epoch"] in want
            and (str(r["action"]).startswith("keep")
                 or any(r["file"].startswith(x) for x in force))]
    before = list(kept)
    drop = ov.get("drop_gratings", ())
    if drop:
        kept = [r for r in kept if str(r["grating"]).upper() not in drop]
    dropf = ov.get("drop_files", ())
    if dropf:
        kept = [r for r in kept if not any(r["file"].startswith(x) for x in dropf)]
    removed = [r for r in before if r not in kept]
    return b, kept, removed


def exposure_facts(path):
    """exptime, median flux and median S/N for one exposure, or None."""
    try:
        with fits.open(path) as h:
            h0, d = h[0].header, h[1].data
            w = np.ravel(d["WAVELENGTH"]); f = np.ravel(d["FLUX"])
            en = "ERROR" if "ERROR" in d.names else (
                "NET_ERROR" if "NET_ERROR" in d.names else None)
            e = np.ravel(d[en]) if en else np.full_like(f, np.nan)
            g = np.isfinite(f) & (f != 0)
            ge = g & np.isfinite(e) & (e > 0)
            return dict(
                grating=h0.get("OPT_ELEM"),
                exptime=float(h0.get("TEXPTIME", h[1].header.get("EXPTIME", np.nan))),
                medflux=float(np.nanmedian(f[g])) if g.any() else np.nan,
                medsn=float(np.nanmedian((f / e)[ge])) if ge.any() else np.nan,
                wmin=float(np.nanmin(w[g])) if g.any() else np.nan,
                wmax=float(np.nanmax(w[g])) if g.any() else np.nan)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-exptime", type=float, default=10.0,
                    help="report exposures shorter than this, in seconds (default 10)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--index", type=int, nargs="*", default=None)
    a = ap.parse_args()

    queue = {int(r["order"]): r for r in csv.DictReader(
        open(os.path.join(HERE, "pipeline_output", "work_queue.csv")))}
    cat = pd.read_csv("/Users/gtr/Work/git/HST-Paper/Data/master_catalog_v22.csv")
    zcat = {}
    for c in ("common_name", "name_mast_key", "name_pub"):
        if c in cat.columns and "best_z" in cat.columns:
            for k, v in zip(cat[c], cat["best_z"]):
                if isinstance(k, str):
                    zcat.setdefault(k.strip(), v)

    todo = sorted(queue) if a.index is None else a.index
    live, fixed, rows = [], [], []
    for i in todo:
        q = queue[i]
        if str(q.get("work_status")) not in ("fit", "civ_partial"):
            continue
        name = q["name_mast_key"]
        z = np.nan
        for k in (name, q.get("common_name"), q.get("name_pub")):
            if isinstance(k, str) and k.strip() in zcat:
                try:
                    z = float(zcat[k.strip()])
                except Exception:
                    z = np.nan
                if np.isfinite(z):
                    break
        if not np.isfinite(z):
            continue
        try:
            b, kept, removed = kept_files(i, name, z)
        except Exception:
            continue
        if not kept:
            continue
        src = os.path.join(F.MAST, name, b["inst"])
        # The visit's own scale, from the exposures that are NOT short: a short
        # frame is only a problem when it disagrees with its companions.
        facts = {}
        for r in kept + removed:
            p = os.path.join(src, r["file"])
            hits = glob.glob(p + "*") or glob.glob(os.path.join(glob.escape(src),
                                                                r["file"] + "*"))
            if not hits:
                continue
            fx = exposure_facts(hits[0])
            if fx:
                facts[r["file"]] = fx
        # Compare a short exposure against its OWN GRATING's companions, not
        # against the visit as a whole.  A visit-wide median mixes gratings whose
        # flux scales differ by an order of magnitude on their own -- index 37
        # runs G140L at 4e-14 and G750L at 2.3e-15 -- so a visit-wide ratio
        # reports a nonsense number for any multi-grating visit.  Falls back to
        # the visit median only when a grating has no long exposure to compare
        # against, which is the 201/303 case.
        def _ref_for(grating):
            same = [v["medflux"] for v in facts.values()
                    if np.isfinite(v["exptime"]) and v["exptime"] >= a.max_exptime
                    and np.isfinite(v["medflux"]) and v["grating"] == grating]
            if same:
                return float(np.nanmedian(same)), "same grating"
            other = [v["medflux"] for v in facts.values()
                     if np.isfinite(v["exptime"]) and v["exptime"] >= a.max_exptime
                     and np.isfinite(v["medflux"])]
            return (float(np.nanmedian(other)) if other else np.nan), "visit median"
        for r, where in [(x, "IN CO-ADD") for x in kept] + \
                        [(x, "already dropped") for x in removed]:
            fx = facts.get(r["file"])
            if not fx or not np.isfinite(fx["exptime"]):
                continue
            if fx["exptime"] >= a.max_exptime:
                continue
            ref, refkind = _ref_for(fx["grating"])
            ratio = fx["medflux"] / ref if (np.isfinite(ref) and ref) else np.nan
            rec = dict(index=i, name_pub=q.get("name_pub"), inst=b["inst"],
                       epoch=b["epoch"], file=r["file"], grating=fx["grating"],
                       exptime=fx["exptime"], medflux=fx["medflux"],
                       medsn=fx["medsn"], ref_flux=ref, ref_kind=refkind,
                       flux_ratio=ratio, status=where)
            rows.append(rec)
            (live if where == "IN CO-ADD" else fixed).append(rec)

    def show(title, rs):
        print("\n%s: %d exposure(s)" % (title, len(rs)))
        if not rs:
            return
        print("  %4s %-26s %-5s %-7s %-20s %8s %10s %8s %9s"
              % ("idx", "name", "inst", "grating", "file", "exptime",
                 "flux/ref", "med S/N", "medflux"))
        for r in sorted(rs, key=lambda x: (-abs(np.log10(abs(x["flux_ratio"]))
                                           if np.isfinite(x["flux_ratio"]) and x["flux_ratio"] else 0),
                                           x["index"])):
            print("  %4d %-26s %-5s %-7s %-20s %8.1f %10s %8.1f %9.3g"
                  % (r["index"], str(r["name_pub"])[:26], r["inst"],
                     str(r["grating"]), r["file"][:20], r["exptime"],
                     ("%.1fx" % r["flux_ratio"]) if np.isfinite(r["flux_ratio"]) else "--",
                     r["medsn"], r["medflux"]))

    print("scanned for exposures shorter than %.0f s" % a.max_exptime)
    show("STILL IN A CHOSEN VISIT'S CO-ADD", live)
    show("already excluded by an override", fixed)
    if a.csv and rows:
        pd.DataFrame(rows).to_csv(a.csv, index=False)
        print("\nwrote %d rows -> %s" % (len(rows), a.csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
