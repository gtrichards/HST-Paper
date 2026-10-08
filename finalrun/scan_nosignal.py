"""Where else has one over-bright exposure condemned its good companions?

The "no signal" test in exposure_decisions drops an exposure whose S/N is below
SNR_DEAD and whose level sits more than LEVEL_DEAD below best[grating] -- the
median level across that grating's epoch groups.  The guard is one-directional:
it was written so a half-dead visit cannot define its own normal, but nothing
stops a single over-bright exposure from defining the normal instead.  On index
119 that inverted the visit, keeping the one corrupt exposure and dropping the
45-group good one.

For every object this reports, per grating, whether best[grating] is set by an
exposure that is a far outlier ABOVE its fellows, and whether any exposure was
dropped "no signal" that would survive if that outlier were excluded.

    python scan_nosignal.py            # whole queue
    python scan_nosignal.py 119 81     # named indices
"""
import csv, os, sys
import numpy as np
import pandas as pd

HERE = "/Users/gtr/Work/projects/hstica/finalrun"
sys.path.insert(0, HERE); sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
import exposure_decisions as E
from exposure_decisions import read_exposures, per_file, assign_epochs

MAST = os.path.join(HERE, "data_v23", "MAST_v23")
OUT = os.path.join(HERE, "pipeline_output", "final", "nosignal_scan.csv")

queue = {int(r["order"]): r for r in csv.DictReader(
    open(os.path.join(HERE, "pipeline_output", "work_queue.csv")))}
dec = pd.read_csv(os.path.join(HERE, "pipeline_output", "final",
                               "pass_decisions.csv")).set_index("index")
cat = pd.read_csv("/Users/gtr/Work/git/HST-Paper/Data/master_catalog_v22.csv")
ZCAT = {}
for c in ("common_name", "name_mast_key", "name_pub"):
    if c in cat.columns and "best_z" in cat.columns:
        for k, v in zip(cat[c], cat["best_z"]):
            if isinstance(k, str):
                ZCAT.setdefault(k.strip(), v)

want = [int(a) for a in sys.argv[1:]] or sorted(queue)
rows = []
for i in want:
    q = queue.get(i)
    if not q:
        continue
    name = q["name_mast_key"]
    z = np.nan
    if i in dec.index:
        try:
            z = float(dec.loc[i, "z_adopted"])
        except Exception:
            z = np.nan
    if not np.isfinite(z):
        for k in (name, q.get("common_name"), q.get("name_pub")):
            if isinstance(k, str) and k.strip() in ZCAT:
                try:
                    z = float(ZCAT[k.strip()])
                except Exception:
                    z = np.nan
                if np.isfinite(z):
                    break
    if not np.isfinite(z):
        continue
    base = os.path.join(MAST, name)
    if not os.path.isdir(base):
        continue
    for inst in sorted(os.listdir(base)):
        d = os.path.join(base, inst)
        if not os.path.isdir(d):
            continue
        try:
            recs = assign_epochs(per_file(read_exposures(d, inst, z)))
        except Exception as exc:
            print("%03d %s/%s: %s" % (i, name, inst, exc), flush=True)
            continue
        if not recs:
            continue
        # reproduce best[grating] exactly as recommend() computes it
        by_ep = {}
        for r in recs:
            by_ep.setdefault((r["epoch"], r["grating"]), []).append(r)
        ep_level = {}
        for key, group in by_ep.items():
            lv = np.array([g["level"] for g in group], float)
            ep_level[key] = (np.nanmedian(lv[np.isfinite(lv)])
                             if np.isfinite(lv).any() else np.nan)
        best = {}
        for (ep, gr), lev in ep_level.items():
            if np.isfinite(lev) and (gr not in best or lev > best[gr]):
                best[gr] = lev
        by_grating = {}
        for r in recs:
            by_grating.setdefault(r["grating"], []).append(r)

        for gr, group in by_grating.items():
            ref = best.get(gr, np.nan)
            if not (np.isfinite(ref) and ref > 0):
                continue
            lv = np.array([g["level"] for g in group], float)
            lv = lv[np.isfinite(lv)]
            if len(lv) < 2:
                continue
            # who is condemned as it stands?
            dropped = [g for g in group
                       if np.isfinite(g["snr"]) and g["snr"] < E.SNR_DEAD
                       and np.isfinite(g["level"])
                       and abs(g["level"]) < ref / E.LEVEL_DEAD]
            if not dropped:
                continue
            # is the reference set by an exposure far ABOVE the rest?
            med_all = float(np.median(lv))
            top = float(np.max(lv))
            # recompute best with the single brightest exposure removed
            lv2 = np.sort(lv)[:-1]
            alt_by_ep = {}
            hi = max(group, key=lambda g: (g["level"] if np.isfinite(g["level"])
                                           else -np.inf))
            for key, grp in by_ep.items():
                if key[1] != gr:
                    continue
                vals = [g["level"] for g in grp
                        if g is not hi and np.isfinite(g["level"])]
                if vals:
                    alt_by_ep[key] = float(np.median(vals))
            alt_ref = max(alt_by_ep.values()) if alt_by_ep else np.nan
            if not (np.isfinite(alt_ref) and alt_ref > 0):
                continue
            # would any of the dropped survive against the alternative reference?
            saved = [g for g in dropped
                     if abs(g["level"]) >= alt_ref / E.LEVEL_DEAD]
            if not saved:
                continue
            rows.append(dict(
                index=i, row_num=q.get("row_num", ""),
                name_pub=(dec.loc[i, "name_pub"] if i in dec.index else name),
                name_mast_key=name, inst=inst, grating=gr,
                n_exposures=len(group),
                outlier_file=hi["file"], outlier_level="%.4g" % hi["level"],
                outlier_snr="%.1f" % hi["snr"],
                median_level="%.4g" % med_all,
                ratio_outlier_to_median="%.1f" % (top / med_all
                                                  if med_all else np.nan),
                best_used="%.4g" % ref, best_without_outlier="%.4g" % alt_ref,
                n_dropped=len(dropped), n_would_be_saved=len(saved),
                saved_files=" ".join(sorted({g["file"] for g in saved})),
                manmask=(dec.loc[i, "manmask"] if i in dec.index else ""),
            ))
            print("%03d %-26s %-5s %-6s outlier %s at %.3gx the median; "
                  "%d dropped, %d would survive"
                  % (i, str(rows[-1]["name_pub"])[:26], inst, gr,
                     hi["file"], top / med_all if med_all else np.nan,
                     len(dropped), len(saved)), flush=True)

if rows:
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("\n%d (object, instrument, grating) cases -> %s" % (len(rows), OUT))
else:
    print("\nno cases found")
