"""
Record the redshift check for every object on the definitive pass.

Redshifts are checked as objects are worked rather than in a separate campaign,
so the check needs somewhere to live that travels with the object. This writes
final/redshift_check.csv, one row per object in final/decisions.csv:

    z_catalog     the xlsx value the pipeline ran with
    z_ned         NED's preferred redshift for the object at that position
    dv_ned_kms    velocity difference, +ve when NED is higher
    z_simbad      SIMBAD's, with its bibcode and quality grade
    dv_simbad_kms
    z_adopted     the redshift of the adopted fit (the BEST_ record)
    anchor_check  the adopted fit's own C III] / Mg II cross-correlation offsets
    flag          "ok" if every available external value is within one grid
                  quantum (69 km/s) of the catalogue, else "CHECK"

Neither NED nor SIMBAD says which line a redshift came from, so this is a
cross-check, not a source of narrow-line redshifts; the flag marks objects worth
a look, it does not change anything. Objects are found by cone search on the
coordinate designation, nearest AGN-typed match, which is the only reliable route
given how these objects are named. Results are cached so re-runs are free.
"""

import csv
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")
ITERDIR = os.path.join(OUTDIR, "fit_iterations")
QUEUE = os.path.join(OUTDIR, "work_queue.csv")
FINAL = os.path.join(OUTDIR, "final")
DECISIONS = os.path.join(FINAL, "decisions.csv")
OUT = os.path.join(FINAL, "redshift_check.csv")
CACHE = os.path.join(FINAL, "redshift_check_cache.json")

C = 299792.458
QUANTUM = 69.0


def dv(z_new, z_ref):
    return C * (float(z_new) - float(z_ref)) / (1.0 + float(z_ref))


def main():
    from query_redshifts import desig_to_coord, query_ned, query_simbad

    from object_paths import folder_for
    queue = {r["name_mast_key"]: r for r in csv.DictReader(open(QUEUE))}
    by_folder = {}
    for r in queue.values():
        by_folder[folder_for("%s_COMBINED" % r["name_mast_key"])] = r
        by_folder[folder_for("%s_SPLICE" % r["name_mast_key"])] = r
        for st in r["stems"].split(";"):
            if st:
                by_folder[folder_for(st)] = r

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    rows = []
    for dec in csv.DictReader(open(DECISIONS)):
        q = by_folder.get(dec["folder"])
        if q is None:
            print("no queue entry for %s" % dec["folder"])
            continue
        desig = q["designation"]
        if desig not in cache:
            coord = desig_to_coord(desig)
            ned = query_ned(coord)
            time.sleep(0.5)
            sim = query_simbad(coord)
            time.sleep(0.5)
            cache[desig] = {"ned": ned, "sim": sim}
            json.dump(cache, open(CACHE, "w"), indent=1)
        ned, sim = cache[desig]["ned"], cache[desig]["sim"]

        rec_dir = os.path.join(ITERDIR, dec["folder"], "records")
        chosen = (glob.glob(os.path.join(rec_dir, "BEST_*.json"))
                  or glob.glob(os.path.join(rec_dir, "REJECT_ALL_*.json")))
        adopted = json.load(open(chosen[0])) if chosen else None
        z_cat = float(q["z"]) if q["z"] not in ("", None) else None
        zn, zs = ned.get("ned_z"), sim.get("simbad_z")
        dvn = round(dv(zn, z_cat), 1) if (zn is not None and z_cat is not None) else ""
        dvs = round(dv(zs, z_cat), 1) if (zs is not None and z_cat is not None) else ""
        flag = "ok"
        for d in (dvn, dvs):
            if d != "" and abs(d) >= QUANTUM:
                flag = "CHECK"
        anchor = ""
        if adopted and adopted.get("redshift_check_kms"):
            a = adopted["redshift_check_kms"]
            anchor = "CIII] %s / MgII %s" % (a.get("ciii_xcorr_dv", "--"), a.get("mgii_xcorr_dv", "--"))
        rows.append({
            "folder": dec["folder"],
            "row_num": q["row_num"],
            "name_pub": q["name_pub"],
            "designation": desig,
            "z_catalog": q["z"],
            "z_ned": zn if zn is not None else "",
            "ned_name": ned.get("ned_name", ""),
            "dv_ned_kms": dvn,
            "z_simbad": zs if zs is not None else "",
            "simbad_bibcode": sim.get("simbad_bibcode", ""),
            "simbad_qual": sim.get("simbad_qual", ""),
            "dv_simbad_kms": dvs,
            "z_adopted": adopted["result"]["z"] if adopted else "",
            "anchor_check_kms": anchor,
            "flag": flag,
        })
        print("%-12s z_cat=%-10s NED=%-10s (%+6s)  SIMBAD=%-11s (%+6s)  %s"
              % (q["name_pub"][:12], q["z"], zn if zn is not None else "--", dvn or "--",
                 zs if zs is not None else "--", dvs or "--", flag))

    os.makedirs(FINAL, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("-> %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
