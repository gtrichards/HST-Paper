"""
Resolve every catalogue object to a canonical identity, and flag the ones where
that identity is not safe to assume.

Two separate questions, deliberately kept apart:

  Which object is this?  The xlsx `common_name` values are MAST-resolvable names,
  which is why the pipeline uses them and why the rebinned FITS stems are built
  from them. They are not wrong, but they are often not the familiar name --
  2MASS J08105865+7602424 is PG 0804+761. Recording the familiar name alongside,
  rather than renaming anything, keeps the existing joins intact.

  Could we have the wrong object?  A nearest-match cone search is not proof of
  identity: distinct AGN can sit close together, as 1628.5+3808 and 1628.6+3806
  do at 121 arcsec. So this also counts AGN-like neighbours out to a wider radius
  and flags anything with company, rather than silently taking the nearest.

Writes one row per object. Resolves nothing by force: where SIMBAD finds nothing,
or finds more than one plausible AGN, the row says so and the decision is left
open.
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import astropy.units as u
from astropy.coordinates import SkyCoord

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")
MAP = os.path.join(OUTDIR, "identity_map.csv")
CACHE = os.path.join(OUTDIR, "identity_resolution_cache.json")
OUT = os.path.join(OUTDIR, "identity_resolution.csv")

# Tight radius to identify the object; wider one to notice company.
ID_RADIUS = 5 * u.arcsec
NEIGHBOUR_RADIUS = 3 * u.arcmin

# SIMBAD otypes that could plausibly be confused with the target.
AGN_TYPES = {"QSO", "AGN", "Sy1", "Sy2", "SyG", "Bla", "BLL", "BLLac", "LIN", "rG"}


def desig_to_coord(d):
    from query_redshifts import desig_to_coord as f
    return f(d)


def resolve(coord, simbad):
    out = {"main_id": "", "otype": "", "sep_arcsec": None, "z": None,
           "n_agn_near": 0, "neighbours": [], "error": ""}
    try:
        t = simbad.query_region(coord, radius=NEIGHBOUR_RADIUS)
    except Exception as exc:
        out["error"] = type(exc).__name__
        return out
    if t is None or len(t) == 0:
        out["error"] = "nothing within %s" % NEIGHBOUR_RADIUS
        return out
    if not {"ra", "dec"} <= set(t.colnames):
        out["error"] = "no coordinates returned"
        return out

    seps = coord.separation(SkyCoord(list(t["ra"]), list(t["dec"]),
                                     unit=(u.deg, u.deg)))
    order = sorted(range(len(t)), key=lambda i: seps[i].arcsec)
    g = lambda row, k: (row[k] if k in t.colnames else None)

    for i in order:
        row = t[i]
        otype = str(g(row, "otype") or "").strip()
        arcsec = float(seps[i].arcsec)
        if otype in AGN_TYPES:
            out["n_agn_near"] += 1
            if len(out["neighbours"]) < 4:
                out["neighbours"].append(
                    {"id": str(g(row, "main_id") or "").strip(),
                     "otype": otype, "sep": round(arcsec, 1)})
        # The identification itself must be close, not merely nearest.
        if not out["main_id"] and arcsec <= ID_RADIUS.to_value(u.arcsec):
            out["main_id"] = str(g(row, "main_id") or "").strip()
            out["otype"] = otype
            out["sep_arcsec"] = round(arcsec, 2)
            try:
                out["z"] = float(g(row, "rvz_redshift"))
            except (TypeError, ValueError):
                out["z"] = None
    if not out["main_id"]:
        out["error"] = "no match within %s" % ID_RADIUS
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    from astroquery.simbad import Simbad
    simbad = Simbad()
    for f in ("rvz_redshift", "rvz_qual", "rvz_bibcode", "otype"):
        try:
            simbad.add_votable_fields(f)
        except Exception:
            pass

    rows = list(csv.DictReader(open(MAP)))
    if args.limit:
        rows = rows[:args.limit]
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    out_rows = []

    for i, r in enumerate(rows, 1):
        key = str(r["designation"])
        coord = desig_to_coord(key)
        if coord is None:
            continue
        if key not in cache:
            cache[key] = resolve(coord, simbad)
            json.dump(cache, open(CACHE, "w"), indent=1)
            time.sleep(args.sleep)
        res = cache[key]

        # Does the familiar name differ from the one the pipeline keys on?
        def squash(s):
            return "".join(ch for ch in str(s).lower() if ch.isalnum())
        differs = bool(res["main_id"]) and squash(res["main_id"]) != squash(r["common_name"])

        out_rows.append({
            "row_num": r["row_num"],
            "designation": key,
            "name_mast": r["common_name"],
            "name_simbad": res["main_id"],
            "name_differs": differs,
            "otype": res["otype"],
            "id_sep_arcsec": res["sep_arcsec"] if res["sep_arcsec"] is not None else "",
            "simbad_z": res["z"] if res["z"] is not None else "",
            "n_agn_within_3arcmin": res["n_agn_near"],
            "confusion_risk": "yes" if res["n_agn_near"] > 1 else "no",
            "neighbours": "; ".join("%s (%s, %.0f\")" % (n["id"], n["otype"], n["sep"])
                                    for n in res["neighbours"]),
            "manmask": r["manmask"],
            "civ_usable": r["civ_usable"],
            "error": res["error"],
        })
        if i % 50 == 0:
            print("  %d/%d" % (i, len(rows)), flush=True)

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    n_unres = sum(1 for r in out_rows if not r["name_simbad"])
    n_diff = sum(1 for r in out_rows if r["name_differs"])
    n_risk = sum(1 for r in out_rows if r["confusion_risk"] == "yes")
    print("\nwrote %d rows -> %s" % (len(out_rows), OUT))
    print("unresolved within %s        : %d" % (ID_RADIUS, n_unres))
    print("SIMBAD name differs from MAST: %d" % n_diff)
    print("confusion risk (>1 AGN near) : %d" % n_risk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
