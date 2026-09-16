"""
Collect published redshifts from NED and SIMBAD for objects whose catalogued
redshift is too coarse to support a wind measurement.

Motivation: the C IV blueshift is reported on a 69 km/s grid, so a redshift
quoted to 3 decimal places at z ~ 0.1 carries ~135 km/s of implied uncertainty --
two grid steps -- and two objects in this catalogue are quoted to *one* decimal
place. Those cannot support a velocity measurement at all.

What this does NOT do is pick a winner. The intended standard is a redshift from
narrow optical lines, but NED's per-measurement metadata turns out to be almost
entirely unpopulated -- for PG 0804+761, 76 of 78 measurements have a blank
Spectral Range, the Measurement Mode columns are blank throughout, and 75 are
flagged "Uncertain origin". There is therefore no reliable way to select
narrow-line measurements automatically, and pretending otherwise would produce a
confident-looking number with nothing behind it.

Worse, NED's per-object redshift tables are contaminated: the published values for
PG 1411+442 span 147,000 km/s, which is impossible for a single AGN, so the table
evidently mixes in measurements of other sources in the field. Any "best" value
selected from such a list is meaningless -- an early version of this script picked
z=0.068 for Mrk 876, whose true redshift is 0.129.

So NED is used for what it is reliable at -- resolving a coordinate to a canonical
object name, which is genuinely valuable here (2MASS J08105865+7602424 turns out to
be PG 0804+761) -- and SIMBAD supplies the candidate redshift, since its curated
value carries an uncertainty, a bibcode and a quality grade. NED's published table
is fetched only with --with-ned-table, and then only as a sanity check: a median to
compare against, and a spread that says whether the table is trustworthy at all.

Nothing here is adopted automatically. Neither service reports which line a
redshift came from, so neither can confirm the narrow-optical-line standard this
work needs; these are candidates for a human to check, not answers.

Objects are matched by cone search on the coordinate-encoded `designation`
rather than by name, because names like "LEDA 3095439" and
"2MASS J08105865+7602424" resolve poorly -- the latter is in fact PG 0804+761.

Results are cached to JSON so reruns do not re-query the services.
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import astropy.units as u
from astropy.coordinates import SkyCoord

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")
MAP = os.path.join(OUTDIR, "identity_map.csv")
CACHE = os.path.join(OUTDIR, "redshift_lookup_cache.json")
OUT = os.path.join(OUTDIR, "redshift_lookup.csv")

C = 299792.458
QUANTUM = 69.0
CONE = 10 * u.arcsec


def desig_to_coord(d):
    m = re.match(r"^J(\d{2})(\d{2})(\d{2}\.?\d*)([+-])(\d{2})(\d{2})(\d{2}\.?\d*)$",
                 str(d).strip())
    if not m:
        return None
    h, mi, s, sgn, dd, dm, ds = m.groups()
    return SkyCoord("%s:%s:%s %s%s:%s:%s" % (h, mi, s, sgn, dd, dm, ds),
                    unit=(u.hourangle, u.deg))


def dv(z_a, z_b, z_ref):
    """Velocity difference between two redshifts, at the reference redshift."""
    return C * abs(float(z_a) - float(z_b)) / (1.0 + float(z_ref))


def query_ned(coord, want_table=False):
    from astroquery.ipac.ned import Ned
    Ned.TIMEOUT = 180
    out = {"ned_name": "", "ned_z": None, "published": [], "error": ""}
    try:
        reg = Ned.query_region(coord, radius=CONE)
    except Exception as exc:
        out["error"] = "region: %s" % type(exc).__name__
        return out
    if reg is None or len(reg) == 0:
        out["error"] = "no NED object within %s" % CONE
        return out
    # Sorting by separation alone picks whichever catalogue happens to have the
    # closest cross-match, which can be a radio or X-ray counterpart carrying no
    # redshift (3C 120 matched a NVGRC radio source, Pic A a Chandra source).
    # Prefer a match that actually has a redshift, then the nearest of those.
    def _has_z(r):
        try:
            return 0.0 <= float(r["Redshift"]) < 8
        except (TypeError, ValueError):
            return False
    reg.sort("Separation")
    with_z = [r for r in reg if _has_z(r)]
    row = with_z[0] if with_z else reg[0]
    out["ned_name"] = str(row["Object Name"]).strip()
    try:
        out["ned_z"] = float(row["Redshift"])
    except (TypeError, ValueError):
        out["ned_z"] = None
    if not want_table:
        return out
    try:
        t = Ned.get_table(out["ned_name"], table="redshifts")
    except Exception as exc:
        out["error"] = "table: %s" % type(exc).__name__
        return out
    for r in t:
        try:
            z = float(r["Published Redshift"])
        except (TypeError, ValueError):
            continue
        if not (-0.01 < z < 8):
            continue
        unc = r["Published Redshift Uncertainty"]
        try:
            unc = float(unc)
            unc = None if unc <= 0 else unc
        except (TypeError, ValueError):
            unc = None
        out["published"].append({
            "z": z,
            "unc": unc,
            "refcode": str(r["Refcode"]).strip(),
            "spectral_range": str(r["Spectral Range"]).strip(),
            "qualifiers": str(r["Qualifiers"]).strip(),
        })
    return out


def query_simbad(coord):
    from astroquery.simbad import Simbad
    s = Simbad()
    for f in ("rvz_redshift", "rvz_error", "rvz_type", "rvz_bibcode", "rvz_qual",
              "otype"):
        try:
            s.add_votable_fields(f)
        except Exception:
            pass
    out = {"simbad_name": "", "simbad_z": None, "simbad_err": None,
           "simbad_bibcode": "", "simbad_qual": "", "simbad_otype": "",
           "simbad_sep_arcsec": None, "error": ""}
    try:
        t = s.query_region(coord, radius=CONE)
    except Exception as exc:
        out["error"] = type(exc).__name__
        return out
    if t is None or len(t) == 0:
        out["error"] = "no SIMBAD object within %s" % CONE
        return out
    # SIMBAD does NOT return region results sorted by separation. Taking row 0
    # picked a foreground galaxy at z=0.12 over QSO J0635-7516 at z=0.65, which
    # sat third in the list. Sort by true angular separation and take the nearest.
    if {"ra", "dec"} <= set(t.colnames):
        seps = coord.separation(
            SkyCoord(list(t["ra"]), list(t["dec"]), unit=(u.deg, u.deg)))
        order = sorted(range(len(t)), key=lambda i: seps[i].arcsec)
        r = t[order[0]]
        out["simbad_sep_arcsec"] = round(float(seps[order[0]].arcsec), 2)
    else:
        r = t[0]
    g = lambda k: r[k] if k in t.colnames else None
    out["simbad_name"] = str(g("main_id") or "").strip()
    for key, col in (("simbad_z", "rvz_redshift"), ("simbad_err", "rvz_err")):
        v = g(col)
        try:
            out[key] = None if v is None else float(v)
        except (TypeError, ValueError):
            out[key] = None
    out["simbad_bibcode"] = str(g("rvz_bibcode") or "").strip()
    out["simbad_qual"] = str(g("rvz_qual") or "").strip()
    out["simbad_otype"] = str(g("otype") or "").strip()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-dv", type=float, default=QUANTUM,
                    help="only objects whose quoted precision implies at least "
                         "this velocity uncertainty (default: one 69 km/s quantum)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--with-ned-table", action="store_true",
                    help="also fetch NED's published-redshift table as a sanity "
                         "check; slow, error-prone, and often contaminated")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(MAP)))
    queue = [r for r in rows
             if r["z_dv_kms"] not in ("", None) and float(r["z_dv_kms"]) >= args.min_dv]
    queue.sort(key=lambda r: -float(r["z_dv_kms"]))
    if args.limit:
        queue = queue[:args.limit]
    print("queue: %d objects (implied uncertainty >= %.0f km/s)\n"
          % (len(queue), args.min_dv), flush=True)

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    out_rows = []

    for i, r in enumerate(queue, 1):
        key = str(r["designation"])
        coord = desig_to_coord(key)
        if coord is None:
            print("  %-3d %-30s DESIGNATION UNPARSEABLE" % (i, r["common_name"][:30]))
            continue
        if key not in cache:
            ned = query_ned(coord, want_table=args.with_ned_table)
            time.sleep(args.sleep)
            sim = query_simbad(coord)
            time.sleep(args.sleep)
            cache[key] = {"ned": ned, "simbad": sim}
            json.dump(cache, open(CACHE, "w"), indent=1)
        ned, sim = cache[key]["ned"], cache[key]["simbad"]

        z_cat = float(r["z"])
        pub = [p["z"] for p in ned["published"]]
        spread = (dv(max(pub), min(pub), z_cat) if len(pub) > 1 else None)
        # A spread of more than a few thousand km/s means the table is mixing in
        # other sources; its median is then not evidence about this object.
        table_ok = "" if spread is None else ("yes" if spread < 3000 else "no")

        rec = {
            "row_num": r["row_num"],
            "common_name": r["common_name"],
            "designation": key,
            "manmask": r["manmask"],
            "civ_usable": r["civ_usable"],
            "validation_route": r["validation_route"],
            "z_catalog": z_cat,
            "z_decimals": r["z_decimals"],
            "z_dv_kms": r["z_dv_kms"],
            "ned_name": ned["ned_name"],
            "n_published": len(pub),
            "pub_median": round(statistics.median(pub), 6) if pub else "",
            "pub_spread_kms": round(spread, 1) if spread is not None else "",
            "ned_table_reliable": table_ok,
            "simbad_name": sim["simbad_name"],
            "simbad_z": sim["simbad_z"] if sim["simbad_z"] is not None else "",
            "simbad_err": sim["simbad_err"] if sim["simbad_err"] is not None else "",
            "simbad_bibcode": sim["simbad_bibcode"],
            "simbad_qual": sim["simbad_qual"],
            "simbad_otype": sim.get("simbad_otype", ""),
            "simbad_sep_arcsec": sim.get("simbad_sep_arcsec", ""),
            "dv_catalog_vs_simbad": (round(dv(z_cat, sim["simbad_z"], z_cat), 1)
                                     if sim["simbad_z"] is not None else ""),
            "changes_answer": "",
            "error": "; ".join(x for x in (ned["error"], sim["error"]) if x),
        }
        if rec["dv_catalog_vs_simbad"] != "":
            d = rec["dv_catalog_vs_simbad"]
            # An offset this large is not a redshift correction, it is a
            # different object. Flag rather than silently propose a 10^5 km/s fix.
            if d > 5000:
                rec["changes_answer"] = "LIKELY MISMATCH"
            else:
                rec["changes_answer"] = "yes" if d >= QUANTUM else "no"
        out_rows.append(rec)
        print("  %-3d %-28s NED=%-22s n_pub=%-4s simbad_z=%-12s dv=%s km/s"
              % (i, r["common_name"][:28], (ned["ned_name"] or "-")[:22],
                 len(pub), rec["simbad_z"], rec["dv_catalog_vs_simbad"]), flush=True)

    if out_rows:
        with open(OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print("\nwrote %d rows -> %s" % (len(out_rows), OUT))
        n = sum(1 for r in out_rows if r["changes_answer"] == "yes")
        print("objects where an external redshift moves the answer "
              "by >= one quantum: %d of %d" % (n, len(out_rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
