"""
Order the 474 catalogue objects into the sequence they should be worked through.

Ordering is the user's: category first, redshift second.

  Category is whether an SDSS spectrum exists, because that determines what
  evidence is available for confirming the systemic redshift -- and redshift
  confirmation happens while looking at the object, not as a separate campaign.

  Redshift second, because line coverage changes monotonically with z. Once a
  line has shifted into or out of the observed band it stays that way for every
  subsequent object, so neighbouring rows behave alike, decisions generalise
  across a run of them, and each transition is crossed once instead of
  repeatedly.

Nothing is dropped. Objects with no usable C IV, no rebinned spectrum, or only
IUE coverage all keep their row and carry a `work_status` saying why they are not
fittable, so that an object can never vanish quietly between the spreadsheet and
the sample.

Also derives the publication name. `common_name` is a MAST-resolvable name and is
the key the FITS stems and both override stores are built from, so it must not
change; but it is often not the recognisable name (2MASS J08105865+7602424 is
PG 0804+761). The preferred name is therefore Sulentic's where the object appears
in Sulentic et al. (2007), so readers can cross-reference that paper, then
SIMBAD's main identifier, then the MAST name. All three are carried so any choice
can be revisited.
"""

import csv
import json
import re
import os
import sys
import collections

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")
MAP = os.path.join(OUTDIR, "identity_map.csv")
RESOLUTION = os.path.join(OUTDIR, "identity_resolution.csv")
RES_CACHE = os.path.join(OUTDIR, "identity_resolution_cache.json")
SULENTIC = os.path.join(HERE, "Sulentic2007_Table1_parsed.csv")
XLSX = "/Users/gtr/Dropbox/HST/Summer2026/CIV_measurements_v22.xlsx"
OUT = os.path.join(OUTDIR, "work_queue.csv")

C = 299792.458


def squash(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


DESIG_RE = re.compile(r"(\d{4})\s*([+-])\s*(\d{2,4})")


def desig_key(name):
    """Extract a B1950-style coordinate stub, e.g. '1628+380', from a name.

    Sulentic writes "[HB89] 1628+380" where SIMBAD writes "QSO B1628+380" -- the
    prefixes differ but the coordinate stub is the same, and crucially it
    distinguishes 1628+380 from its neighbour 1628+3803, which Sulentic's own
    coordinates (good only to 0.1 min of RA) cannot.
    """
    m = DESIG_RE.search(str(name or ""))
    return "%s%s%s" % (m.group(1), m.group(2), m.group(3)) if m else ""


def load_sulentic_names(xl_by_row, idx, resolution):
    """Map row_num -> Sulentic name for objects flagged as Sulentic members.

    Matched by name stub first and position second. Position alone is not safe:
    Sulentic's IAU string is only precise to 0.1 min of RA and 1 arcmin of Dec,
    which for the pair 1628.5+3808 / 1628.6+3806 puts its single entry 65 arcsec
    from one object and 150 arcsec from the other -- and picks the wrong one. The
    name stub resolves it, since SIMBAD calls them B1628+380 and B1628+3803.

    Each Sulentic entry is assigned at most once.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from query_redshifts import desig_to_coord

    sul, coords = [], []
    for r in csv.DictReader(open(SULENTIC)):
        m = re.match(r"^J(\d{2})(\d{2})(\d)([+-])(\d{2})(\d{2})$", str(r["IAU"]).strip())
        if not m:
            continue
        hh, mm, tenth, sgn, dd, dm = m.groups()
        coords.append(((int(hh) + (int(mm) + int(tenth) / 10.0) / 60.0) * 15.0,
                       (1 if sgn == "+" else -1) * (int(dd) + int(dm) / 60.0)))
        sul.append(r)
    scat = SkyCoord([c[0] for c in coords], [c[1] for c in coords], unit=(u.deg, u.deg))
    sul_keys = [desig_key(r["Name"]) for r in sul]

    # sul_flag == 1 means "is the Sulentic match"; -1 is the user's annotation
    # for "confusable with a Sulentic object but NOT the match", so it must be
    # excluded rather than treated as membership. There are exactly 130 of the
    # former, matching the 130 rows of Sulentic Table 1.
    flagged = [rn for rn, v in xl_by_row.items()
               if str(v[idx["sul_flag"]]).strip() == "1"]
    out, taken, failed = {}, set(), []

    # Pass 1: unambiguous name-stub agreement.
    for rn in flagged:
        keys = {desig_key(resolution.get(rn, {}).get("name_simbad", "")),
                desig_key(xl_by_row[rn][idx["common_name"]])} - {""}
        hits = [i for i, k in enumerate(sul_keys) if k and k in keys and i not in taken]
        if len(hits) == 1:
            out[rn] = str(sul[hits[0]]["Name"]).strip()
            taken.add(hits[0])

    # Pass 2: nearest unclaimed entry, confirmed by redshift.
    for rn in flagged:
        if rn in out:
            continue
        v = xl_by_row[rn]
        c = desig_to_coord(str(v[idx["designation"]]).strip())
        if c is None:
            failed.append((rn, "unparseable designation"))
            continue
        seps = c.separation(scat).arcsec
        order = sorted(range(len(sul)), key=lambda i: seps[i])
        for i in order:
            if i in taken or seps[i] > 120:
                continue
            try:
                zo, zs = float(v[idx["z"]]), float(sul[i]["z"])
                if C * abs(zo - zs) / (1 + min(zo, zs)) > 3000:
                    continue
            except (TypeError, ValueError):
                pass
            out[rn] = str(sul[i]["Name"]).strip()
            taken.add(i)
            break
        else:
            failed.append((rn, "no unclaimed Sulentic entry within 120 arcsec"))

    if failed:
        print("Sulentic name unmatched for %d flagged objects:" % len(failed))
        for rn, why in failed[:10]:
            print("   row %-5s %s" % (rn, why))
        print()
    return out


def main():
    rows = list(csv.DictReader(open(MAP)))
    res = {r["row_num"]: r for r in csv.DictReader(open(RESOLUTION))}
    cache = json.load(open(RES_CACHE)) if os.path.exists(RES_CACHE) else {}

    ws = openpyxl.load_workbook(XLSX)["CIV_measurements_v22"]
    hdr = [c.value for c in ws[1]]
    idx = {h: n for n, h in enumerate(hdr)}
    xl_by_row = {}
    for row in ws.iter_rows(min_row=2):
        v = [c.value for c in row]
        if isinstance(v[idx["row_num"]], (int, float)) and v[idx["designation"]]:
            xl_by_row[str(int(v[idx["row_num"]]))] = v
    sul_names = load_sulentic_names(xl_by_row, idx, res)

    # Companions close enough to matter, excluding duplicate catalogue entries
    # for the object itself (anything within 2 arcsec sharing the resolved name).
    companions = {}
    for desig, v in cache.items():
        tgt = squash(v.get("main_id", ""))
        near = [x for x in v.get("neighbours", [])
                if x["sep"] > 2.0 and squash(x["id"]) != tgt and x["sep"] <= 30]
        if near:
            companions[desig] = near

    out = []
    for r in rows:
        rn = r["row_num"]
        rr = res.get(rn, {})
        xv = xl_by_row.get(rn, [])
        sul_name = sul_names.get(rn, "")
        name_simbad = rr.get("name_simbad", "")

        # A bracketed SIMBAD identifier is a survey-internal designation, not a
        # name a reader would recognise -- and it can name a sub-component rather
        # than the object ("[CTH90] NGC 4151 C03" is a knot inside NGC 4151).
        # The MAST name is better in every such case here.
        simbad_usable = bool(name_simbad) and not name_simbad.strip().startswith("[")
        if sul_name:
            name_pub, name_src = sul_name, "sulentic"
        elif simbad_usable:
            name_pub, name_src = name_simbad, "simbad"
        else:
            name_pub, name_src = r["common_name"], "mast"

        if not r["in_master"] or r["in_master"] == "False":
            status = ("iue_only" if r["instrument_family"] == "IUE"
                      else "no_rebinned_spectrum")
        elif r["civ_usable"] == "True":
            status = "fit"
        elif float(r["civ_frac_best"]) > 0:
            status = "civ_partial"
        else:
            status = "no_civ_coverage"

        has_sdss = str(r["sdss_flag"]).strip() in ("1", "2")
        comp = companions.get(r["designation"], [])

        out.append({
            "category": "sdss" if has_sdss else "no_sdss",
            "z_sort": float(r["z"]) if r["z"] not in ("", None) else 9.99,
            "row_num": int(rn),
            "designation": r["designation"],
            "name_pub": name_pub,
            "name_source": name_src,
            "name_mast_key": r["common_name"],
            "name_simbad": name_simbad,
            "sulentic": "yes" if sul_name else "",
            "z": r["z"],
            "z_dv_kms": r["z_dv_kms"],
            "z_needs_check": r["z_matters"],
            "sdss_flag": r["sdss_flag"],
            "oiii_observable": r["oiii_observable"],
            "validation_route": r["validation_route"],
            "work_status": status,
            "manmask": r["manmask"],
            "civ_frac_best": r["civ_frac_best"],
            "ciii_frac_best": r["ciii_frac_best"],
            "mgii_frac_best": r["mgii_frac_best"],
            "n_spectra": r["n_spectra"],
            "instruments_master": r["instruments_master"],
            "stems": r["stems"],
            "measurement_provenance": r["measurement_provenance"],
            "companion_within_30as": "; ".join(
                "%s (%s, %.0f\")" % (c["id"], c["otype"], c["sep"]) for c in comp),
        })

    # No-SDSS first, per the user. These are the HST-only objects: the core of
    # the sample and the simpler case, since nothing about them depends on the
    # still-unsettled HST+SDSS splicing question.
    out.sort(key=lambda r: (0 if r["category"] == "no_sdss" else 1, r["z_sort"]))
    for i, r in enumerate(out, 1):
        r["order"] = i
    cols = ["order"] + [k for k in out[0] if k not in ("order", "z_sort")]

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print("wrote %d rows -> %s\n" % (len(out), OUT))

    print("%-9s %-20s %5s" % ("category", "work_status", "n"))
    cc = collections.Counter((r["category"], r["work_status"]) for r in out)
    for (cat, st), n in sorted(cc.items()):
        print("%-9s %-20s %5d" % (cat, st, n))
    print("\nfittable (work_status=fit): %d"
          % sum(1 for r in out if r["work_status"] == "fit"))
    print("publication name source:",
          dict(collections.Counter(r["name_source"] for r in out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
