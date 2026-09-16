"""
Build the one-row-per-object spine for the C IV catalogue.

Every one of the 474 object rows in CIV_measurements_v22.xlsx gets exactly one row
here (the sheet's 475th populated row is a totals row of Excel formulas, not an
object -- that is the source of the long-standing 475-vs-474 discrepancy),
whether or not it has data, a fit, or a future. Nothing is dropped; objects that
cannot be used are flagged. That is the whole point -- the failure mode this
replaces is an object quietly vanishing between the spreadsheet and the sample.

It joins five records that until now had no common key:

  xlsx row           keyed by row_num / designation / common_name
  master spectra     keyed by FITS stem, "<common_name>_<INSTRUMENT>"
  GTR override store keyed by the same stem
  GTR measurements   keyed by the same stem
  AP override CSVs   keyed by "<common_name>_<INSTRUMENT>" too, in two generations

The xlsx <-> stem join runs on a normalised common_name; that resolves 426 of the
426 distinct stem names, so it is a complete correspondence rather than a
best-effort match.

Also derives two things that decide later work:

  Redshift precision. The quoted decimal places imply a velocity uncertainty of
  c * (half-ulp) / (1+z). Compared against the 69 km/s grid quantum, this says
  whether a coarse redshift can actually move the published blueshift -- most
  cannot, and only those that can are worth chasing.

  Validation route. Which evidence exists to check the redshift at all: a UV
  anchor (C III] / Mg II in the HST spectrum), an SDSS spectrum with [O III]
  observable, an SDSS spectrum without it, or nothing but NED/SIMBAD.
"""

import csv
import json
import os
import re
import sys
import collections

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")

XLSX = "/Users/gtr/Dropbox/HST/Summer2026/CIV_measurements_v22.xlsx"
SHEET = "CIV_measurements_v22"
COVERAGE = os.path.join(OUTDIR, "master_line_coverage.csv")
REPO = "/Users/gtr/Work/git/HST-Paper"
OV_JSON = os.path.join(REPO, "ica", "manual_fix_overrides.json")
MEAS_CSV = os.path.join(REPO, "ica", "manual_fix_measurements.csv")
GTR_PLOTS = os.path.join(REPO, "ICA_Plots_Rebin_master", "ManualFix")
AP_DIRS = [
    ("overrides", "/Users/gtr/Dropbox/HST/Pratsos/ICA_Plots_Rebin_overrides"),
    ("overrides2", "/Users/gtr/Dropbox/HST/Pratsos/ICA_Plots_Rebin_overrides2"),
]

C = 299792.458
QUANTUM = 69.0          # km/s; the C IV blueshift grid step
GOOD = 0.8              # fraction of a line window that must be usable
OIII = 5006.84
SDSS_LO, SDSS_HI = 3800.0, 9200.0
IUE = {"SWP", "LWP", "LWR"}


def norm(s):
    return re.sub(r"[^a-z0-9+.-]", "", str(s).strip().lower()) if s is not None else ""


def decimals(v):
    """Decimal places in the value as written, or -1 if not a number."""
    s = str(v).strip()
    try:
        float(s)
    except (TypeError, ValueError):
        return -1
    return len(s.split(".")[1]) if "." in s else 0


def z_uncertainty_kms(z, dp):
    """Velocity implied by the quoted precision. Half-ulp, redshift-corrected."""
    if dp < 0 or z is None:
        return None
    return C * (0.5 * 10 ** (-dp)) / (1.0 + float(z))


def instrument_family(instr):
    parts = {p.strip().upper() for p in str(instr or "").replace(",", "/").split("/") if p.strip()}
    if not parts:
        return "none"
    if parts <= IUE:
        return "IUE"
    if parts & IUE:
        return "mixed"
    return "HST"


def load_alex():
    """Alex's recorded fits, keyed by object name, with what is reproducible."""
    out = collections.defaultdict(list)
    for gen, d in AP_DIRS:
        path = os.path.join(d, "ICA_Results_Rebin_overrides.csv")
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            out[norm(r["object_name"].rsplit("_", 1)[0])].append({
                "gen": gen,
                "object_name": r["object_name"],
                "override_type": r.get("override_type", ""),
                "masking": str(r.get("manual_masking_applied", "")).strip().lower() == "true",
                "forced": str(r.get("forced_components", "")).strip().lower() == "true",
                "civ_blueshift": r.get("CIV_blueshift", ""),
                "redshift": r.get("redshift", ""),
            })
    return out


def main():
    cov = list(csv.DictReader(open(COVERAGE)))
    by_name = collections.defaultdict(list)
    for r in cov:
        by_name[norm(r["name_from_stem"])].append(r)

    overrides = json.load(open(OV_JSON))
    measurements = {r["object_name"]: r for r in csv.DictReader(open(MEAS_CSV))}
    gtr_plots = (set(os.listdir(GTR_PLOTS)) if os.path.isdir(GTR_PLOTS) else set())
    alex = load_alex()

    ws = openpyxl.load_workbook(XLSX)[SHEET]
    hdr = [c.value for c in ws[1]]
    idx = {h: n for n, h in enumerate(hdr)}
    # The sheet ends with a blank row and a totals row carrying Excel formulas
    # (=SUM/=COUNTIF), which is not an object. Keep only rows with a numeric
    # row_num and a designation -- that is 474 objects, matching the crossref.
    xl = []
    for row in ws.iter_rows(min_row=2):
        v = [c.value for c in row]
        if not any(x is not None for x in v):
            continue
        if not isinstance(v[idx["row_num"]], (int, float)):
            continue
        if v[idx["designation"]] in (None, ""):
            continue
        xl.append(v)

    rows = []
    for r in xl:
        cname = r[idx["common_name"]]
        key = norm(cname)
        specs = by_name.get(key, [])
        fl = str(r[idx["sdss_flag"]]).strip() if r[idx["sdss_flag"]] not in (None, "") else ""
        z = r[idx["z"]]
        dp = decimals(z)
        dv = z_uncertainty_kms(z, dp) if dp >= 0 else None

        fr = lambda s, k: float(s[k])
        civ_best = max((fr(s, "civ_frac") for s in specs), default=0.0)
        ciii_best = max((fr(s, "ciii_frac") for s in specs), default=0.0)
        mgii_best = max((fr(s, "mgii_frac") for s in specs), default=0.0)
        siiv_best = max((fr(s, "siiv_frac") for s in specs), default=0.0)

        has_sdss = fl in ("1", "2")
        oiii_ok = bool(has_sdss and z not in (None, "")
                       and SDSS_LO <= OIII * (1 + float(z)) <= SDSS_HI)
        anchor = (ciii_best >= GOOD) or (mgii_best >= GOOD)
        if anchor:
            route = "uv_anchor"
        elif oiii_ok:
            route = "sdss_oiii"
        elif has_sdss:
            route = "sdss_no_oiii"
        else:
            route = "ned_simbad"

        stems = [s["stem"] for s in specs]
        ov_hits = [s for s in stems if s in overrides]
        me_hits = [s for s in stems if s in measurements]
        plot_hits = [s for s in stems if s + ".png" in gtr_plots]
        ap = alex.get(key, [])
        ap_mask = [a for a in ap if a["masking"]]

        # Which class of provenance does an already-recorded measurement have?
        if r[idx["CIV_blue"]] in (None, ""):
            prov = ""
        elif me_hits:
            prov = "gtr_reproducible"
        elif ap_mask:
            prov = "alex_mask_NOT_reproducible"
        elif ap:
            prov = "alex_partly_reproducible"
        else:
            prov = "unknown_origin"

        rows.append({
            "row_num": r[idx["row_num"]],
            "designation": r[idx["designation"]],
            "common_name": cname,
            "instrument_xlsx": r[idx["instrument"]],
            "instrument_family": instrument_family(r[idx["instrument"]]),
            "manmask": r[idx["CIVgood_manmask"]],
            "gtr_notes": r[idx["GTR Notes"]],
            "CIV_blue": r[idx["CIV_blue"]],
            "CIV_EW": r[idx["CIV_EW"]],
            "z": z,
            "z_source": r[idx["z_source"]],
            "z_rescaled": r[idx["z_rescaled"]],
            "z_decimals": dp,
            "z_dv_kms": None if dv is None else round(dv, 1),
            "z_matters": "" if dv is None else ("yes" if dv >= QUANTUM else "no"),
            "sdss_flag": fl,
            "oiii_observable": oiii_ok,
            "validation_route": route,
            "in_master": bool(specs),
            "n_spectra": len(specs),
            "stems": ";".join(sorted(stems)),
            "instruments_master": ";".join(sorted({s["instrument_from_stem"] for s in specs})),
            "civ_frac_best": round(civ_best, 4),
            "ciii_frac_best": round(ciii_best, 4),
            "mgii_frac_best": round(mgii_best, 4),
            "siiv_frac_best": round(siiv_best, 4),
            "civ_usable": civ_best >= GOOD,
            "n_override_gtr": len(ov_hits),
            "n_measurement_gtr": len(me_hits),
            "n_plot_gtr": len(plot_hits),
            "n_fits_alex": len(ap),
            "alex_generations": ";".join(sorted({a["gen"] for a in ap})),
            "alex_override_types": ";".join(sorted({a["override_type"] for a in ap if a["override_type"]})),
            "alex_mask_based": bool(ap_mask),
            "measurement_provenance": prov,
        })

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "identity_map.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %d rows -> %s\n" % (len(rows), out))

    def count(pred):
        return sum(1 for r in rows if pred(r))

    print("matched to master spectra : %d of %d" % (count(lambda r: r["in_master"]), len(rows)))
    print("  unmatched, HST          : %d" % count(lambda r: not r["in_master"] and r["instrument_family"] == "HST"))
    print("  unmatched, IUE          : %d" % count(lambda r: not r["in_master"] and r["instrument_family"] == "IUE"))
    print("usable C IV               : %d" % count(lambda r: r["civ_usable"]))

    print("\nvalidation route (objects with usable C IV):")
    c = collections.Counter(r["validation_route"] for r in rows if r["civ_usable"])
    for k, v in c.most_common():
        print("   %-16s %d" % (k, v))

    print("\nrecorded measurements by provenance:")
    c = collections.Counter(r["measurement_provenance"] for r in rows if r["measurement_provenance"])
    for k, v in c.most_common():
        print("   %-28s %d" % (k, v))

    print("\nredshift precision (all rows):")
    print("   coarse enough to matter (>=%.0f km/s) : %d"
          % (QUANTUM, count(lambda r: r["z_matters"] == "yes")))
    print("   below one quantum                    : %d"
          % count(lambda r: r["z_matters"] == "no"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
