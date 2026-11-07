"""On how many adopted objects could the HST solution be overplotted on IUE data?

GTR, 2026-11-06: "I also want to push back on overplotting.  Clearly we could
overplot the HST solution on the IUE data for I Zw 1, for example."  He was
right and the earlier answer of "zero" was wrong because it was computed from
the 31 objects already RETRIEVED from the IUE archive -- the 22 IUE-only
objects plus the 9 gap objects -- which by construction excludes every object
that has a usable HST spectrum.  The question is about the ARCHIVE, not about
what happens to be on this disk.

The v22 catalogue cannot answer it either.  Its `instrument` column records the
instrument a MEASUREMENT came from, so it names SWP/LWP/LWR for exactly the ten
IUE-only objects and HST for everybody else; reading it as a record of IUE
holdings reproduces the same wrong answer in a different way.  Only MAST knows.

So this queries MAST by position for every adopted object (manmask 2 or 3) and
asks whether IUE holds a low-dispersion spectrum in the camera that reaches C IV
at that object's redshift:

    SWP   1150-1980 A    C IV 1549.48 A for z < 0.28
    LWP   1850-3350 A    C IV for 0.19 < z < 1.16
    LWR   1850-3350 A    same

An object is an OVERPLOT CANDIDATE when such an observation exists.  That is a
statement about coverage only -- IUE is a 45 cm telescope and most of these are
faint, so a candidate is not a promise that the overplot will be informative;
texp_useful_s is reported so depth can be judged, and the 22 IUE-only objects
showed that about half of what the archive holds is too shallow to fit.

Position comes from the catalogue designation, which is Jhhmmss.ss+ddmmss.s and
so localises an object to well under an arcsecond -- unlike Amogh's HHMM+DDM
designations, which needed a 600 arcsec radius.  5 arcsec is therefore right
here, and is the radius survey_iue.py already uses.

    python scan_iue_overplot.py              # every adopted object
    python scan_iue_overplot.py --limit 20   # a quick check of the plumbing
"""
import argparse
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from astroquery.mast import Observations

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = "/Users/gtr/Work/git/HST-Paper/Data/CIV_measurements_v23.csv"
LEDGER = "/Users/gtr/Work/git/HST-Paper/Data/decisions.csv"
QUEUE = os.path.join(os.path.dirname(HERE), "pipeline_output", "work_queue.csv")
OUT = os.path.join(HERE, "iue_overplot_scan.csv")

RADIUS_ARCSEC = 5.0
CIV = 1549.48
CAM = {"SWP": (1150.0, 1980.0), "LWP": (1850.0, 3350.0), "LWR": (1850.0, 3350.0)}
#: Low dispersion is what the pipeline reads (MXLO); high dispersion needs a
#: different extraction and was never part of this analysis.
LOWDISP = ("LOW", "LO")


def designation_to_coord(d):
    """Jhhmmss.ss+ddmmss.s -> (ra_deg, dec_deg)."""
    s = str(d).strip().lstrip("J")
    i = max(s.rfind("+"), s.rfind("-"))
    ra_s, de_s = s[:i], s[i:]
    ra = (float(ra_s[0:2]) + float(ra_s[2:4]) / 60.0 + float(ra_s[4:]) / 3600.0) * 15.0
    sign = -1.0 if de_s[0] == "-" else 1.0
    de_s = de_s[1:]
    dec = sign * (float(de_s[0:2]) + float(de_s[2:4]) / 60.0 + float(de_s[4:]) / 3600.0)
    return ra, dec


def adopted_rows():
    """(order, row_num, manmask) for every manmask 2 or 3 object."""
    mm = {}
    for r in pd.read_csv(LEDGER).to_dict("records"):
        o = int(re.match(r"^(\d+)", str(r["folder"])).group(1))
        mm[o] = str(r["manmask"]).strip()
    o_for_row = {int(r["row_num"]): int(r["order"])
                 for r in pd.read_csv(QUEUE).to_dict("records")}
    out = {}
    for rn, o in o_for_row.items():
        if mm.get(o) in ("2", "3"):
            out[rn] = (o, mm[o])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    t = pd.read_csv(TABLE)
    keep = adopted_rows()
    t = t[t.row_num.astype(int).isin(keep)].sort_values("row_num")
    if a.limit:
        t = t.head(a.limit)
    print("querying MAST for IUE holdings of %d adopted objects" % len(t), flush=True)

    rad = RADIUS_ARCSEC / 3600.0
    rows = []
    for _, r in t.iterrows():
        rn = int(r["row_num"])
        order, mmv = keep[rn]
        name = str(r["common_name"]).strip()
        z = float(r["z"])
        civ_obs = CIV * (1.0 + z)
        wants = [c for c, (lo, hi) in CAM.items() if lo <= civ_obs <= hi]
        n_tot = n_use = 0
        texp = 0.0
        cams = "-"
        err = ""
        if wants:
            try:
                ra, dec = designation_to_coord(r["designation"])
                q = Observations.query_criteria(
                    obs_collection="IUE", dataproduct_type="spectrum",
                    s_ra=[ra - rad, ra + rad], s_dec=[dec - rad, dec + rad])
                d = q.to_pandas() if len(q) else pd.DataFrame()
            except Exception as exc:
                d = pd.DataFrame()
                err = type(exc).__name__
            if len(d):
                n_tot = len(d)
                cams = "/".join("%s:%d" % (k, v) for k, v in
                                sorted(dict(d["instrument_name"].value_counts()).items()))
                u = d[d["instrument_name"].isin(wants)]
                # Keep low dispersion only where the column exists; MAST's IUE
                # rows do not always carry it, and dropping a row for a missing
                # keyword would undercount coverage.
                if "filters" in u.columns:
                    f = u["filters"].astype(str).str.upper()
                    known = f.str.contains("LOW") | f.str.contains("LARGE") | f.str.contains("SMALL")
                    u = u[~known | f.str.contains("LOW")] if known.any() else u
                n_use = len(u)
                texp = float(u["t_exptime"].sum()) if n_use else 0.0
        rows.append(dict(order=order, row_num=rn, name_pub=name, z=z, manmask=mmv,
                         civ_observed_A=round(civ_obs, 1),
                         camera_for_civ="/".join(sorted(wants)) or "NONE",
                         n_obs_total=n_tot, n_obs_useful=n_use,
                         cameras_found=cams, texp_useful_s=round(texp, 0),
                         overplot=int(n_use > 0), query_error=err))
        print("  %-5d %-26s z=%.4f  C IV %6.1f A -> %-7s | %3d obs (%s) | useful %3d, %8.0f s%s"
              % (order, name[:26], z, civ_obs, rows[-1]["camera_for_civ"], n_tot,
                 cams[:40], n_use, texp, ("  [%s]" % err) if err else ""), flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print()
    print("ADOPTED OBJECTS SCANNED                      : %d" % len(d))
    print("  C IV outside every IUE camera at their z   : %d" % (d.camera_for_civ == "NONE").sum())
    print("  no IUE observation at all                  : %d" % ((d.n_obs_total == 0) &
                                                                 (d.camera_for_civ != "NONE")).sum())
    print("  IUE data but not in a C IV-covering camera : %d" % ((d.n_obs_total > 0) &
                                                                 (d.n_obs_useful == 0)).sum())
    print()
    print("  *** OVERPLOT CANDIDATES                    : %d ***" % int(d.overplot.sum()))
    k = d[d.overplot == 1].sort_values("texp_useful_s", ascending=False)
    for band, lo in (("> 10 ks of useful exposure", 10000.0),
                     ("2-10 ks", 2000.0), ("< 2 ks", 0.0)):
        s = k[(k.texp_useful_s >= lo) & (k.texp_useful_s < (1e30 if lo == 10000 else
                                                            (10000.0 if lo == 2000 else 2000.0)))]
        print("        %-28s %d" % (band, len(s)))
    if (d.query_error != "").any():
        print()
        print("  QUERIES THAT FAILED (not counted either way): %d"
              % int((d.query_error != "").sum()))
    print()
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
