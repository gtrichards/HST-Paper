"""What IUE data exists for the manmask-1 objects, and why most are not in
Amogh's "IUE good" set.

GTR: "if there are objects in Table 1 with manmask=1 that aren't included, then
we need to figure out why.  That probably means downloading the IUE data.  That
would allow us to determine if some really shouldn't be included (as was the
case for some HST objects), spot check Amogh's work (who may not have masked
enough outside the CIV region), and make any measurements that he did not."

Of our 22 manmask-1 (IUE-only) objects, only three appear in Amogh's set:
3C 120, MCG8-11-11 and PG1310-1051.  This surveys the other 19 at MAST before
anything is downloaded, so we know what we are dealing with.

C IV COVERAGE BY CAMERA.  C IV 1549.48 A lands in SWP (about 1150-1980 A) for
z < 0.28 and in LWP/LWR (about 1850-3350 A) for z > 0.19.  Every one of the 19
has at least one camera that should reach it, so coverage is not the obvious
explanation and depth probably is -- IUE is a 45 cm telescope and these are
mostly faint.

Writes iue/iue_survey.csv: one row per object with the camera counts, total
exposure time, and which camera should carry C IV at that redshift.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from astroquery.mast import Observations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "git", "HST-Paper",
                                "paper_figures_v23"))
TABLE = "/Users/gtr/Work/git/HST-Paper/Data/CIV_measurements_v23.csv"
OUT = os.path.join(HERE, "iue_survey.csv")
RADIUS_ARCSEC = 5.0
CIV = 1549.48
#: nominal camera ranges, Angstrom
CAM = {"SWP": (1150.0, 1980.0), "LWP": (1850.0, 3350.0), "LWR": (1850.0, 3350.0)}
ALREADY = {"3C 120", "MCG8-11-11", "PG1310-1051"}


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


def main():
    t = pd.read_csv(TABLE)
    m1 = t[t["CIVgood_manmask"] == 1].sort_values("row_num")
    rad = RADIUS_ARCSEC / 3600.0
    rows = []
    for _, r in m1.iterrows():
        name = str(r["common_name"]); z = float(r["z"])
        ra, dec = designation_to_coord(r["designation"])
        civ_obs = CIV * (1.0 + z)
        wants = [c for c, (lo, hi) in CAM.items() if lo <= civ_obs <= hi]
        try:
            q = Observations.query_criteria(
                obs_collection="IUE", dataproduct_type="spectrum",
                s_ra=[ra - rad, ra + rad], s_dec=[dec - rad, dec + rad])
        except Exception as exc:
            print("  %-14s QUERY FAILED %s" % (name, type(exc).__name__), flush=True)
            continue
        d = q.to_pandas() if len(q) else pd.DataFrame()
        cams = dict(d["instrument_name"].value_counts()) if len(d) else {}
        # only the cameras that can see C IV at this redshift are useful
        useful = {c: int(n) for c, n in cams.items() if c in wants}
        texp_useful = float(d[d["instrument_name"].isin(wants)]["t_exptime"].sum()) if len(d) else 0.0
        rows.append(dict(
            row_num=int(r["row_num"]), common_name=name, z=z,
            civ_observed_A=round(civ_obs, 1),
            camera_for_civ="/".join(sorted(wants)) or "NONE",
            v22_cameras=str(r["instrument"]),
            n_obs_total=int(len(d)),
            n_obs_useful=int(sum(useful.values())),
            cameras_found="/".join("%s:%d" % (k, v) for k, v in sorted(cams.items())) or "-",
            texp_useful_s=round(texp_useful, 0),
            in_amogh_set=int(name in ALREADY)))
        print("  %-14s z=%.4f  C IV at %6.1f A -> %-7s | MAST %2d obs (%s) | useful %2d, %.0f s"
              % (name, z, civ_obs, rows[-1]["camera_for_civ"], len(d),
                 rows[-1]["cameras_found"], rows[-1]["n_obs_useful"], texp_useful), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("\n%d objects -> %s" % (len(out), OUT))
    if len(out):
        print("  with NO IUE observation at all      : %d" % (out["n_obs_total"] == 0).sum())
        print("  with no camera covering C IV        : %d" % (out["camera_for_civ"] == "NONE").sum())
        print("  with no USEFUL (C IV-covering) obs  : %d" % (out["n_obs_useful"] == 0).sum())


if __name__ == "__main__":
    main()
