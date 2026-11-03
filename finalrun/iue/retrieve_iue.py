"""Retrieve the IUE extracted spectra for the manmask-1 objects.

GTR wants to know why 19 of our 22 IUE-only objects are absent from Amogh's
"IUE good" set, to spot check the three that are in it, and to make the
measurements he did not.  survey_iue.py already established that the absence is
NOT missing data: every one of the 22 has IUE exposures covering C IV at its
redshift, and the cut is not a simple exposure threshold either -- PG1310-1051
is in his set on a single 7 ks exposure while WAS 49, with 15 ks, is not.

All 22 are retrieved, including his three, because reproducing those is how his
masking gets checked.

Only the cameras that can see C IV at the object's redshift are fetched: SWP
(about 1150-1980 A) below z = 0.28, LWP/LWR (about 1850-3350 A) above z = 0.19.
MAST serves IUE as one extracted-spectrum FITS per exposure, about 50 kB each,
so the whole retrieval is a few MB.

Writes iue/spectra/<COMMON_NAME>/ and iue/iue_products.csv recording what was
fetched, so the provenance of anything measured from these is on disk.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from astroquery.mast import Observations

HERE = os.path.dirname(os.path.abspath(__file__))
SURVEY = os.path.join(HERE, "iue_survey.csv")
OUTDIR = os.path.join(HERE, "spectra")
MANIFEST = os.path.join(HERE, "iue_products.csv")
RADIUS = 5.0 / 3600.0
CIV = 1549.48
CAM = {"SWP": (1150.0, 1980.0), "LWP": (1850.0, 3350.0), "LWR": (1850.0, 3350.0)}


def designation_to_coord(d):
    s = str(d).strip().lstrip("J")
    i = max(s.rfind("+"), s.rfind("-"))
    ra_s, de_s = s[:i], s[i:]
    ra = (float(ra_s[0:2]) + float(ra_s[2:4]) / 60.0 + float(ra_s[4:]) / 3600.0) * 15.0
    sign = -1.0 if de_s[0] == "-" else 1.0
    de_s = de_s[1:]
    return ra, sign * (float(de_s[0:2]) + float(de_s[2:4]) / 60.0 + float(de_s[4:]) / 3600.0)


def safe(name):
    return "".join(c if (c.isalnum() or c in "+-._") else "_" for c in str(name))


def main():
    tab = pd.read_csv("/Users/gtr/Work/git/HST-Paper/Data/CIV_measurements_v23.csv")
    m1 = tab[tab["CIVgood_manmask"] == 1].sort_values("row_num")
    only = set(sys.argv[1:])
    rows = []
    for _, r in m1.iterrows():
        name = str(r["common_name"])
        if only and name not in only:
            continue
        z = float(r["z"])
        civ_obs = CIV * (1.0 + z)
        wants = [c for c, (lo, hi) in CAM.items() if lo <= civ_obs <= hi]
        ra, dec = designation_to_coord(r["designation"])
        try:
            o = Observations.query_criteria(
                obs_collection="IUE", dataproduct_type="spectrum",
                s_ra=[ra - RADIUS, ra + RADIUS], s_dec=[dec - RADIUS, dec + RADIUS])
        except Exception as exc:
            print("  %-14s query failed: %s" % (name, type(exc).__name__), flush=True)
            continue
        keep = [i for i, c in enumerate(o["instrument_name"]) if c in wants]
        if not keep:
            print("  %-14s no camera covering C IV" % name, flush=True)
            continue
        o = o[keep]
        d = os.path.join(OUTDIR, safe(name))
        os.makedirs(d, exist_ok=True)
        try:
            prods = Observations.get_product_list(o)
            man = Observations.download_products(
                prods, download_dir=d, productType="SCIENCE", cache=True)
        except Exception as exc:
            print("  %-14s download failed: %s" % (name, type(exc).__name__), flush=True)
            continue
        got = [p for p in (man["Local Path"] if man is not None else [])]
        for p in got:
            rows.append(dict(common_name=name, row_num=int(r["row_num"]), z=z,
                             civ_observed_A=round(civ_obs, 1),
                             cameras=" ".join(sorted(wants)), path=p))
        print("  %-14s %2d exposures -> %s" % (name, len(got), os.path.relpath(d, HERE)),
              flush=True)
    pd.DataFrame(rows).to_csv(MANIFEST, index=False)
    print("\n%d files -> %s" % (len(rows), MANIFEST))


if __name__ == "__main__":
    main()
