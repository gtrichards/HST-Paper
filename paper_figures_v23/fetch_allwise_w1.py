"""
AllWISE W1 magnitudes for the HST, SDSS (Rankine+20) and GNIRS-DQS samples,
for the L_WISE panels of Figs 5 and 6.

Trevor's L_WISE panels used unWISE W1 for SDSS (a Rankine file matched to
unWISE that is no longer available) and W1 for HST from his own table.  Here
all three samples are matched to the same catalogue -- AllWISE
(allwise_p3as_psd at IRSA) -- in the same way: the nearest AllWISE source
within MATCH_RADIUS of each position.

Samples and positions:
  HST    all 474 rows of CIV_measurements_v23.csv (designation -> RA, Dec)
  SDSS   Rankine+20 rows with good C IV and LOGL1350_dr16 > 0 (the Fig 5a/6a set)
  GNIRS  all rows of gnirsdqs_wRankine_wdr16Lum.csv

Output: Data/AllWISE_W1_v23.csv with
  sample, key (HST row_num / SDSS_NAME / GNIRS index), ra, dec,
  w1mpro, w1sigmpro (Vega mag), sep_arcsec (blank where nothing matched)

Needs network access to IRSA; run once, the CSV is committed.
Usage:  python fetch_allwise_w1.py
"""

import os
import time
import numpy as np
import pandas as pd
import pyvo
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from hst_sample import designation_to_coord

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Data")
OUT = os.path.join(DATA, "AllWISE_W1_v23.csv")
MATCH_RADIUS = 3.0          # arcsec
CHUNK = 5000                # positions per IRSA upload (20000 hit IRSA's gateway timeout)
#: Each finished chunk is saved here, so a failed run resumes where it stopped.
#: Not part of the repository; delete it to force a fresh query.
CACHE = os.environ.get("ALLWISE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "allwise_w1_v23"))
TAP = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
QUERY = """SELECT my_table.in_id, a.ra, a.dec, a.w1mpro, a.w1sigmpro
FROM TAP_UPLOAD.my_table AS my_table, allwise_p3as_psd AS a
WHERE 1=CONTAINS(POINT('ICRS',a.ra,a.dec),
                 CIRCLE('ICRS',my_table.in_ra,my_table.in_dec,%f/3600.))""" % MATCH_RADIUS


def positions():
    """All positions to match: DataFrame with sample, key, ra, dec."""
    hst = pd.read_csv(os.path.join(DATA, "CIV_measurements_v23.csv"))
    c = SkyCoord([designation_to_coord(d) for d in hst["designation"]])
    out = [pd.DataFrame({"sample": "HST", "key": hst["row_num"].astype(str),
                         "ra": c.ra.deg, "dec": c.dec.deg})]
    t = Table.read(os.path.join(DATA, "Rankine20_CIV_HeII_wDelGI_wdr16luminosity.fits"))
    s = t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()
    s = s[(s["LOGL1350_dr16"] > 0) & (s["good"] == True)]
    out.append(pd.DataFrame({"sample": "SDSS",
                             "key": s["SDSS_NAME"].astype(str).str.strip("b'").values,
                             "ra": s["RA"].values, "dec": s["Dec"].values}))
    g = pd.read_csv(os.path.join(DATA, "gnirsdqs_wRankine_wdr16Lum.csv"))
    out.append(pd.DataFrame({"sample": "GNIRS", "key": g.index.astype(str),
                             "ra": g["RA"].values, "dec": g["DEC"].values}))
    return pd.concat(out, ignore_index=True)


def match_chunk(p):
    """Nearest AllWISE source for each row of p (one IRSA upload, retried)."""
    up = Table({"in_id": np.arange(len(p), dtype=np.int32),
                "in_ra": p["ra"].values.astype(float),
                "in_dec": p["dec"].values.astype(float)})
    for attempt in range(6):
        try:
            r = TAP.run_async(QUERY, uploads={"my_table": up}).to_table().to_pandas()
            break
        except Exception as e:                      # IRSA drops connections at times
            print("    attempt %d failed: %s" % (attempt + 1, str(e)[:100]))
            time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError("IRSA query failed 6 times")
    # nearest match per input position
    i = r["in_id"].values
    sep = SkyCoord(r["ra"].values * u.deg, r["dec"].values * u.deg).separation(
        SkyCoord(p["ra"].values[i] * u.deg, p["dec"].values[i] * u.deg)).arcsec
    r["sep_arcsec"] = sep
    r = r.sort_values("sep_arcsec").drop_duplicates("in_id")
    out = p.reset_index(drop=True).copy()
    for col in ["w1mpro", "w1sigmpro", "sep_arcsec"]:
        out[col] = np.nan
        out.loc[r["in_id"].values, col] = r[col].values
    return out


def main():
    pos = positions()
    print("positions: %s" % pos["sample"].value_counts().to_dict())
    os.makedirs(CACHE, exist_ok=True)
    done = []
    for k in range(0, len(pos), CHUNK):
        f = os.path.join(CACHE, "chunk_%06d_%d.csv" % (k, CHUNK))
        if os.path.exists(f):                     # finished on an earlier run
            done.append(pd.read_csv(f, dtype={"key": str}))
            continue
        t0 = time.time()
        res_k = match_chunk(pos.iloc[k:k + CHUNK])
        res_k.to_csv(f, index=False)
        done.append(res_k)
        print("  rows %d-%d matched in %.0fs" % (k, min(k + CHUNK, len(pos)), time.time() - t0), flush=True)
    res = pd.concat(done, ignore_index=True)
    res.to_csv(OUT, index=False)
    for s, g in res.groupby("sample"):
        print("  %-5s %6d positions, %6d with AllWISE W1 within %.0f\"" %
              (s, len(g), g["w1mpro"].notna().sum(), MATCH_RADIUS))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
