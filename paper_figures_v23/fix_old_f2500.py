"""Correct the old catalogue's F2500 and F1450, and write them out for the
figures that still read it.

THE DEFECT.  HST_CIV_Sulentic2007_HSLA2018_finalprops_f2500AP.csv stores
F1450_spec and F2500/F2500_AP as the co-adder's CONTINUUM NORMALISATION at
that wavelength -- the erg/s/cm2/A scale factor -- without multiplying by the
spectrum.  Matched object by object against the v23 fit records:

    our continuum_scale alone / old F1450_spec : median +0.000 dex (n=190)
    our continuum_scale alone / old F2500_AP   : median -0.009 dex (n=73)

Agreement at the third decimal over that many objects is not a coincidence:
the old columns ARE the normalisation.  The full flux is

    f_lambda(obs) = f_lambda(model, fit units) x continuum normalisation

so the old values are low by the model value at that wavelength, which is 2.97
at 1450 A and 1.76 at 2500 A -- the quasar continuum shape, steeper to the blue
((1450/3000)^-1.5 = 2.98).  In luminosity that is -0.470 dex at 1450 and -0.236
dex at 2500, flat in redshift.

WHY THE FIX IS POSSIBLE FOR EVERY MATCHED OBJECT.  The model value f_lambda is
recorded whether or not our own continuum reaches that wavelength (it is the
shape, which the reconstruction always has).  So the correction uses THEIR
continuum and supplies only OUR missing factor:

    F_corrected = F_old x f_lambda(model)

Objects with no v23 fit (manmask 0 or 1) cannot be corrected and are written
with a blank, to be dropped rather than plotted wrong.

Output: Data/HST_old_catalogue_fluxes_corrected.csv
    Final_Name, F1450_corr, F2500_corr, f_lambda_1450, f_lambda_2500, matched
"""
import csv as _csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Data")
FR = "/Users/gtr/Work/projects/hstica/finalrun"
sys.path.insert(0, FR)
from object_paths import folder_for                      # noqa: E402
from hst_sample import load_adopted, designation_to_coord  # noqa: E402

OLD = os.path.join(DATA, "HST_CIV_Sulentic2007_HSLA2018_finalprops_f2500AP.csv")
OUT = os.path.join(DATA, "HST_old_catalogue_fluxes_corrected.csv")


def main():
    best = {}
    for p in glob.glob(os.path.join(FR, "pipeline_output", "chosen_visit",
                                    "fits", "*", "records", "BEST_*.json")):
        best[int(os.path.basename(os.path.dirname(os.path.dirname(p)))[:3])] = p
    q = {}
    for r in _csv.DictReader(open(os.path.join(FR, "pipeline_output", "work_queue.csv"))):
        for n in ([r["name_mast_key"] + "_COMBINED", r["name_mast_key"] + "_SPLICE"] +
                  [s for s in r["stems"].split(";") if s]):
            q[folder_key_cache(n)] = int(r["row_num"])

    dec = pd.read_csv(os.path.join(DATA, "decisions.csv"))
    model = {}
    for _, d in dec[dec["manmask"].isin([2, 3])].iterrows():
        i = int(d["folder"][:3]); rn = q.get(d["folder"])
        if i not in best or rn is None:
            continue
        mono = json.load(open(best[i]))["monochromatic"]
        model[rn] = (mono["1450"]["f_lambda"], mono["2500"]["f_lambda"])

    hst = load_adopted(DATA)
    old = pd.read_csv(OLD)
    ch = SkyCoord([designation_to_coord(x) for x in hst["designation"]])
    co = SkyCoord(old["_RA"].values * u.deg, old["_DE"].values * u.deg)
    idx, sep, _ = ch.match_to_catalog_sky(co)

    # old-catalogue row -> our row_num, for pairs within 5 arcsec
    old_to_row = {}
    for our_i, (oi, s) in enumerate(zip(idx, sep.arcsec)):
        if s < 5:
            old_to_row[int(oi)] = int(hst["row_num"].values[our_i])

    rows = []
    n_ok = 0
    for oi, r in old.iterrows():
        rn = old_to_row.get(int(oi))
        m = model.get(rn) if rn is not None else None
        if m is None:
            rows.append(dict(Final_Name=r["Final_Name"], F1450_corr="", F2500_corr="",
                             f_lambda_1450="", f_lambda_2500="", matched=0))
            continue
        f1450m, f2500m = m
        f1450 = r["F1450_spec (erg/s/cm2/A)"]
        f2500 = r["F2500_AP"]
        rows.append(dict(
            Final_Name=r["Final_Name"],
            F1450_corr=(f1450 * f1450m) if np.isfinite(f1450) and np.isfinite(f1450m) else "",
            F2500_corr=(f2500 * f2500m) if np.isfinite(f2500) and np.isfinite(f2500m) else "",
            f_lambda_1450=f1450m, f_lambda_2500=f2500m, matched=1))
        n_ok += 1
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote %s" % OUT)
    print("  old-catalogue rows      : %d" % len(old))
    print("  corrected (v23 matched) : %d" % n_ok)
    print("  left blank (no v23 fit) : %d" % (len(old) - n_ok))
    fm = [r["f_lambda_2500"] for r in rows if r["matched"]]
    print("  median model f_lambda: 1450 %.3f, 2500 %.3f"
          % (np.median([r["f_lambda_1450"] for r in rows if r["matched"]]), np.median(fm)))


_fk = {}
def folder_key_cache(name):
    if name not in _fk:
        _fk[name] = folder_for(name)
    return _fk[name]


if __name__ == "__main__":
    main()
