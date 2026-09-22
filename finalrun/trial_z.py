"""
Is a model-vs-data offset at the anchor lines a redshift error, or the model?

On NGC 3516 the reconstruction sits redward of the data by ~810 km/s at C III]
and ~440 km/s at Mg II. The external redshift is not in doubt -- catalogue
0.0088 and NED 0.008836 agree to 11 km/s -- but the two anchors lean the same
way, which is what a redshift error would look like, so the question deserves a
test rather than an argument.

The test: refit the same object at a grid of trial redshifts and watch the two
anchor offsets. A genuine redshift error makes both go to zero at the same trial
value. A model-shape mismatch does not -- the offsets move together with the
shift applied but never both vanish, or vanish at different values.

The trial spectra are the object's combined FITS with rest wavelengths rescaled
by (1+z0)/(1+z_trial), written as temporary stems. The mask ranges are rescaled
by the same factor, because absorption features move with the data and a mask
left at its old wavelength would miss them. Each trial is also screened for
sub-continuum pixels in C IV, since a fit that fails that veto is not evidence of
anything.

No figures are written -- this is a table, not a review.
"""

import argparse
import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from astropy.io import fits

sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")
WORKDIR = os.path.join(OUTDIR, "coadd_work")
from rebin_path import REBIN as _REBIN_DEFAULT
REBIN = _REBIN_DEFAULT   # single spectra live here
from rebin_path import ITERDIR   # env HSTICA_ITERDIR overrides
C = 299792.458


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="combined stem, e.g. 'NGC 3516_COMBINED'")
    ap.add_argument("--record", required=True, help="iteration JSON whose masks to use")
    ap.add_argument("--dv", type=float, nargs="+",
                    default=[-800, -600, -400, -200, 0, 200, 400],
                    help="trial velocity offsets in km/s (+ve raises z)")
    args = ap.parse_args()

    from ica.manual_fix import ICAManualFixProcessor
    from model_data_shift import xcorr_shift
    from iterate_fit import civ_subcontinuum

    src = os.path.join(WORKDIR, args.name + ".fits")
    if not os.path.exists(src):
        src = os.path.join(REBIN, args.name + ".fits")   # a single spectrum, not a merge
    h = fits.open(src)
    z0 = float(h[1].data["Redshift"][0])
    ov = json.load(open(args.record))["override"]
    masks0 = ov.get("mask_ranges", [])

    from object_paths import folder_for
    outcsv = os.path.join(ITERDIR, folder_for(args.name), "records", "trial_z.csv")
    rows = []
    print("%-8s %-10s %9s %9s %9s %9s %8s %6s" % ("dv", "z_trial", "CIII]", "MgII", "CIV", "blue", "ew", "veto"))
    for dv in args.dv:
        z = (1 + z0) * (1 + dv / C) - 1
        scale = (1 + z0) / (1 + z)
        stem = "%s_trialz%+05.0f" % (args.name, dv)
        hh = fits.open(src)
        hh[1].data["Rest-frame Wavelength"][:] = hh[1].data["Rest-frame Wavelength"] * scale
        hh[1].data["Redshift"][:] = z
        hh.writeto(os.path.join(WORKDIR, stem + ".fits"), overwrite=True)
        masks = [[lo * scale, hi * scale] for lo, hi in masks0]

        proc = ICAManualFixProcessor(rebin_path=WORKDIR, master_mode=True)
        res = proc.fit_with_overrides(stem, mask_ranges=masks,
                                      unmask_ranges=ov.get("unmask_ranges", []),
                                      comps_use=ov.get("comps_use"))
        w, f, m = res["wave_arb"], res["flux_arb"], np.asarray(res["mask_arb"])
        model = np.interp(w, res["wave_ica"], res["flux_ica"], left=np.nan, right=np.nan)
        data = np.where(m == 0, f, np.nan)
        xc = {}
        for lab, lo, hi in (("CIII]", 1860, 1960), ("MgII", 2740, 2860), ("CIV", 1500, 1600)):
            v, n = xcorr_shift(w, data, model, lo, hi)
            xc[lab] = None if not np.isfinite(v) else float(v)
        sub = civ_subcontinuum(res)
        row = dict(dv=dv, z_trial=round(z, 6),
                   ciii_xcorr=xc["CIII]"], mgii_xcorr=xc["MgII"], civ_xcorr=xc["CIV"],
                   civ_blue=res["civ_blue"], civ_ew=res["civ_ew"],
                   veto="pass" if sub["ok"] else "REJECT(%d)" % sub["n_px"])
        rows.append(row)
        fmt = lambda v: "--" if v is None else "%+.0f" % v
        print("%+8.0f %-10.6f %9s %9s %9s %9.1f %8.2f %6s"
              % (dv, z, fmt(xc["CIII]"]), fmt(xc["MgII"]), fmt(xc["CIV"]),
                 res["civ_blue"], res["civ_ew"], row["veto"]), flush=True)
        os.remove(os.path.join(WORKDIR, stem + ".fits"))

    os.makedirs(os.path.dirname(outcsv), exist_ok=True)
    with open(outcsv, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print("\n(xcorr: +ve = model sits blueward of data, i.e. z too low)\n-> %s" % outcsv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
