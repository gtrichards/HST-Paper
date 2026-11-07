"""Does the HST rebin invent flux across flagged stretches, as the IUE one did?

REPORT ONLY.  GTR, 2026-11-06: "Yes, check, but don't take any action based on
the check yet."  Nothing here writes to a product or a ledger.

The IUE version of this (iue/check_gap_fill.py) found that lower_res_rebin
interpolates across stretches the native exposure flags bad instead of leaving a
hole, so five flagged pixels in swp26578 became the HIGHEST point of Mrk 1014's
C IV line and were drawn as data.  The fitter was already discounting them --
their errors inflate four- to fivefold -- so no measurement moved, but the plot
showed a spike-and-notch the model could not follow, and the fit looked broken.
GTR: "Amogh's fit is fine.  Because he simply ignores the missing pixels.  As
should we."

lower_res_rebin is shared by both paths, so the HST products may carry the same
thing, and that pass is complete and published.  This measures it.

THE TEST IS NECESSARILY WEAKER THAN THE IUE ONE and the difference matters when
reading the output.  For IUE each object has one or a few MXLO exposures with a
QUALITY column, so "is there native data behind this pixel" is exact.  An HST
object's chosen visit may stack many x1d exposures across segments and gratings,
each with its own DQ convention, so the comparison here is:

    a rebinned pixel is ORPHANED if no exposure contributing to its product has
    a good pixel within half that exposure's dispersion of it

Good means DQ == 0 where a DQ column exists, plus finite flux and a positive
error.  Where an instrument's DQ is missing the exposure contributes on flux and
error alone, which makes this an UNDERCOUNT: a pixel flagged only by DQ in such
a file will not be caught.  So a zero here is weaker evidence than a zero on the
IUE side, and a non-zero is real.

    python check_gap_fill_hst.py              # C IV window, adopted objects
    python check_gap_fill_hst.py --all        # every rebinned product
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
REBIN = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
STAGE = os.path.join(HERE, "pipeline_output", "chosen_visit", "stage")
FITS = os.path.join(HERE, "pipeline_output", "chosen_visit", "fits")
OUT = os.path.join(HERE, "pipeline_output", "final", "hst_gap_fill.csv")

CIV = (1500.0, 1600.0)
STEP = 2.0


def native_good(stage_dir, z):
    """Rest-frame wavelengths of good native pixels, and the median dispersion."""
    ws, disp = [], []
    pats = ("*_x1d.fits", "*_sx1.fits", "*_c1f.fits", "*_c0f.fits")
    files = []
    for p in pats:
        files += sorted(glob.glob(os.path.join(glob.escape(stage_dir), p)))
    for f in files:
        try:
            d = fits.getdata(f, 1)
        except Exception:
            continue
        names = getattr(d, "names", []) or []
        if "WAVELENGTH" in names and "FLUX" in names:
            rows = range(len(d))
            for r in rows:
                w = np.asarray(d["WAVELENGTH"][r], float)
                fl = np.asarray(d["FLUX"][r], float)
                er = (np.asarray(d["ERROR"][r], float) if "ERROR" in names
                      else np.full(len(w), 1.0))
                # NO DQ CUT -- this must match what the pipeline actually does.
                # read_spec_data.py imports Cut_Edge_Pix_TVM_NoDQ and says so in
                # its header: "The DQ==0 quality cut is removed entirely."  HST
                # keeps DQ-flagged pixels as data, so there is no gap for the
                # rebin to interpolate across, and a test that excludes them
                # invents orphans that the pipeline never produced.  Applying a
                # DQ==0 cut here reported 41 objects materially affected, 30-46
                # per cent of their line flux, almost all STIS G230L -- entirely
                # an artefact of the mismatch.  IUE is the opposite case:
                # read_iue_flat DOES mask QUALITY != 0, which is why the gap is
                # real there.
                g = np.isfinite(fl) & np.isfinite(er) & (er > 0)
                if g.sum() > 2:
                    ws.append(w[g] / (1.0 + z))
                    disp.append(float(np.median(np.diff(np.sort(w[g] / (1.0 + z))))))
    if not ws:
        return np.array([]), np.nan
    return np.sort(np.concatenate(ws)), (float(np.median(disp)) if disp else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    # adopted objects, by index, with the stage directory the fit used
    objs = []
    for rec in sorted(glob.glob(os.path.join(FITS, "*", "records", "BEST_*.json"))):
        i = int(re.search(r"fits/(\d+)_", rec).group(1))
        d = json.load(open(rec))
        nm = d.get("name", "")
        p = os.path.join(REBIN, "%s.fits" % nm)
        if not os.path.exists(p):
            continue
        inst = nm.rsplit("_", 1)[-1]
        sd = os.path.join(STAGE, "%03d_%s" % (i, inst))
        objs.append((i, nm, p, sd))
    print("adopted HST objects with a product and a stage directory: %d"
          % sum(1 for _, _, _, sd in objs if os.path.isdir(sd)))
    print()

    rows = []
    for i, nm, p, sd in objs:
        if not os.path.isdir(sd):
            continue
        t = Table.read(p)
        w = np.asarray(t["Rest-frame Wavelength"], float)
        f = np.asarray(t["Coadded Flux (Arbitrary Units)"], float)
        e = np.asarray(t["Coadded Flux Errors"], float)
        m = np.asarray(t["Bad Pixel Mask"], float)
        z = float(np.nanmedian(np.asarray(t["Redshift"], float)))
        g = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)
        nat, disp = native_good(sd, z)
        if not nat.size or not np.isfinite(disp):
            continue
        lo_lim, hi_lim = ((w[g].min(), w[g].max()) if a.all else CIV)
        sel = g & (w >= lo_lim) & (w <= hi_lim)
        if sel.sum() < 5:
            continue
        # The tolerance must be the LARGER of half a native dispersion and half
        # a rebinned pixel, and getting this wrong inverts the answer.  For IUE
        # the native dispersion (1.676 A) is COARSER than the 69.03 km/s lattice
        # (0.357 A at C IV), so half a native dispersion is the right scale.  For
        # HST it is far FINER -- COS G130M is about 0.01 A per pixel -- so half a
        # native dispersion is a 0.005 A tolerance and almost every rebinned pixel
        # sits further than that from any native one, whether or not there is data
        # behind it.  A first run with that tolerance reported 99 affected objects
        # with three of them "missing" 92-99 per cent of a line they visibly have.
        lat = float(np.median(np.diff(w))) if w.size > 2 else 0.357
        tol = 0.5 * max(disp, lat)
        j = np.clip(np.searchsorted(nat, w), 1, nat.size - 1)
        near = np.minimum(np.abs(w - nat[j - 1]),
                          np.abs(w - nat[np.minimum(j, nat.size - 1)]))
        orph = sel & (near > tol)
        if orph.any():
            civ = g & (w >= CIV[0]) & (w <= CIV[1])
            loc = float(np.nanmedian(f[civ])) if civ.any() else np.nan
            tot = float(np.nansum(np.clip(f[civ] - loc, 0, None))) if civ.any() else np.nan
            fr = (float(np.nansum(np.clip(f[orph & civ] - loc, 0, None)) / tot)
                  if tot and tot > 0 else 0.0)
            rows.append(dict(index=i, name=nm, orphan_px=int(orph.sum()),
                             civ_orphan_px=int((orph & civ).sum()),
                             frac_civ_line_flux=round(fr, 4),
                             lo=round(float(w[orph].min()), 1),
                             hi=round(float(w[orph].max()), 1)))
    d = pd.DataFrame(rows)
    if len(d):
        d = d.sort_values("frac_civ_line_flux", ascending=False)
        d.to_csv(OUT, index=False)
    print("objects with rebinned pixels that have no good native pixel behind them: %d"
          % len(d))
    if len(d):
        print()
        print("%-6s %-30s %9s %9s %10s  %s" % ("index", "product", "orphan", "in C IV",
                                               "% line flux", "span"))
        for _, r in d.head(20).iterrows():
            print("%-6d %-30s %9d %9d %9.1f%%  %.1f-%.1f"
                  % (r["index"], r["name"][:30], r.orphan_px, r.civ_orphan_px,
                     100 * r.frac_civ_line_flux, r.lo, r.hi))
        print()
        print("  materially affected (>5%% of the C IV line flux): %d"
              % int((d.frac_civ_line_flux > 0.05).sum()))
        print("  wrote %s" % OUT)
    print()
    print("REPORT ONLY -- nothing was changed.")


if __name__ == "__main__":
    main()
