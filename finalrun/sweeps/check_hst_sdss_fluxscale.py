"""Is the HST absolute flux scale right?  Compare it against SDSS at 2500 A.

GTR said repeatedly that the HST luminosities looked too high against SDSS, and
they were: spec_morph.morph2 multiplied the flux by its morph coefficient and the
Continuum Normalisation column never divided it out, so `cn x arb` over-counted
the continuum once (sweeps/fix_luminosity_morph.py).  This is the independent
check on the repair, and the one that speaks to his actual question.

It is independent in the way that matters.  The earlier HST/SDSS agreement that
this project recorded was measured INSIDE a spliced product, where both halves
had been through morph2 and the common factor cancelled -- which is exactly why
it reported agreement while the scale was a factor of three out.  Here the SDSS
side never touches this pipeline: it is read straight from the survey's own lite
file, whose flux is absolutely calibrated in units of 1e-17 erg/s/cm2/A, and
compared against the HST product's `cn x arb`.  A common-mode error cannot hide.

2500 A is the only window where the two overlap for a useful number of objects
(137 SDSS spectra reach it), so the comparison is made there.

Expect scatter and not identity.  These are different instruments, different
apertures (an HST slit against a 3 inch fibre) and different epochs on variable
sources, so the spread is real astrophysics plus real calibration differences.
The question is only whether the RATIO sits near 1 or near 3.

    python sweeps/check_hst_sdss_fluxscale.py                     # corrected tree
    python sweeps/check_hst_sdss_fluxscale.py --tree rebin        # the old one
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDSS_DIR = os.path.join(HERE, "data_v23", "SDSS_spec", "lite")
SDSS_TABLE = os.path.join(HERE, "pipeline_output", "final", "sdss_spectra.csv")
FITS = os.path.join(HERE, "pipeline_output", "chosen_visit", "fits")
WIN = (2490.0, 2510.0)

#: Narrower than this and the two medians are comparing noise, not flux.
MIN_OVERLAP = 20.0


def hst_name(index):
    """The product Identifier for this index, taken from its adopted record."""
    for d in sorted(glob.glob(os.path.join(FITS, "%03d_*" % index))):
        for rec in sorted(glob.glob(os.path.join(d, "records", "BEST_*.json"))):
            try:
                nm = json.load(open(rec)).get("name")
            except Exception:
                continue
            if isinstance(nm, str) and nm.strip():
                return nm
    return None


def sdss_flux(path, z, lo, hi):
    with fits.open(path) as h:
        d = h[1].data
        f = np.asarray(d["flux"], float) * 1e-17
        try:
            lam = 10.0 ** np.asarray(d["loglam"], float)
        except Exception:
            c0, c1 = h[0].header["COEFF0"], h[0].header["COEFF1"]
            lam = 10.0 ** (c0 + c1 * np.arange(len(f)))
        try:
            iv = np.asarray(d["ivar"], float)
        except Exception:
            iv = np.ones_like(f)
    rest = lam / (1.0 + z)
    m = (rest >= lo) & (rest <= hi) & np.isfinite(f) & (iv > 0)
    return (float(np.nanmedian(f[m])), int(m.sum())) if m.sum() > 3 else (np.nan, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="rebin_morphfix",
                    help="rebin tree under pipeline_output/chosen_visit/")
    ap.add_argument("--overlap", action="store_true",
                    help="compare over each object's own HST/SDSS overlap region "
                         "instead of the fixed 2490-2510 A window.  Only 8 objects "
                         "have HST data in the 2500 A window AND an SDSS spectrum; "
                         "using whatever stretch the two actually share -- at "
                         "least MIN_OVERLAP wide, both sides unmasked -- uses far "
                         "more of them, at the cost of comparing each object at a "
                         "different wavelength.")
    ap.add_argument("--max-rest", type=float, default=0.0,
                    help="ignore overlap redward of this rest wavelength.  Use "
                         "3000: the ICA components and the morph reference both "
                         "end near 3000 A, so an overlap at 3000-7200 A rest -- "
                         "which is where the low-z objects' FOS red gratings meet "
                         "SDSS -- is being compared outside the range the fit and "
                         "the continuum column mean anything in.")
    a = ap.parse_args()
    tree = os.path.join(HERE, "pipeline_output", "chosen_visit", a.tree)

    t = pd.read_csv(SDSS_TABLE)
    if not a.overlap:
        t = t[t.covers_2500 == True]
    rows = []
    for _, r in t.iterrows():
        try:
            i = int(r["index"])
        except Exception:
            continue
        sp = os.path.join(SDSS_DIR, str(r["fn_sdss"]))
        if not os.path.exists(sp):
            continue
        nm = hst_name(i)
        if not nm:
            continue
        hp = os.path.join(tree, "%s.fits" % nm)
        if not os.path.exists(hp):
            continue
        tb = Table.read(hp)
        w = np.asarray(tb["Rest-frame Wavelength"], float)
        fa = np.asarray(tb["Coadded Flux (Arbitrary Units)"], float)
        cn = np.asarray(tb["Continuum Normalisation"], float)
        mk = np.asarray(tb["Bad Pixel Mask"], float)
        z = float(np.nanmedian(np.asarray(tb["Redshift"], float)))
        if a.overlap:
            good = np.isfinite(fa) & np.isfinite(cn) & (mk == 0)
            if not good.any():
                continue
            lo = max(float(np.nanmin(w[good])), float(r["rest_min"]))
            hi = min(float(np.nanmax(w[good])), float(r["rest_max"]))
            if a.max_rest:
                hi = min(hi, a.max_rest)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < MIN_OVERLAP:
                continue
        else:
            lo, hi = WIN
        m = ((w >= lo) & (w <= hi) & np.isfinite(fa) & np.isfinite(cn) & (mk == 0))
        if m.sum() <= 3:
            continue
        fh = float(np.nanmedian(fa[m] * cn[m]))
        fs, nsd = sdss_flux(sp, z, lo, hi)
        if not (np.isfinite(fh) and np.isfinite(fs) and fs > 0 and fh > 0):
            continue
        rows.append(dict(index=i, name=nm, z=z, lo=round(lo, 1), hi=round(hi, 1),
                         f_hst=fh, f_sdss=fs, ratio=fh / fs,
                         npix_hst=int(m.sum()), npix_sdss=nsd))

    if not rows:
        print("no objects with both an HST 2500 A window and an SDSS spectrum")
        return
    d = pd.DataFrame(rows)
    rat = d.ratio.values
    lr = np.log10(rat)
    print("tree: %s" % a.tree)
    print("objects compared (%s): %d"
          % ("own overlap region" if a.overlap else "2490-2510 A", len(d)))
    if a.overlap:
        print("  overlap width: median %.0f A   comparison wavelength %.0f-%.0f A"
              % (np.median(d.hi - d.lo), d.lo.min(), d.hi.max()))
    print("  HST / SDSS flux ratio : median %.3f   16-84%% %.3f - %.3f"
          % (np.median(rat), np.percentile(rat, 16), np.percentile(rat, 84)))
    print("  in dex                : median %+.3f   scatter (sd) %.3f"
          % (np.median(lr), np.std(lr)))
    print("  within a factor 1.5   : %d of %d" % (int(np.sum((rat > 1/1.5) & (rat < 1.5))), len(d)))
    out = os.path.join(HERE, "pipeline_output", "final",
                       "hst_sdss_fluxscale_%s%s.csv"
                       % (a.tree, ("_overlap%s" % (("_lt%d" % a.max_rest)
                                                    if a.max_rest else ""))
                          if a.overlap else ""))
    d.sort_values("index").to_csv(out, index=False)
    print("  wrote %s" % out)


if __name__ == "__main__":
    main()
