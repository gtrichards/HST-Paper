"""Co-add and rebin the IUE exposures through the SAME path as the HST data.

GTR: "Build the full path for me to do the measurements.  That way the HST and
IUE data will have been analyzed in the same way.  Amogh's work will then be a
double check."

So this does NOT implement its own co-addition.  It calls rebinning.coadd.rebin,
the same function the HST pass used, with data_origin="IUE".  Two additions were
needed in the shared package, both purely additive:

  rebinning/read_spec_data.py   read_iue_flat(), reading MAST's MXLO
                                single-row tables (WAVE/FLUX/SIGMA/QUALITY) into
                                the (waves, fluxes, errs, masks) arrays the HST
                                readers return.  QUALITY != 0 is masked.
  rebinning/small_pix_filter.py a filter width for IUE (11 px, about 2.7
                                resolution elements at 1.5 A/px and 6 A
                                resolution).  The existing widths are empirical,
                                so this is a judgement -- see the comment there.

`SpecCuts` is imported but never called in coadd.py (0 call sites; the comments
there say the imports are vestigial), so the missing IUE branch costs nothing.

The products land on the SAME 69.03 km/s log lattice as the HST files, with the
same six columns including Continuum Normalisation, so iterate_fit.py and the
manual-fix GUI read them with no special case:

    HSTICA_REBIN=<this output dir> python iterate_fit.py "<NAME>_IUE" ...

Output: iue/rebin/<NAME>_IUE.fits and iue/rebin_log.csv
"""
import os, sys, glob, warnings, io, contextlib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from rebinning import coadd as CO

TABLE = "/Users/gtr/Work/git/HST-Paper/Data/CIV_measurements_v23.csv"
SPECTRA = os.path.join(HERE, "spectra")
OUTDIR = os.path.join(HERE, "rebin")
LOG = os.path.join(HERE, "rebin_log.csv")


def safe(name):
    return "".join(c if (c.isalnum() or c in "+-._") else "_" for c in str(name))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    tab = pd.read_csv(TABLE)
    m1 = tab[tab["CIVgood_manmask"] == 1].sort_values("row_num")
    only = set(sys.argv[1:])
    rows = []
    for _, r in m1.iterrows():
        name = str(r["common_name"]); z = float(r["z"])
        if only and name not in only:
            continue
        path = os.path.join(SPECTRA, safe(name))
        if not os.path.isdir(path):
            print("  %-14s no retrieved spectra" % name, flush=True); continue
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                CO.rebin(name, z, "IUE", None, data_path=path,
                         output_dir=OUTDIR, flat=True)
        except Exception as exc:
            print("  %-14s REBIN FAILED: %s: %s"
                  % (name, type(exc).__name__, str(exc)[:90]), flush=True)
            rows.append(dict(common_name=name, row_num=int(r["row_num"]), z=z,
                             ok=0, note="%s: %s" % (type(exc).__name__, str(exc)[:120])))
            continue
        out = os.path.join(OUTDIR, "%s_IUE.fits" % name)
        if not os.path.exists(out):
            print("  %-14s produced no file" % name, flush=True)
            rows.append(dict(common_name=name, row_num=int(r["row_num"]), z=z,
                             ok=0, note="no output file"))
            continue
        d = fits.open(out)[1].data
        w = np.asarray(d["Rest-frame Wavelength"], float)
        f = np.asarray(d["Coadded Flux (Arbitrary Units)"], float)
        e = np.asarray(d["Coadded Flux Errors"], float)
        m = np.asarray(d["Bad Pixel Mask"], float)
        cn = np.asarray(d["Continuum Normalisation"], float)
        g = np.isfinite(f) & (m == 0) & (e > 0)
        kc = g & (w >= 1500) & (w <= 1600)
        ka = g & (w >= 1445) & (w <= 1465)
        rows.append(dict(
            common_name=name, row_num=int(r["row_num"]), z=z, ok=1,
            rest_lo=round(float(w[g].min()), 1), rest_hi=round(float(w[g].max()), 1),
            civ_px=int(kc.sum()),
            civ_sn=round(float(np.median(f[kc] / e[kc])), 2) if kc.sum() else np.nan,
            civ_blue_px=int((kc & (w < 1549.48)).sum()),
            civ_red_px=int((kc & (w >= 1549.48)).sum()),
            anchor_px=int(ka.sum()),
            cont_norm_ok=int((np.isfinite(cn) & (cn > 0)).sum()),
            note=""))
        q = rows[-1]
        print("  %-14s %4d-%4d A | C IV %3d px (%3d blue/%3d red) S/N %5.2f | 1450 %2d px"
              % (name, q["rest_lo"], q["rest_hi"], q["civ_px"], q["civ_blue_px"],
                 q["civ_red_px"], q["civ_sn"], q["anchor_px"]), flush=True)
    pd.DataFrame(rows).to_csv(LOG, index=False)
    ok = sum(r["ok"] for r in rows)
    print("\n%d of %d rebinned -> %s" % (ok, len(rows), OUTDIR))
    print("fit one with:")
    print('  HSTICA_REBIN=%s \\\n    python iterate_fit.py "<NAME>_IUE" --label iue' % OUTDIR)


if __name__ == "__main__":
    main()
