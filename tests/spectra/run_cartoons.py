"""Do the cartoons come back with the answer they were built with?

Unlike the real-spectrum fixtures, these have a known truth, so this script
checks RECOVERY rather than stability.  Two questions per cartoon:

  1. does the measured C IV blueshift and EW match what was injected?
  2. is the reported error honest -- does scatter/error come out near 1?

The second is the one that settles the co-add's error estimator.  Nine
exposures of the same truth must combine to sigma/3; nine that disagree by 20
per cent must NOT, because their disagreement is real and the combined value is
less certain than the formal error implies.

    python run_cartoons.py
"""
import glob
import json
import os
import shutil
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PAPER)
sys.path.insert(0, os.path.join(PAPER, "finalrun"))
CART = os.path.join(HERE, "cartoons")
CIV = 1549.48


def run(name):
    from astropy.table import Table
    from rebinning import coadd as CO
    meta = json.load(open(os.path.join(CART, name, "fixture.json")))
    z = meta["z"]
    work = tempfile.mkdtemp(prefix="cartoon_")
    try:
        stage, reb = os.path.join(work, "s"), os.path.join(work, "r")
        os.makedirs(stage); os.makedirs(reb)
        for p in glob.glob(os.path.join(CART, name, "exposures", "*")):
            shutil.copy2(p, stage)
        CO.rebin(meta["obj"], z, "FOS", None, data_path=stage, output_dir=reb, flat=True)
        p = sorted(glob.glob(os.path.join(reb, "*.fits")))
        if not p:
            return None
        t = Table.read(p[0])
        w = np.asarray(t["Rest-frame Wavelength"], float)
        f = np.asarray(t["Coadded Flux (Arbitrary Units)"], float)
        e = np.asarray(t["Coadded Flux Errors"], float)
        m = np.asarray(t["Bad Pixel Mask"], float)
        cn = np.asarray(t["Continuum Normalisation"], float)
        g = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)
        # measure the line directly off the product with the measurement's own
        # prescription (run_ica.civ_measure: power-law continuum through the
        # 1445-1465 / 1700-1705 A medians, clamped cumulative EW over 1500-1600 A,
        # half-flux pixel by span) -- no ICA, so this tests the REDUCTION alone.
        # Until 2026-11-17 this used its own convention (np.interp of the
        # cumulative at pixel centres, wide 1400-1480 / 1640-1700 wings), which
        # read +17 to +107 km/s high on a +1000 km/s truth for reasons of its own.
        from ica.run_ica import civ_measure
        sel = g & (w >= 1400) & (w <= 1720)
        blue, ew, _, _, _ = civ_measure(w[sel], f[sel])
        # is the error honest?
        lf = g & (w >= 1640) & (w <= 1700)
        x = f[lf]
        d2 = x[2:] - 2 * x[1:-1] + x[:-2]
        ratio = float((np.nanstd(d2) / np.sqrt(6.0)) / np.nanmedian(e[lf]))
        return dict(blue=blue, ew=ew, ratio=ratio,
                    n_usable=int(g.sum()), n_masked=int((m != 0).sum()),
                    med_err=float(np.nanmedian(e[g])), meta=meta)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    names = sorted(os.path.basename(d) for d in glob.glob(os.path.join(CART, "*"))
                   if os.path.isdir(d))
    print("CARTOON VALIDATION   truth: blueshift +1000 km/s, EW 40 A")
    print("  %-30s %10s %9s %8s %9s %7s" %
          ("cartoon", "blueshift", "EW", "d_blue", "scat/err", "masked"))
    res = {}
    for n in names:
        r = run(n)
        if r is None:
            print("  %-30s  no product" % n)
            continue
        res[n] = r
        t = r["meta"]["truth"]
        print("  %-30s %+10.1f %9.2f %+8.1f %9.2f %7d"
              % (n, r["blue"], r["ew"], r["blue"] - t["blueshift_kms"],
                 r["ratio"], r["n_masked"]))
    print()
    print("  d_blue is the error in the recovered blueshift, in km/s "
          "(one lattice pixel is 69.03)")
    print("  scat/err near 1.0 means the reported error matches the data's own noise")
    print()
    a, b = res.get("cartoon_single_clean"), res.get("cartoon_nine_agreeing")
    if a and b:
        print("  *** THE CO-ADD ERROR QUESTION ***")
        print("      one exposure          median error %.4e   scatter/error %.2f"
              % (a["med_err"], a["ratio"]))
        print("      nine agreeing         median error %.4e   scatter/error %.2f"
              % (b["med_err"], b["ratio"]))
        print("      error ratio one/nine  %.2f      -- should be 3.00 if the co-add "
              "takes the sqrt(N) gain" % (a["med_err"] / b["med_err"]))
    # CORRECTED 2026-11-14.  This used to say of cartoon_nine_disagreeing that
    # "its error should EXCEED the agreeing case, because the exposures really
    # do differ".  They do not.  coadd.py divides each exposure by its OWN
    # continuum before combining, which removes a pure level offset exactly:
    # measured, the declared 20 per cent spread becomes 0.7 per cent and
    # chi2/dof in the line core is 1.01 against 0.96 for the agreeing case.
    # cartoon_nine_varying_line is the one that actually disagrees.
    c = res.get("cartoon_nine_disagreeing")
    if b and c:
        print("      nine level-offset     median error %.4e   scatter/error %.2f"
              % (c["med_err"], c["ratio"]))
        print("      its error should MATCH the agreeing case: the per-exposure")
        print("      continuum division removes a level offset exactly")
    d = res.get("cartoon_nine_varying_line")
    if b and d:
        print("      nine VARYING LINE     median error %.4e   scatter/error %.2f"
              % (d["med_err"], d["ratio"]))
        print("      the only cartoon whose exposures still disagree after")
        print("      normalisation (chi2/dof 19.8 in the core, 1.01 outside), so")
        print("      the only one that exercises the chi2 arm of _combined_error")
    return 0


if __name__ == "__main__":
    sys.exit(main())
