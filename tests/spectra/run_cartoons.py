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
        # measure the line directly off the product, against a local continuum
        # from windows either side -- no ICA, so this tests the REDUCTION alone
        sel = g & (w >= 1400) & (w <= 1700)
        wing = sel & (((w >= 1400) & (w <= 1480)) | ((w >= 1640) & (w <= 1700)))
        sl, ic = np.polyfit(w[wing], f[wing], 1)
        k = sel & (w >= 1500) & (w <= 1600)
        cont = sl * w[k] + ic
        dl = np.gradient(w[k])
        ew = float(np.sum((f[k] - cont) / cont * dl))
        line = np.clip(f[k] - cont, 0, None)
        cum = np.cumsum(line * dl)
        lam = float(np.interp(cum[-1] / 2.0, cum, w[k]))
        blue = (CIV - lam) / CIV * 299792.458
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
    c = res.get("cartoon_nine_disagreeing")
    if b and c:
        print("      nine DISagreeing      median error %.4e   scatter/error %.2f"
              % (c["med_err"], c["ratio"]))
        print("      its error should EXCEED the agreeing case, because the "
              "exposures really do differ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
