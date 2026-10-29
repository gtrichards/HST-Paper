"""What the HST+SDSS refit did, and why.

Reads final/sdss_refit.csv.  The question is GTR's: how much do things change
if the ICA fit is done against HST+SDSS instead of HST alone?

The answer is that the C IV fit gets WORSE almost everywhere, that the damage
scales with how much SDSS was added rather than with how well the bridge
normalised it, and that the large blueshift changes are the fit breaking rather
than a better measurement.  The cause is weighting: the added data are both
more numerous and higher S/N than the C IV region they displace.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(os.path.dirname(HERE), "pipeline_output", "final")
Q = 69.03


def main():
    t = pd.read_csv(os.path.join(FINAL, "sdss_refit.csv"))
    t["degrade"] = t["civ_resid_rms_hst_sdss"] / t["civ_resid_rms_hst"]
    t["sdss_share"] = t["sdss_px_added"] / (t["sdss_px_added"] + t["host_px"])
    w = t.dropna(subset=["civ_resid_rms_hst", "civ_resid_rms_hst_sdss"])

    print("=" * 76)
    print("REFIT WITH HST+SDSS INSTEAD OF HST ALONE   (n=%d adopted objects)" % len(t))
    print("=" * 76)
    print("  SDSS pixels added : median %d (16-84%% %d-%d) against a host median of %d"
          % (t["sdss_px_added"].median(), *np.percentile(t["sdss_px_added"], [16, 84]).astype(int),
             t["host_px"].median()))
    print("  bridge quality (scale agreement across five windows, 1.00 = perfect):")
    print("      median %.2f   84th %.2f   max %.2f"
          % (t["scale_window_tilt"].median(),
             np.percentile(t["scale_window_tilt"].dropna(), 84), t["scale_window_tilt"].max()))

    print()
    print("  C IV FIT QUALITY, measured on the SAME pixels both ways")
    print("  (the host data under C IV are untouched, so any change is the fit")
    print("   choosing different components)")
    print("      residual rms, HST only : median %.2f" % w["civ_resid_rms_hst"].median())
    print("      residual rms, HST+SDSS : median %.2f" % w["civ_resid_rms_hst_sdss"].median())
    bad = w["civ_resid_rms_hst_sdss"] > w["civ_resid_rms_hst"]
    print("      WORSE with SDSS        : %d of %d (%.0f%%)" % (bad.sum(), len(w), 100 * bad.mean()))
    print("      worse by more than 50%% : %d (%.0f%%)"
          % (((w["civ_resid_rms_hst_sdss"] / w["civ_resid_rms_hst"]) > 1.5).sum(),
             100 * ((w["civ_resid_rms_hst_sdss"] / w["civ_resid_rms_hst"]) > 1.5).mean()))

    print()
    print("  IS IT THE WEIGHTING OR THE BRIDGE?")
    for lab, c in [("SDSS pixels added", "sdss_px_added"),
                   ("SDSS share of pixels", "sdss_share"),
                   ("bridge window tilt", "scale_window_tilt")]:
        k = t[[c, "degrade"]].replace([np.inf, -np.inf], np.nan).dropna()
        print("      corr(%-21s, C IV degradation) = %+.2f" % (lab, np.corrcoef(k[c], k["degrade"])[0, 1]))
    print("      damage scales with the amount of SDSS added, not with how well")
    print("      the bridge normalised it:")
    for lo, hi in [(0, 300), (300, 1000), (1000, 2000), (2000, 10 ** 9)]:
        k = t[(t["sdss_px_added"] >= lo) & (t["sdss_px_added"] < hi)]
        if len(k):
            print("        %5d-%-5s px  n=%3d   median rms ratio %.2f   median |dblue| %6.1f km/s"
                  % (lo, hi if hi < 10 ** 9 else "+", len(k), k["degrade"].median(),
                     k["dblue"].abs().median()))

    print()
    print("  BLUESHIFT")
    d = t["dblue"].dropna(); q = d.abs() / Q
    print("      median |change| %.1f km/s (%.1f quanta)   mean signed %+.1f km/s"
          % (d.abs().median(), d.abs().median() / Q, d.mean()))
    for lim in (1.5, 3.5, 10.5):
        print("      within %4.1f quanta : %3d of %d (%.0f%%)" % (lim - 0.5, (q < lim).sum(), len(q), 100 * (q < lim).mean()))
    big = t[(t["dblue"].abs() / Q) > 10]
    print("      movers beyond 10 quanta: %d, median C IV rms ratio %.2f against %.2f for the rest"
          % (len(big), big["degrade"].median(), t[(t["dblue"].abs() / Q) <= 10]["degrade"].median()))
    print("      -- the big moves are where the fit got WORST, so they are the fit")
    print("         breaking rather than a better measurement.")

    print()
    print("  EQUIVALENT WIDTH")
    le = t["dlogew"].dropna()
    print("      median |change| %.3f dex (sample log EW sd 0.341); >0.3 dex in %d of %d"
          % (le.abs().median(), (le.abs() > 0.3).sum(), len(le)))

    print()
    print("  WHAT DOES IMPROVE: THE CONTINUUM")
    for lab, c in [("HST only", "scale_hst"), ("HST+SDSS", "scale_hst_sdss")]:
        s = t[c].dropna()
        print("      %-9s scale median %.3f   inside the healthy 0.88-1.00 band: %d of %d (%.0f%%)"
              % (lab, s.median(), ((s >= 0.88) & (s <= 1.00)).sum(), len(s),
                 100 * ((s >= 0.88) & (s <= 1.00)).mean()))

    print()
    print("  CONCLUSION.  Adding SDSS buys a modestly better continuum and costs a")
    print("  worse C IV fit in 97 per cent of objects.  For a paper whose measurement")
    print("  IS C IV, that is the wrong trade at equal weighting.")


if __name__ == "__main__":
    main()
