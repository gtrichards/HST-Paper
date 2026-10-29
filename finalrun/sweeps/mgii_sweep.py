"""Mg II sweep: does the C IV-driven reconstruction reproduce the Mg II line
that SDSS actually observed?

GTR deferred this to the end of the 474-object pass: "We will do a sweep of
MgII at the end.  I've tried not to be too biased by it, but we might want to
consider if/how things would change if we did actually incorporate the SDSS
data."

THE MEASUREMENT.  Mg II emission equivalent width over 2700-2900 A against a
LOCAL LINEAR CONTINUUM fitted to 2650-2700 and 2900-2950 A, measured the same
way on the SDSS data and on each component set's reconstruction.  Both are
measured against their own local continuum, so the arbitrary flux scale of the
ICA model drops out and the two equivalent widths are comparable.

THREE GUARDS, each from something that went wrong earlier in the pass:

  TRUNCATION.  Mg II truncated by the red end of the SDSS spectrum produced
  spurious cross-correlation velocities at indices 397 and 400-402.  Both
  continuum windows and the line window must be properly populated, and the
  spectrum must reach 2950 A rest, or the object is reported as GUARDED rather
  than measured.

  BALs.  Index 422: "It is a BAL and only low is ignoring (properly) the BAL."
  A reconstruction that correctly ignores absorption EXCEEDS the observed line,
  so a model-to-data ratio above 1 is the expected signature for a BAL, not
  evidence against the fit.  BAL candidates are ANNOTATED, not silently
  dropped, so the signature can be seen instead of assumed.

  A LINE THAT FILLS THE WINDOW.  At indices 463 and 465 the C IV line was so
  broad that any continuum anchor inside the integration window sat on its own
  wing.  The Mg II windows here are fixed and outside the line, but the same
  failure is checked for: if the two continuum anchors disagree by more than a
  factor of three the local continuum is not a continuum and the object is
  GUARDED.

Mg II is FITTED when it falls in the HST spectrum and is only a CHECK when it
appears solely in SDSS, so the sweep runs on the check sample and reports the
fitted ones separately.
"""
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
FINALRUN = os.path.dirname(HERE)
sys.path.insert(0, FINALRUN)
import object_diagnostics as diag

LITE = os.path.join(FINALRUN, "data_v23", "SDSS_spec", "lite")
SDSS_TABLE = os.path.join(FINALRUN, "pipeline_output", "final", "sdss_spectra.csv")
FITS_DIR = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "fits")
REBIN = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "rebin")
from rebin_path import DECISIONS

LINE = (2700.0, 2900.0)
BLUE = (2650.0, 2700.0)
RED  = (2900.0, 2950.0)
MIN_CONT_PX, MIN_LINE_PX = 15, 80
MAX_ANCHOR_RATIO = 3.0


def ew(w, f, good):
    """Mg II EW over LINE against the linear continuum through BLUE and RED.

    Returns (ew, reason).  reason is None when the measurement stands.
    """
    def anchor(lo, hi):
        k = good & (w >= lo) & (w <= hi)
        if k.sum() < MIN_CONT_PX:
            return np.nan, np.nan, int(k.sum())
        return float(np.mean(w[k])), float(np.median(f[k])), int(k.sum())
    xb, yb, nb = anchor(*BLUE)
    xr, yr, nr = anchor(*RED)
    k = good & (w >= LINE[0]) & (w <= LINE[1])
    if nb < MIN_CONT_PX or nr < MIN_CONT_PX:
        return np.nan, "continuum anchors thin (%d/%d px)" % (nb, nr)
    if k.sum() < MIN_LINE_PX:
        return np.nan, "line window thin (%d px)" % k.sum()
    if yb <= 0 or yr <= 0:
        return np.nan, "non-positive continuum anchor"
    ratio = max(yb / yr, yr / yb)
    if ratio > MAX_ANCHOR_RATIO:
        return np.nan, "anchors disagree by %.1fx -- not a continuum" % ratio
    cont = (w[k] - xb) * (yr - yb) / (xr - xb) + yb
    if np.any(cont <= 0):
        return np.nan, "continuum crosses zero inside the line"
    return float(np.trapezoid((f[k] - cont) / cont, w[k])), None


def sdss_mgii(fn, z):
    path = os.path.join(LITE, fn)
    if not os.path.exists(path):
        return np.nan, "SDSS file missing", np.nan
    with fits.open(path) as h:
        d = h[1].data
        lam = 10.0 ** np.asarray(d["loglam"], float)
        fl = np.asarray(d["flux"], float)
        iv = np.asarray(d["ivar"], float)
    w = lam / (1.0 + z)
    good = (iv > 0) & np.isfinite(fl)
    if w[good].max() < RED[1]:
        return np.nan, "SDSS stops at %.0f A rest, short of %.0f" % (w[good].max(), RED[1]), w[good].max()
    e, why = ew(w, fl, good)
    return e, why, float(w[good].max())


def model_mgii(stem, override, comps):
    res = diag.fit(stem, override, comps, REBIN)
    wi = np.asarray(res["wave_ica"], float)
    fi = np.asarray(res["flux_ica"], float)
    good = np.isfinite(fi)
    if wi[good].max() < RED[1]:
        return np.nan, "model stops at %.0f A" % wi[good].max()
    return ew(wi, fi, good)


def main():
    dec = pd.read_csv(DECISIONS)
    dec["index"] = dec["folder"].map(lambda f: int(str(f).split("_")[0]))
    adopted = dec[dec["manmask"].isin([2, 3])]
    sd = pd.read_csv(SDSS_TABLE)
    sd = sd[sd["fn_sdss"].notna()]
    sdmap = {int(r["index"]): r for _, r in sd.iterrows()}

    only = sys.argv[1:] or None
    rows = []
    for _, a in adopted.iterrows():
        i = int(a["index"])
        if only and str(i) not in only:
            continue
        s = sdmap.get(i)
        if s is None or not bool(s.get("covers_mgii")):
            continue
        recs = glob.glob(os.path.join(FITS_DIR, "%d_*" % i, "records", "BEST_*.json"))
        if not recs:
            continue
        rec = json.load(open(recs[0]))
        stem, z = rec["name"], rec["result"]["z"]
        mgii_fitted = "MgII: --" not in str(rec.get("windows", ""))
        d_ew, d_why, rmax = sdss_mgii(str(s["fn_sdss"]), z)
        out = dict(index=i, folder=a["folder"], manmask=a["manmask"], z=z,
                   mgii_in_hst=mgii_fitted, sdss_rest_max=rmax,
                   sdss_ew=d_ew, guard=d_why,
                   adopted_comps=rec["override"].get("comps_use") or "auto",
                   civ_blue=rec["result"]["civ_blue"], civ_ew=rec["result"]["civ_ew"])
        for lab, c in [("auto", None), ("low", "low"), ("mod", "mod"), ("high", "high")]:
            try:
                m_ew, m_why = model_mgii(stem, rec["override"], c)
            except Exception as exc:
                m_ew, m_why = np.nan, type(exc).__name__
            out["model_ew_" + lab] = m_ew
            if m_why and not out["guard"]:
                out["guard"] = "model: " + m_why
        try:
            m_ew, m_why = model_mgii(stem, rec["override"], rec["override"].get("comps_use"))
        except Exception:
            m_ew = np.nan
        out["model_ew_adopted"] = m_ew
        out["ratio_adopted"] = m_ew / d_ew if (np.isfinite(m_ew) and np.isfinite(d_ew) and d_ew > 0) else np.nan
        rows.append(out)
        print("  %4d %-40s sdss %7s  adopted %7s  ratio %6s  %s"
              % (i, str(a["folder"])[:40],
                 "%.1f" % d_ew if np.isfinite(d_ew) else "--",
                 "%.1f" % m_ew if np.isfinite(m_ew) else "--",
                 "%.2f" % out["ratio_adopted"] if np.isfinite(out["ratio_adopted"]) else "--",
                 out["guard"] or ""), flush=True)
    t = pd.DataFrame(rows)
    outp = os.path.join(FINALRUN, "pipeline_output", "final", "mgii_sweep.csv")
    t.to_csv(outp, index=False)
    print("\n%d objects -> %s" % (len(t), outp))


if __name__ == "__main__":
    main()
