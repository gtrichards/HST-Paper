"""Order the 22 IUE objects by whether C IV is measurable, before any are opened.

Every object still gets a look -- that is GTR's standing rule and nothing here
drops one.  This decides the ORDER of the looks, and warns where the figure is
likely to mislead.

Why it is worth doing for IUE when it was not for HST.  The HST pass worked
through indices in order because almost every object had a measurable line.  IUE
is different: these are 1980s low-dispersion spectra of nearby Seyferts, and the
per-pixel C IV signal-to-noise across the set spans 0.3 to 11.  At the bottom of
that range the data cannot constrain a reconstruction at all, and the figure does
not say so -- that is the hollow-residual trap this project hit repeatedly on
HST: below per-pixel S/N of about 1-2 the error bars swallow any smooth curve, so
the residuals look fine and the fit is unconstrained.  Opening those first wastes
the looks that matter, and invites a manmask 3 on an object that cannot support
one.

The columns that decide measurability:

  civ_px        usable pixels in 1500-1600 A.  Zero means no decision to make.
  sn_civ        MEDIAN PER-PIXEL S/N over the C IV window.  The trap threshold.
  peak_over_err C IV peak flux over the local error.  Catches the case where the
                median is low because the window is mostly continuum but the line
                itself is strong -- a real line can sit on a poor median.
  anchor_sig    significance of the MEDIAN in the 1445-1465 A window, per-pixel
                S/N times sqrt(n).  This is the luminosity anchor, and it fails
                independently of C IV: index 179 on the HST side has a sound line
                and a 1.0 sigma anchor.
  in_amogh      whether the student's table carries this object, so his value is
                available as the double check GTR intended.

`class` is advice on sequencing, not a verdict.  Only GTR grades.

    python triage_iue.py              # table, newest products
    python triage_iue.py --csv        # also write iue_triage.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
REBIN = os.path.join(HERE, "rebin")
FITS = os.path.join(HERE, "fits")

CIV = (1500.0, 1600.0)
ANCHOR = (1445.0, 1465.0)

#: Per-pixel S/N below which a smooth reconstruction is unconstrained by the
#: data and the residual panel stops being informative.  1.5 is the middle of
#: the 1-2 range the HST pass settled on.
TRAP_SN = 1.5


def measure(path):
    t = Table.read(path)
    w = np.asarray(t["Rest-frame Wavelength"], float)
    f = np.asarray(t["Coadded Flux (Arbitrary Units)"], float)
    e = np.asarray(t["Coadded Flux Errors"], float)
    m = np.asarray(t["Bad Pixel Mask"], float)
    c = np.asarray(t["Continuum Normalisation"], float)
    g = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    out = dict(rest_lo=np.nan, rest_hi=np.nan, civ_px=0, sn_civ=np.nan,
               peak_over_err=np.nan, anchor_px=0, anchor_sig=np.nan,
               cont_norm_ok=bool(np.isfinite(c).any()))
    if g.any():
        out["rest_lo"], out["rest_hi"] = float(w[g].min()), float(w[g].max())
    civ = g & (w >= CIV[0]) & (w <= CIV[1])
    out["civ_px"] = int(civ.sum())
    if civ.sum() > 3:
        out["sn_civ"] = float(np.nanmedian(f[civ] / e[civ]))
        j = int(np.nanargmax(f[civ]))
        out["peak_over_err"] = float(f[civ][j] / e[civ][j])
    a = g & (w >= ANCHOR[0]) & (w <= ANCHOR[1])
    out["anchor_px"] = int(a.sum())
    if a.sum() > 3:
        out["anchor_sig"] = float(np.nanmedian(f[a] / e[a]) * np.sqrt(a.sum()))
    return out


def classify(r):
    if r["civ_px"] == 0:
        return "no C IV coverage"
    if not np.isfinite(r["sn_civ"]):
        return "no C IV coverage"
    if r["sn_civ"] >= 3.0:
        return "measurable"
    if r["sn_civ"] >= TRAP_SN:
        return "marginal"
    if np.isfinite(r["peak_over_err"]) and r["peak_over_err"] >= 5.0:
        return "marginal (line over a poor median)"
    return "trap risk"


def adopted(name):
    for d in glob.glob(os.path.join(FITS, "*_%s" % name.replace(" ", "_"))):
        if glob.glob(os.path.join(d, "records", "BEST_*.json")):
            return "adopted"
    for d in glob.glob(os.path.join(FITS, "*")):
        if os.path.basename(d).endswith(name.replace(" ", "_")) and \
           glob.glob(os.path.join(d, "records", "BEST_*.json")):
            return "adopted"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()

    asmt = pd.read_csv(os.path.join(HERE, "iue_assessment.csv"))
    chosen = pd.read_csv(os.path.join(HERE, "iue_chosen_run.csv"))
    nexp = {str(r["common_name"]): int(r["n_exp"]) for _, r in chosen.iterrows()}

    rows = []
    for _, s in asmt.iterrows():
        nm = str(s["common_name"])
        p = os.path.join(REBIN, "%s_IUE.fits" % nm)
        if not os.path.exists(p):
            continue
        r = measure(p)
        r.update(row_num=int(s["row_num"]), common_name=nm, z=float(s["z"]),
                 n_exp=nexp.get(nm, np.nan),
                 in_amogh=bool(s.get("in_amogh", False)),
                 state=adopted("%s_IUE" % nm))
        r["class"] = classify(r)
        rows.append(r)

    d = pd.DataFrame(rows)
    order = {"measurable": 0, "marginal (line over a poor median)": 1,
             "marginal": 1, "trap risk": 2, "no C IV coverage": 3}
    d["_o"] = d["class"].map(order)
    d = d.sort_values(["_o", "sn_civ"], ascending=[True, False]).drop(columns="_o")

    print("IUE objects, ordered by whether C IV can be measured")
    print("  per-pixel S/N below %.1f is the hollow-residual trap: the residual panel"
          % TRAP_SN)
    print("  stops being informative, so the figure can look fine on unconstrained data.")
    print()
    hdr = ("%-4s %-16s %-7s %4s %5s %6s %6s %6s %7s %-6s %-5s %s"
           % ("row", "object", "z", "nexp", "civpx", "snCIV", "peak", "anchpx",
              "anchsig", "amogh", "state", "class"))
    print(hdr)
    print("-" * len(hdr))
    for _, r in d.iterrows():
        print("%-4d %-16s %-7.4f %4s %5d %6.2f %6.1f %6d %7s %-6s %-5s %s"
              % (r.row_num, r.common_name[:16], r.z,
                 ("%d" % r.n_exp) if np.isfinite(r.n_exp) else "-",
                 r.civ_px, r.sn_civ if np.isfinite(r.sn_civ) else -1,
                 r.peak_over_err if np.isfinite(r.peak_over_err) else -1,
                 r.anchor_px,
                 ("%.1f" % r.anchor_sig) if np.isfinite(r.anchor_sig) else "--",
                 "yes" if r.in_amogh else "no", r.state or "-", r["class"]))
    print()
    for k in ("measurable", "marginal", "marginal (line over a poor median)",
              "trap risk", "no C IV coverage"):
        n = int((d["class"] == k).sum())
        if n:
            print("  %-34s %d" % (k, n))
    if a.csv:
        out = os.path.join(HERE, "iue_triage.csv")
        d.to_csv(out, index=False)
        print("\n  wrote %s" % out)


if __name__ == "__main__":
    main()
