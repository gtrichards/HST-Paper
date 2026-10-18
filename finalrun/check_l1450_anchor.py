"""Is log L1450 measured against a real continuum, or against the 1440-1460 gap?

The failure this screens for was identified at index 245 and anatomised at 279.
spec_morph.emission_lines2 excludes Si IV at 1360-1440 A and C IV at 1460-1685 A,
which leaves a 20 A gap at 1440-1460 A as the ONLY continuum anchor between the
two lines -- and the 1450 A luminosity window (1445-1465) sits on it.
cont_filtered takes a running median in that gap and interpolates linearly across
C IV from it.  So whatever the data do in those 20 A sets the continuum
normalisation under the whole C IV region, and the 1450 A luminosity with it.

That is Rankine's prescription, not a defect of this project's code, so this
tool reports the damage rather than repairing it.

Three diagnostics, because they fail independently:

  gap significance   the MEDIAN's significance in 1440-1460 A on the
                   de-normalised flux (cn x arb), i.e. per-pixel S/N x sqrt(n).
                   Per-pixel S/N is the wrong statistic and flagged 70 healthy
                   objects when it was tried: the window holds ~60 rebinned
                   pixels, so a median can be well determined at a per-pixel S/N
                   near 1.  Index 245, the archetype, is 0.4 sigma on this
                   measure where index 347, which is sound, is 21.7.
  gap/neighbours   gap flux over the mean of 1400-1440 and 1460-1480 A.  WEAK,
                   and recorded only for continuity with the earlier screen: a
                   BAL trough wider than 20 A sits on the neighbours too, so the
                   ratio comes out near 1 on exactly the objects it should
                   catch.  Index 317 reads 0.89 here while a power law fitted
                   redward of C IV puts its anchor at 0.48 of the continuum, and
                   005 reads 0.88 against 0.75.  Prefer the power-law test
                   (--powerlaw), which is what the index-317 note used.
  cn(1450)/cn(1700)   the Continuum Normalisation column at the two anchors.
                   A lambda^-1.5 continuum gives 1.27, and the adopted sample
                   runs 0.95-1.28 (16-84%, median 1.13).  This is the diagnostic
                   that decides the luminosity, because it is what multiplies
                   it, and it is the one that catches the broad-trough cases the
                   neighbour ratio misses.

And the direct consequence: log L1450 - log L1700, which for f_lambda ~
lambda^-1.5 must be +0.035 dex.  A negative value is backwards for a quasar
continuum and is the signature the screen exists to catch.

    python check_l1450_anchor.py                 # every adopted fit
    python check_l1450_anchor.py 125 005 317     # named indices only
    python check_l1450_anchor.py --flagged        # only rows that fail a test
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rebin_path import REBIN, ITERDIR            # env-overridable

#: lambda L_lambda at 1450 A exceeds 1700 A by (1700/1450)^0.5 for f ~ lam^-1.5.
EXPECTED_DEX = float(np.log10((1700.0 / 1450.0) ** 0.5))
#: cn(1450)/cn(1700) for the same continuum.
EXPECTED_CN = float((1450.0 / 1700.0) ** -1.5)

GAP = (1440.0, 1460.0)                 #: the only anchor between Si IV and C IV
NEIGHBOURS = ((1400.0, 1440.0), (1460.0, 1480.0))

#: Significance of the gap MEDIAN, not the per-pixel S/N -- see the docstring.
GAP_SIG_MIN = 5.0
#: Reported, not trusted; see the docstring.  Only extreme values mean anything.
GAP_RATIO_MIN = 0.40
CN_RATIO_MIN, CN_RATIO_MAX = 0.80, 2.0 #: cn ratio outside this distorts the luminosity
DEX_MIN = -0.10                        #: below this the ordering is backwards


def measure(spec, mono):
    d = fits.open(spec)[1].data
    w = d["Rest-frame Wavelength"]
    arb = d["Coadded Flux (Arbitrary Units)"]
    err = d["Coadded Flux Errors"]
    msk = d["Bad Pixel Mask"]
    cn = d["Continuum Normalisation"]
    # De-normalise: the gap is a statement about the observed spectrum, and the
    # arbitrary-unit flux has already been divided by the very continuum whose
    # soundness is the question.
    f, e = arb * cn, err * cn
    ok = (msk == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)

    def win(lo, hi):
        t = ok & (w >= lo) & (w <= hi)
        if t.sum() < 3:
            return None
        return dict(n=int(t.sum()), flux=float(np.median(f[t])),
                    snr=float(np.median(f[t] / e[t])))

    out = {}
    g = win(*GAP)
    out["gap_n"] = g["n"] if g else 0
    out["gap_flux"] = g["flux"] if g else float("nan")
    out["gap_snr"] = g["snr"] if g else float("nan")
    out["gap_sig"] = (g["snr"] * np.sqrt(g["n"])) if g else float("nan")
    nb = [win(lo, hi) for lo, hi in NEIGHBOURS]
    vals = [x["flux"] for x in nb if x]
    out["nb_flux"] = float(np.mean(vals)) if vals else float("nan")
    out["gap_ratio"] = out["gap_flux"] / out["nb_flux"] if vals else float("nan")
    c14 = (mono.get("1450") or {}).get("nearest_cont")
    c17 = (mono.get("1700") or {}).get("nearest_cont")
    out["cn1450"], out["cn1700"] = c14, c17
    out["cn_ratio"] = (c14 / c17) if (c14 and c17) else float("nan")
    L14 = (mono.get("1450") or {}).get("logL")
    L17 = (mono.get("1700") or {}).get("logL")
    out["logL1450"], out["logL1700"] = L14, L17
    try:
        out["dex"] = float(L14) - float(L17)
    except (TypeError, ValueError):
        out["dex"] = float("nan")
    if out["dex"] != out["dex"]:
        out["dex"] = float("nan")
    return out


def faults(r):
    """Which tests this row fails, in the order that decides what to do."""
    out = []
    if not r["gap_n"]:
        return ["no anchor data"]
    if np.isfinite(r["gap_sig"]) and r["gap_sig"] < GAP_SIG_MIN:
        out.append("undetected anchor (median %.1f sigma)" % r["gap_sig"])
    elif np.isfinite(r["gap_ratio"]) and r["gap_ratio"] < GAP_RATIO_MIN:
        out.append("absorbed anchor (gap/nbr %.2f)" % r["gap_ratio"])
    if np.isfinite(r["cn_ratio"]) and not (CN_RATIO_MIN <= r["cn_ratio"] <= CN_RATIO_MAX):
        out.append("cn ratio %.2f vs %.2f expected" % (r["cn_ratio"], EXPECTED_CN))
    if np.isfinite(r["dex"]) and r["dex"] < DEX_MIN:
        out.append("backwards by %.3f dex" % (-r["dex"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("index", nargs="*", help="object indices (default: all)")
    ap.add_argument("--flagged", action="store_true", help="print only failing rows")
    args = ap.parse_args()
    want = set("%03d" % int(i) for i in args.index) if args.index else None

    rows = []
    for folder in sorted(glob.glob(os.path.join(ITERDIR, "*"))):
        base = os.path.basename(folder)
        mm = re.match(r"^(\d{3})_", base)
        if not mm or (want and mm.group(1) not in want):
            continue
        best = glob.glob(os.path.join(folder, "records", "BEST_*.json"))
        if not best:
            continue
        rec = json.load(open(best[0]))
        spec = os.path.join(REBIN, rec["name"] + ".fits")
        if not os.path.exists(spec):
            continue
        r = measure(spec, rec.get("monochromatic", {}))
        r.update(index=mm.group(1), folder=base, stem=rec["name"],
                 record=os.path.basename(best[0]))
        r["faults"] = "; ".join(faults(r))
        rows.append(r)

    print("expected: cn ratio %.2f, dex %+.3f   thresholds: gap median >= %.1f sigma, "
          "gap/nbr >= %.2f, cn %.2f-%.2f, dex >= %+.2f"
          % (EXPECTED_CN, EXPECTED_DEX, GAP_SIG_MIN, GAP_RATIO_MIN,
             CN_RATIO_MIN, CN_RATIO_MAX, DEX_MIN))
    print("%-4s %-40s %7s %7s %7s %8s %8s %7s  %s"
          % ("idx", "folder", "gap sig", "gap/nbr", "cn rat", "logL1450", "logL1700",
             "dex", "fault"))
    n_bad = 0
    for r in rows:
        if r["faults"]:
            n_bad += 1
        elif args.flagged:
            continue
        num = lambda v, f="%7.2f": (f % v) if isinstance(v, float) and np.isfinite(v) else "%7s" % "--"
        print("%-4s %-40s %s %s %s %8s %8s %s  %s" % (
            r["index"], r["folder"][:40], num(r["gap_sig"], "%7.1f"), num(r["gap_ratio"]),
            num(r["cn_ratio"]),
            "%.3f" % r["logL1450"] if isinstance(r["logL1450"], float) and np.isfinite(r["logL1450"]) else "--",
            "%.3f" % r["logL1700"] if isinstance(r["logL1700"], float) and np.isfinite(r["logL1700"]) else "--",
            num(r["dex"], "%+7.3f"), r["faults"]))

    out = os.path.join(HERE, "pipeline_output", "final", "l1450_anchor_check.csv")
    keys = ["index", "folder", "stem", "record", "gap_n", "gap_flux", "gap_snr", "gap_sig",
            "nb_flux", "gap_ratio", "cn1450", "cn1700", "cn_ratio",
            "logL1450", "logL1700", "dex", "faults"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\n%d adopted fits, %d flagged -> %s" % (len(rows), n_bad, out))
    fin = np.array([r["cn_ratio"] for r in rows
                    if isinstance(r["cn_ratio"], float) and np.isfinite(r["cn_ratio"])])
    if fin.size:
        print("cn(1450)/cn(1700): median %.2f  16-84%% %.2f to %.2f  n=%d  (expected %.2f)"
              % (np.median(fin), np.percentile(fin, 16), np.percentile(fin, 84), fin.size,
                 EXPECTED_CN))
    dx = np.array([r["dex"] for r in rows
                   if isinstance(r["dex"], float) and np.isfinite(r["dex"])])
    if dx.size:
        print("log L1450 - log L1700: median %+.3f  16-84%% %+.3f to %+.3f  n=%d  "
              "(expected %+.3f);  %d backwards past %+.2f"
              % (np.median(dx), np.percentile(dx, 16), np.percentile(dx, 84), dx.size,
                 EXPECTED_DEX, int((dx < DEX_MIN).sum()), DEX_MIN))


if __name__ == "__main__":
    main()
