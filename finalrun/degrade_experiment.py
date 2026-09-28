"""How much signal-to-noise does the C IV measurement actually need?

The co-addition policy turns on a number nobody has measured: the S/N at which
the ICA reconstruction of C IV stops being reliable.  Everything else -- how many
exposures to combine, over what window -- follows from it, and combining beyond
it costs wavelength coverage and mixes flux states for nothing.

The experiment: take spectra with high S/N, degrade them in controlled steps,
refit with the masks the adopted fit used, and watch where the blueshift and
equivalent width stop being stable.  Degrading multiplies the errors by k and
adds Gaussian noise of sigma = e*sqrt(k^2-1), so the result has the statistics of
the same object observed k times less deeply.  Several noise realisations per
step, because one realisation cannot distinguish a shift from a fluctuation.

    python degrade_experiment.py --stems "Ton 1480_FOS" --k 1 2 4 8 --trials 3
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
V24 = os.path.join(HERE, "data_v23", "RebinnedSpec_v24")
WORK = os.path.join(HERE, "pipeline_output", "coadd_work")
SCRATCH = os.path.join(HERE, "pipeline_output", "degrade_work")
ITER = os.path.join(HERE, "pipeline_output", "fit_iterations_degrade")


def source_file(stem):
    for d in (V24, WORK):
        p = os.path.join(d, stem + ".fits")
        if os.path.exists(p):
            return p
    return None


def adopted_record(stem):
    """The masks the adopted fit used, so the test changes only the noise.

    Search the chosen-visit tree first, and a hand-labelled record before an
    automatic one: the test asks whether THIS fit is stable under noise, so run
    it on the fit that will be published.  Run against a raw automatic pass it
    measures the instability of a poor fit rather than of the object, which is
    why GTR asked for hand fitting first where the automatic pass is not good
    enough.
    """
    for tree in ("chosen_visit/fits", "fit_iterations_v23", "fit_iterations",
                 "fit_iterations_prerevert"):
        js = sorted(glob.glob(os.path.join(HERE, "pipeline_output", tree,
                                           "*", "records", "*.json")))
        hand = [p for p in js if "gtr_hand" in os.path.basename(p)]
        best = [p for p in js if os.path.basename(p).startswith("BEST_")]
        for cand in (hand, best, js):
            for p in sorted(cand, key=os.path.getmtime, reverse=True):
                try:
                    if json.load(open(p)).get("name") == stem:
                        return p
                except Exception:
                    continue
    return None


def degrade(src, dst, k, rng):
    with fits.open(src) as h:
        hdu = fits.HDUList([x.copy() for x in h])
        t = hdu[1].data
        f = np.asarray(t.field(1), float).copy()
        e = np.asarray(t.field(2), float).copy()
        good = np.isfinite(f) & np.isfinite(e) & (e > 0)
        if k > 1:
            extra = np.zeros_like(f)
            extra[good] = rng.normal(0.0, e[good] * np.sqrt(k * k - 1.0))
            f = f + extra
            e = np.where(good, e * k, e)
        t.field(1)[:] = f
        t.field(2)[:] = e
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        hdu.writeto(dst, overwrite=True)


def civ_snr(path):
    with fits.open(path) as h:
        t = h[1].data
        w = np.asarray(t.field(0), float); f = np.asarray(t.field(1), float)
        e = np.asarray(t.field(2), float); m = np.asarray(t.field(3), float)
    s = (w >= 1500) & (w <= 1600) & (m == 0) & np.isfinite(f) & (e > 0)
    return float(np.median(f[s] / e[s])) if s.sum() > 20 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--k", type=float, nargs="+",
                    default=[1, 1.5, 2, 3, 4, 6, 8, 12, 16])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--stability", action="store_true",
                    help="the per-object check for the pass: a single modest noise "
                         "injection, repeated, reporting the spread rather than a "
                         "degradation curve.  Deliberately gentle -- the point is to "
                         "find fits that flip, not to manufacture error.")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--out", default=os.path.join(HERE, "pipeline_output", "final",
                                                  "degrade_experiment.csv"))
    a = ap.parse_args()

    if a.stability:
        a.k = [1.0, 1.4]
        a.trials = max(a.trials, 3)
    rng = np.random.default_rng(a.seed)
    rows = []
    for stem in a.stems:
        src = source_file(stem)
        rec = adopted_record(stem)
        if src is None:
            print("no rebinned spectrum for %r" % stem); continue
        print("\n%s   S/N %.1f   masks from %s"
              % (stem, civ_snr(src), os.path.basename(rec) if rec else "none"), flush=True)
        for k in a.k:
            n_trials = 1 if k == 1 else a.trials
            for trial in range(n_trials):
                tag = "%s__k%05.2f_t%d" % (stem, k, trial)
                dst = os.path.join(SCRATCH, tag + ".fits")
                degrade(src, dst, k, rng)
                snr = civ_snr(dst)
                cmd = [PY, os.path.join(HERE, "iterate_fit.py"), tag, "--label", "degrade"]
                if rec:
                    cmd += ["--base", rec]
                env = dict(os.environ, HSTICA_REBIN=SCRATCH, HSTICA_ITERDIR=ITER)
                r = subprocess.run(cmd, capture_output=True, text=True, env=env)
                blue = ew = np.nan
                veto = None
                recs = sorted(glob.glob(os.path.join(ITER, "*%s*" % tag.replace(" ", "_"),
                                                     "records", "*.json")))
                if recs:
                    j = json.load(open(recs[-1]))
                    blue, ew = j["result"]["civ_blue"], j["result"]["civ_ew"]
                    veto = j["civ_subcontinuum"]["ok"]
                rows.append(dict(stem=stem, k=k, trial=trial, civ_snr=snr,
                                 blue=blue, ew=ew, veto=veto,
                                 fit_tested=os.path.basename(rec) if rec else "none",
                                 hand_fitted=bool(rec and "gtr_hand" in os.path.basename(rec))))
                print("   k=%5.2f  S/N %6.2f  blue %8.1f  ew %7.2f  veto %s"
                      % (k, snr, blue, ew, veto), flush=True)
    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    d.to_csv(a.out, index=False)
    if a.stability and len(d):
        print("\nstability (blueshift spread under a 1.4x noise injection):")
        for stem, g in d.groupby("stem"):
            b0 = g[g.k == 1].blue
            b0 = float(b0.iloc[0]) if len(b0) else np.nan
            t = g[g.k > 1]
            if not len(t):
                continue
            spread = float(np.nanmax(t.blue) - np.nanmin(t.blue))
            worst = float(np.nanmax(np.abs(t.blue - b0)))
            ewsp = float(np.nanmax(np.abs(t.ew - float(g[g.k == 1].ew.iloc[0]))))
            print("  %-30s %8.1f km/s  spread %6.0f km/s (%4.1f quanta)  "
                  "worst %6.0f  EW moves %5.1f%%"
                  % (stem, b0, spread, spread / 69.0, worst,
                     100 * ewsp / float(g[g.k == 1].ew.iloc[0])))
    print("\n-> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
