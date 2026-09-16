"""
Find absorption by iterating, because one pass cannot.

The mask proposer detects absorption as flux sitting below the reconstruction --
but on a heavily absorbed object the reconstruction is itself dragged down by
that same absorption, so the first pass finds only the deepest troughs and the
model stays too low. Masking those, refitting, and looking again lets the model
rise, which exposes the next layer of absorption, and so on.

This is the mechanism behind a complaint that a fit "isn't masking enough
pixels": the missing masks are invisible to a single pass by construction.

Each round reports the peak of the reconstruction inside C IV alongside the
measurements, since a model that is under-reconstructing an absorbed line shows
up there first -- a peak below the unabsorbed shoulders either side of a trough
means the fit has not recovered the intrinsic line.

Stops when a round proposes nothing new, or when the C IV veto trips (a
reconstruction dipping below its own continuum means the masking has gone too
far and the round is discarded).
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")
WORKDIR = os.path.join(OUTDIR, "coadd_work")
REBIN = "/Users/gtr/Dropbox/HST/Pratsos/RebinnedSpec_master"
STEP = np.log(10) / 10000.0


def covered(lo, hi, ranges):
    return any(a <= lo and hi <= b for a, b in ranges)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name")
    ap.add_argument("--base", default=None, help="iteration JSON to start from")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="flux this far below the model counts as absorption")
    ap.add_argument("--grow", type=int, default=3, help="pixels to grow each side")
    ap.add_argument("--window", type=float, nargs=2, default=[1460.0, 1700.0],
                    help="search range; wider than C IV so the wings are included")
    args = ap.parse_args()

    from ica.manual_fix import ICAManualFixProcessor
    from iterate_fit import civ_subcontinuum

    ov = {"mask_ranges": [], "mask_pixels": [], "unmask_ranges": [], "unmask_pixels": [],
          "comps_use": None}
    if args.base:
        b = json.load(open(args.base))["override"]
        ov = {k: list(b.get(k, [])) if isinstance(b.get(k, []), list) else b.get(k)
              for k in ov}
        ov["comps_use"] = b.get("comps_use")

    path = WORKDIR if os.path.exists(os.path.join(WORKDIR, args.name + ".fits")) else REBIN
    proc = ICAManualFixProcessor(rebin_path=path, master_mode=True)
    lo_w, hi_w = args.window

    print("%-6s %6s %9s %8s %9s %9s  %s"
          % ("round", "masks", "blue", "ew", "modelpk", "datapk", "veto"))
    prev = None
    for rnd in range(args.rounds + 1):
        res = proc.fit_with_overrides(
            args.name, mask_ranges=ov["mask_ranges"], mask_pixels=ov["mask_pixels"],
            unmask_ranges=ov["unmask_ranges"], unmask_pixels=ov["unmask_pixels"],
            comps_use=ov["comps_use"])
        w, f, e = res["wave_arb"], res["flux_arb"], res["errs_arb"]
        m = np.asarray(res["mask_arb"])
        model = np.interp(w, res["wave_ica"], res["flux_ica"], left=np.nan, right=np.nan)
        civ = (w >= 1500) & (w <= 1600)
        wi = np.asarray(res["wave_ica"]); fi = np.asarray(res["flux_ica"])
        ci = (wi >= 1500) & (wi <= 1600)
        sub = civ_subcontinuum(res)
        print("%-6d %6d %9.1f %8.2f %9.2f %9.2f  %s"
              % (rnd, len(ov["mask_ranges"]), res["civ_blue"], res["civ_ew"],
                 np.nanmax(fi[ci]) if ci.any() else np.nan,
                 np.nanmax(f[civ]) if civ.any() else np.nan,
                 "pass" if sub["ok"] else "REJECT(%d)" % sub["n_px"]), flush=True)
        if not sub["ok"]:
            print("   veto tripped -- discarding this round, keeping the previous mask set")
            if prev is not None:
                ov = prev
            break
        prev = {k: (list(v) if isinstance(v, list) else v) for k, v in ov.items()}
        if rnd == args.rounds:
            break

        with np.errstate(invalid="ignore", divide="ignore"):
            resid = (f - model) / e
        hit = (np.isfinite(resid) & (resid < -args.sigma) & (m == 0)
               & (w >= lo_w) & (w <= hi_w))
        idx = np.flatnonzero(hit)
        if idx.size == 0:
            print("   nothing further below %.1f sigma" % args.sigma)
            break
        runs, start = [], idx[0]
        for a, b in zip(idx, idx[1:]):
            if b != a + 1:
                runs.append((start, a)); start = b
        runs.append((start, idx[-1]))
        new = []
        for a, b in runs:
            i0 = max(0, a - args.grow); i1 = min(w.size - 1, b + args.grow)
            r = (float(w[i0]), float(w[i1]))
            if not covered(r[0], r[1], ov["mask_ranges"]):
                new.append(list(r))
        if not new:
            print("   no new ranges beyond those already masked")
            break
        ov["mask_ranges"] += new
        print("   + %d new: %s" % (len(new), ", ".join("%.1f-%.1f" % (a, b) for a, b in new[:8])))

    out = os.path.join(OUTDIR, "mask_iteration_%s.json"
                       % "".join(c if c.isalnum() else "_" for c in args.name))
    json.dump({"name": args.name, "override": ov}, open(out, "w"), indent=1)
    print("\n-> %s   (%d mask ranges)" % (out, len(ov["mask_ranges"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
