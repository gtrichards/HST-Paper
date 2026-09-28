"""Group an object's archive exposures into observing epochs and record the flux level of each.

Several objects in the sample were observed years apart and varied in between,
and co-adding across those states is combining measurements that are not of the
same thing -- on NGC 3783 the weighted median landed below every exposure in the
high state, because inverse-variance weighting on continuum-normalised data gave
most of the weight to the low one.  Choosing an epoch is then the right move,
which first requires knowing what the epochs are.

Exposures are grouped by start time with a gap threshold (default 30 days), and
each epoch is reported with its dates, exposure count, gratings and the median
flux in the observed C IV window.  Individual exposures more than a stated
factor from their own epoch's median are flagged: those are candidates for
exclusion rather than for epoch selection, which is a different decision.

Reads the archive files directly -- no rebinning -- so it is cheap enough to run
over the whole sample.

    python epochs.py                 # every object in the work queue
    python epochs.py --index 11 19   # just these
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
MAST = os.path.join(HERE, "data_v23", "MAST_v23")
CIV_REST = 1549.48
HALF_WIDTH = 50.0          # Angstrom, rest frame: the 1500-1600 window


def exposure_rows(path, z):
    """One record per archive file: when it was taken and how bright the source was."""
    out = []
    lo, hi = (CIV_REST - HALF_WIDTH) * (1 + z), (CIV_REST + HALF_WIDTH) * (1 + z)
    for p in sorted(glob.glob(os.path.join(path, "*.fits"))):
        try:
            with fits.open(p, memmap=False) as h:
                hd0, hd1 = h[0].header, h[1].header
                t = h[1].data
                if t is None or "WAVELENGTH" not in t.names:
                    continue
                mjd = hd1.get("EXPSTART", hd0.get("EXPSTART"))
                if mjd is None:
                    d = hd0.get("DATE-OBS")
                    mjd = np.nan if d is None else float(fits.Header().get("x", np.nan))
                vals = []
                for r in np.atleast_1d(t):
                    w = np.asarray(r["WAVELENGTH"], float)
                    f = np.asarray(r["FLUX"], float)
                    m = (w >= lo) & (w <= hi) & np.isfinite(f)
                    if m.sum() > 20:
                        vals.append(np.median(f[m]))
                if not vals:
                    continue
                out.append(dict(file=os.path.basename(p), mjd=float(mjd) if mjd else np.nan,
                                exptime=float(hd1.get("EXPTIME", hd0.get("EXPTIME", np.nan)) or np.nan),
                                grating=hd0.get("OPT_ELEM"), cenwave=hd0.get("CENWAVE"),
                                civ_flux=float(np.median(vals))))
        except Exception:
            continue
    return out


def group(rows, gap_days):
    rows = [r for r in rows if np.isfinite(r["mjd"])]
    rows.sort(key=lambda r: r["mjd"])
    epochs, cur = [], []
    for r in rows:
        if cur and (r["mjd"] - cur[-1]["mjd"]) > gap_days:
            epochs.append(cur); cur = []
        cur.append(r)
    if cur:
        epochs.append(cur)
    return epochs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, nargs="*", default=None)
    ap.add_argument("--gap-days", type=float, default=30.0,
                    help="exposures separated by more than this start a new epoch")
    ap.add_argument("--outlier-factor", type=float, default=3.0,
                    help="flag an exposure this many times from its epoch median")
    ap.add_argument("--out", default=os.path.join(HERE, "pipeline_output", "final", "epochs.csv"))
    a = ap.parse_args()

    queue = {int(r["order"]): r for r in csv.DictReader(
        open(os.path.join(HERE, "pipeline_output", "work_queue.csv")))}
    dec = pd.read_csv(os.path.join(HERE, "pipeline_output", "final",
                                   "pass_decisions.csv")).set_index("index")
    # pass_decisions.csv only carries the objects worked so far, so keying the
    # scan on it covered 93 objects of 474.  The catalogue redshift is the same
    # one rebin_v23.py uses, and is available for every object.
    cat = pd.read_csv("/Users/gtr/Work/git/HST-Paper/Data/master_catalog_v22.csv")
    zcat = {}
    for c in ("common_name", "name_mast_key", "name_pub"):
        if c in cat.columns and "best_z" in cat.columns:
            for k, v in zip(cat[c], cat["best_z"]):
                if isinstance(k, str):
                    zcat.setdefault(k.strip(), v)
    idxs = a.index if a.index else sorted(queue)

    out = []
    for i in idxs:
        name = queue[i]["name_mast_key"]
        z = np.nan
        if i in dec.index:
            try:
                z = float(dec.loc[i, "z_adopted"])
            except Exception:
                z = np.nan
        if not np.isfinite(z):
            for k in (name, queue[i].get("common_name"), queue[i].get("name_pub")):
                if isinstance(k, str) and k.strip() in zcat:
                    try:
                        z = float(zcat[k.strip()])
                    except Exception:
                        z = np.nan
                    if np.isfinite(z):
                        break
        base = os.path.join(MAST, name)
        if not os.path.isdir(base) or not np.isfinite(z):
            continue
        for inst in sorted(os.listdir(base)):
            d = os.path.join(base, inst)
            if not os.path.isdir(d):
                continue
            rows = exposure_rows(d, z)
            if not rows:
                continue
            eps = group(rows, a.gap_days)
            levels = [np.median([r["civ_flux"] for r in e]) for e in eps]
            # An epoch whose median flux is zero is not a faint state: it is a
            # set of exposures carrying no signal in this window at all, which
            # the reader already skips.  Exclude those from the variability span
            # and count them separately -- three of the brightest objects turn
            # out to have a recent 12-exposure epoch of exactly this kind.
            live = [v for v in levels if v > 0]
            span = (max(live) / min(live)) if len(live) >= 2 else (1.0 if live else np.nan)
            n_dead = sum(1 for v in levels if v <= 0)
            # A ratio taken across two different gratings is not a measurement of
            # variability: NGC 4151's STIS epochs mix G140L and E140M, whose flux
            # scales and apertures differ, and NGC 863's span of 67x comes partly
            # from the same confusion.  Quote the span within the most-used
            # grating alongside it, and say when the epochs are mixed.
            gratings = [";".join(sorted({str(r["grating"]) for r in e})) for e in eps]
            modal = max(set(gratings), key=gratings.count) if gratings else ""
            same = [v for v, g in zip(levels, gratings) if g == modal and v > 0]
            span_same = (max(same) / min(same)) if len(same) >= 2 else (1.0 if same else np.nan)
            mixed = int(len(set(gratings)) > 1)
            for n, (e, lev) in enumerate(zip(eps, levels), start=1):
                fl = np.array([r["civ_flux"] for r in e])
                odd = [r["file"] for r, v in zip(e, fl)
                       if lev > 0 and (v > a.outlier_factor * lev or v < lev / a.outlier_factor)]
                out.append(dict(
                    index=i,
                    row_num=(dec.loc[i, "row_num"] if i in dec.index else ""),
                    name_pub=(dec.loc[i, "name_pub"] if i in dec.index else queue[i].get("name_pub", name)),
                    name=name, inst=inst, epoch=n, n_epochs=len(eps),
                    n_exp=len(e), mjd_start=min(r["mjd"] for r in e),
                    mjd_end=max(r["mjd"] for r in e),
                    gratings=";".join(sorted({str(r["grating"]) for r in e})),
                    civ_flux=lev, epoch_flux_span=span,
                    epoch_flux_span_same_grating=span_same, grating_mix=mixed,
                    modal_grating=modal, dead_epoch=int(lev <= 0), n_dead_epochs=n_dead,
                    outliers=";".join(odd), n_outliers=len(odd)))
        if out and out[-1]["index"] == i:
            n_ep = max(r["n_epochs"] for r in out if r["index"] == i)
            sp = max(r["epoch_flux_span"] for r in out if r["index"] == i)
            lbl = str(dec.loc[i, "name_pub"] if i in dec.index else name)[:28]
            print("%3d %-28s %d epoch(s), flux span %.1fx" % (i, lbl, n_ep, sp), flush=True)

    df = pd.DataFrame(out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print("\n%d epochs over %d object/instrument pairs" % (len(df), df.groupby(["index", "inst"]).ngroups if len(df) else 0))
    if len(df):
        multi = df[df.n_epochs > 1]
        print("  object/instrument pairs with more than one epoch : %d"
              % multi.groupby(["index", "inst"]).ngroups)
        print("  pairs whose epochs differ in flux by >1.5x        : %d"
              % df[df.epoch_flux_span > 1.5].groupby(["index", "inst"]).ngroups)
        print("  ... and still >1.5x within one grating            : %d"
              % df[df.epoch_flux_span_same_grating > 1.5].groupby(["index", "inst"]).ngroups)
        print("  exposures flagged as outliers within their epoch  : %d" % df.n_outliers.sum())
        print("  epochs carrying no signal in the C IV window      : %d (%d exposures)"
              % (df.dead_epoch.sum(), df[df.dead_epoch == 1].n_exp.sum()))
    print("-> %s" % a.out)


if __name__ == "__main__":
    sys.exit(main())
