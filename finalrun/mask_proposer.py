"""
Propose masks for an ICA fit: absorption features and cosmic-ray spikes.

This produces a first pass for a human to correct, not a finished mask. It
detects the two things that are mechanically identifiable -- pixels sitting well
below the reconstruction (absorption) and isolated narrow excursions above it
(cosmic rays) -- and leaves every judgment call alone.

Three details follow the user's own practice:

  Grow. Each detection is widened by a pixel or two beyond the pixels that
  actually trip the threshold. Absorption wings extend past what the eye or a
  threshold picks out, and leaving them in drags the fit.

  Spikes are narrow. A cosmic ray is an isolated excursion a couple of pixels
  wide; a broad positive departure from the model is more likely real line
  structure the fit has failed to capture, and masking it would be hiding the
  evidence rather than cleaning the data.

  C IV is handled separately. Absorption inside 1500-1600 A is detected and
  reported but flagged rather than silently applied, because C IV's offset from
  the model is the measurement this paper reports. Masking there is legitimate --
  the user does it routinely, for real absorption -- but it is exactly where a
  false positive fabricates a result, so it wants eyes on it.

Proposals are written to their own file in the override-store schema. Nothing is
written to ica/manual_fix_overrides.json, which holds curated work.

Detection is against the unmasked reconstruction, which is itself dragged by the
absorption being looked for, so this under-detects on badly contaminated objects.
Re-running after applying a round of masks will find more.
"""

import argparse
import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")
QUEUE = os.path.join(OUTDIR, "work_queue.csv")
WORKDIR = os.path.join(OUTDIR, "coadd_work")
PLOTS = os.path.join(OUTDIR, "mask_proposal_plots")
OUT_JSON = os.path.join(OUTDIR, "mask_proposals.json")
OUT_CSV = os.path.join(OUTDIR, "mask_proposals.csv")
from rebin_path import REBIN as _REBIN_DEFAULT
REBIN = _REBIN_DEFAULT

CIV_LO, CIV_HI = 1500.0, 1600.0


def runs(flags):
    """Contiguous True runs as (start, stop_exclusive) index pairs."""
    out, i, n = [], 0, flags.size
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def merge(ranges, wave):
    """Merge overlapping wavelength ranges."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    out = [list(ranges[0])]
    for lo, hi in ranges[1:]:
        if lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(r) for r in out]


def propose(res, absorb_sigma, spike_sigma, grow, max_spike_px):
    wave = res["wave_arb"]
    flux = res["flux_arb"]
    errs = res["errs_arb"]
    already = np.asarray(res["mask_arb"]) != 0

    model = np.interp(wave, res["wave_ica"], res["flux_ica"],
                      left=np.nan, right=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        resid = (flux - model) / errs
    valid = np.isfinite(resid) & ~already

    absorb = valid & (resid < -absorb_sigma)
    spike_hi = valid & (resid > spike_sigma)

    found = []
    for (i, j) in runs(absorb):
        # Masking only downward fluctuations is asymmetric and biases the fit
        # high, so a bare 2.5-sigma dip is not enough on its own. Require either
        # two adjacent pixels below threshold -- noise rarely does that, real
        # absorption does -- or a single pixel deep enough to be a line rather
        # than a fluctuation. At 69 km/s a typical intervening absorber is
        # unresolved, so single-pixel detections cannot simply be discarded.
        depth = float(np.nanmax(np.abs(resid[i:j])))
        if (j - i) >= 2 or depth >= max(absorb_sigma, 4.0):
            found.append(("absorption", i, j))
    for (i, j) in runs(spike_hi):
        # Broad positive departures are unmodelled line structure, not cosmic
        # rays -- masking those would hide the thing worth looking at.
        if (j - i) <= max_spike_px:
            found.append(("cosmic_ray", i, j))

    proposals = []
    for kind, i, j in found:
        a = max(0, i - grow)
        b = min(wave.size - 1, j - 1 + grow)
        lo, hi = float(wave[a]), float(wave[b])
        proposals.append({
            "kind": kind,
            "lo": lo,
            "hi": hi,
            "npix": int(b - a + 1),
            "peak_sigma": float(np.nanmax(np.abs(resid[i:j]))),
            "in_civ": bool(hi >= CIV_LO and lo <= CIV_HI),
        })
    return proposals, resid, model


def plot(res, proposals, resid, model, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wave, flux = res["wave_arb"], res["flux_arb"]
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), constrained_layout=True)
    for ax, (lo, hi), title in (
            (axes[0], (float(np.nanmin(wave)), float(np.nanmax(wave))), "full"),
            (axes[1], (CIV_LO, CIV_HI), "C IV"),
    ):
        ax.plot(wave, flux, "-k", lw=0.7, label="data")
        ax.plot(wave, model, "-r", lw=1.2, label="ICA")
        for p in proposals:
            ax.axvspan(p["lo"], p["hi"], color=("tab:orange" if p["kind"] == "absorption"
                                                else "tab:purple"), alpha=0.30, lw=0)
        ax.set_xlim(lo, hi)
        sel = (wave >= lo) & (wave <= hi) & np.isfinite(flux)
        if sel.any():
            ax.set_ylim(np.nanpercentile(flux[sel], 0.5),
                        np.nanpercentile(flux[sel], 99.8) * 1.10)
        ax.set_title("%s -- %s (orange=absorption, purple=cosmic ray)"
                     % (res["name"], title))
        ax.legend(loc="upper right", fontsize=8)
    axes[2].plot(wave, resid, "-", color="0.4", lw=0.6)
    axes[2].axhline(0, color="r", lw=0.8)
    axes[2].set_xlim(CIV_LO, CIV_HI)
    axes[2].set_ylim(-12, 12)
    axes[2].set_ylabel("residual (sigma)")
    axes[2].set_xlabel("Wavelength (A)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=110)
    plt.close("all")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", type=int, nargs="*", default=[])
    ap.add_argument("--names", nargs="*", default=[],
                    help="explicit spectrum stems to propose on, e.g. a single "
                         "instrument's spectrum when the combination is worse")
    ap.add_argument("--combined", action="store_true", default=True,
                    help="propose on the combined spectrum (default)")
    ap.add_argument("--absorb-sigma", type=float, default=2.5)
    ap.add_argument("--spike-sigma", type=float, default=4.0)
    ap.add_argument("--grow", type=int, default=3,
                    help="pixels to extend each detection on both sides")
    ap.add_argument("--max-spike-px", type=int, default=2)
    ap.add_argument("--plots", action="store_true", default=True)
    args = ap.parse_args()

    from ica.manual_fix import ICAManualFixProcessor

    queue = {int(r["order"]): r for r in csv.DictReader(open(QUEUE))}
    store, rows = {}, []

    targets = []
    for o in args.orders:
        q = queue[o]
        name = "%s_COMBINED" % q["name_mast_key"]
        if os.path.exists(os.path.join(WORKDIR, name + ".fits")):
            targets.append((o, q, name, WORKDIR))
        else:
            targets.append((o, q, q["stems"].split(";")[0], REBIN))
    for nm in args.names:
        q = next((r for r in queue.values() if nm in r["stems"].split(";")), {"name_pub": nm})
        targets.append(("-", q, nm, WORKDIR if os.path.exists(os.path.join(WORKDIR, nm + ".fits")) else REBIN))
    for o, q, name, path in targets:
        try:
            proc = ICAManualFixProcessor(rebin_path=path, master_mode=True)
            res = proc.fit_with_overrides(name)
        except Exception as exc:
            print("order %-3s %-28s FIT FAILED %s" % (o, name, type(exc).__name__))
            continue

        proposals, resid, model = propose(res, args.absorb_sigma, args.spike_sigma,
                                          args.grow, args.max_spike_px)
        civ = [p for p in proposals if p["in_civ"]]
        # Outside C IV the detections are safe to apply; inside, they are
        # reported for review rather than committed.
        ranges = merge([(p["lo"], p["hi"]) for p in proposals if not p["in_civ"]],
                       res["wave_arb"])
        civ_ranges = merge([(p["lo"], p["hi"]) for p in civ], res["wave_arb"])
        store[name] = {"mask_ranges": [list(r) for r in ranges],
                       "mask_pixels": [], "unmask_ranges": [], "unmask_pixels": [],
                       "_civ_candidates": [list(r) for r in civ_ranges]}
        print("order %-3s %-26s  %3d proposals (%d absorption, %d spikes); "
              "%d inside C IV held for review"
              % (o, name, len(proposals),
                 sum(1 for p in proposals if p["kind"] == "absorption"),
                 sum(1 for p in proposals if p["kind"] == "cosmic_ray"),
                 len(civ_ranges)))
        for p in proposals:
            rows.append(dict(order=o, object=q["name_pub"], variant=name, **p))
        if args.plots:
            plot(res, proposals, resid, model,
                 os.path.join(PLOTS, name.replace("/", "_") + ".png"))

    # Merge into the existing store rather than replacing it, so proposing for
    # one batch of objects does not discard the proposals made for earlier ones.
    existing = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    existing.update(store)
    json.dump(existing, open(OUT_JSON, "w"), indent=1)
    if rows:
        with open(OUT_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("\n-> %s\n-> %s\n-> %s/" % (OUT_JSON, OUT_CSV, PLOTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
