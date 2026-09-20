"""
Refit one spectrum under an explicit mask set, render the diagnostic figure, and
record the iteration.

This is the loop the manual-fix GUI runs interactively, made scriptable so a
mask can be proposed, applied, inspected, and revised with a written trail. Each
call writes the override used, the resulting numbers, and the figure into a
per-object iteration directory, so any measurement that ends up in the paper can
be traced back to the exact mask that produced it.

The fit goes through ICAManualFixProcessor.fit_with_overrides, the single
numerical path shared by the GUI and the batch runner, so a fit made here is
bit-for-bit what those would produce under the same override.

Masks are given as wavelength ranges (rest frame, Angstrom). Ranges from
mask_proposals.json can be pulled in wholesale with --from-proposals, optionally
including the C IV candidates it holds back, and further ranges added by hand
with --mask. Unmasking is supported the same way for the pipeline's own NAL/BAL
masks where they have overreached.

Each object's folder holds the figures themselves, with the JSON records in a
records/ subfolder, and the variant currently recommended is prefixed BEST_ so it
is identifiable without opening anything.
"""

import argparse
import csv
import json
import os
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")
WORKDIR = os.path.join(OUTDIR, "coadd_work")
ITERDIR = os.path.join(OUTDIR, "fit_iterations")
PROPOSALS = os.path.join(OUTDIR, "mask_proposals.json")
from rebin_path import REBIN as _REBIN_DEFAULT
REBIN = _REBIN_DEFAULT


def parse_range(s):
    lo, hi = s.split(":")
    return [float(lo), float(hi)]



def civ_continuum(wave_r, flux_r,
                  cont_region=((1445.0, 1465.0), (1700.0, 1705.0)),
                  ew_region=(1500.0, 1600.0)):
    """The straight continuum under C IV, as get_CIV_parameters defines it.

    Recomputed here rather than imported because the pipeline's version indexes
    `continuum[i]` while stepping `flux_r[EW][i]` -- the continuum is sampled from
    the start of the reconstruction (1260 A) instead of the C IV window, 757
    pixels and roughly 240 A away. This uses the aligned values, so the
    sub-continuum count below reflects the physical question -- does the
    reconstruction dip under its own continuum -- rather than the indexing.
    """
    c1 = (wave_r >= cont_region[0][0]) & (wave_r <= cont_region[0][1])
    c2 = (wave_r >= cont_region[1][0]) & (wave_r <= cont_region[1][1])
    if c1.sum() < 2 or c2.sum() < 2:
        return None, None
    m, b = np.polyfit(np.concatenate((wave_r[c1], wave_r[c2])),
                      np.concatenate((flux_r[c1], flux_r[c2])), 1)
    ew = (wave_r >= ew_region[0]) & (wave_r <= ew_region[1])
    return ew, wave_r * m + b


def civ_subcontinuum(res):
    """Pixels where the reconstruction sits below its continuum inside C IV.

    Any such pixel is a veto: it is negative equivalent width for an emission
    line, which is unphysical, and signals the fit has gone wrong. The reported
    CIV_EW will not show it -- the pipeline clamps each pixel's contribution at
    zero -- but the blueshift is the half-flux point of that clamped cumulative
    sum, so clamped pixels distort it.
    """
    wave_r, flux_r = np.asarray(res["wave_ica"]), np.asarray(res["flux_ica"])
    ew, cont = civ_continuum(wave_r, flux_r)
    if ew is None:
        return dict(n_px=0, worst_frac=0.0, ranges=[], ok=None)
    below = ew & (flux_r < cont)
    idx = np.flatnonzero(below)
    ranges = []
    for i in idx:
        if ranges and wave_r[i] - ranges[-1][1] < 0.5:
            ranges[-1][1] = float(wave_r[i])
        else:
            ranges.append([float(wave_r[i]), float(wave_r[i])])
    worst = float(np.min((flux_r[below] - cont[below]) / cont[below])) if idx.size else 0.0
    out = dict(n_px=int(idx.size), worst_frac=round(worst, 4),
               ranges=[[round(a, 2), round(b, 2)] for a, b in ranges], ok=bool(idx.size == 0))
    # A zero EW is not a clean fit, it is no measurement: get_CIV_parameters
    # returns 0 (and a sentinel blueshift of 9574.1) when the continuum windows
    # at 1445-1465 / 1700-1705 hold no data or the line is absent. Treat it as
    # a failure of its own kind so the BEST_ rule cannot pick it.
    if res.get("civ_ew", 1.0) <= 0:
        out["ok"] = False
        out["no_line"] = True

    # A measurement needs data in the window it is measured from. The veto above
    # cannot enforce that: with no observed pixels inside 1500-1600 A there is
    # nothing for the reconstruction to sit below, so it reports a clean pass on
    # a blueshift extrapolated entirely from the rest of the spectrum. Mrk 352's
    # COS spectrum ends at 1447.8 A and still returned -1255.7 km/s and EW 293.3;
    # ICA_Results_Rebin_master.csv carries 27 such rows, spanning -5165 to +5398
    # km/s, sitting in the tails of the distribution. Coverage is what disqualifies
    # these, so coverage is checked here rather than left to the reader.
    w_o = np.asarray(res.get("wave_arb", []), float)
    f_o = np.asarray(res.get("flux_arb", []), float)
    m_o = np.asarray(res.get("mask_arb", []))
    if w_o.size and w_o.size == f_o.size == m_o.size:
        usable = (w_o >= 1500) & (w_o <= 1600) & (m_o == 0) & np.isfinite(f_o)
        out["n_civ_px"] = int(usable.sum())
        if usable.any():
            out["civ_span"] = [round(float(w_o[usable].min()), 1),
                               round(float(w_o[usable].max()), 1)]
        else:
            out["civ_span"] = None
            out["ok"] = False
            out["no_civ_data"] = True
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="spectrum stem, e.g. 'NGC 3516_COMBINED'")
    ap.add_argument("--label", default=None, help="short tag for this iteration")
    ap.add_argument("--from-proposals", action="store_true",
                    help="start from mask_proposals.json ranges for this name")
    ap.add_argument("--include-civ", action="store_true",
                    help="also apply the C IV candidates held back in the proposals")
    ap.add_argument("--mask", action="append", default=[], type=parse_range,
                    metavar="LO:HI", help="additional mask range (repeatable)")
    ap.add_argument("--unmask", action="append", default=[], type=parse_range,
                    metavar="LO:HI", help="range to unmask (repeatable)")
    ap.add_argument("--comps", default=None, choices=[None, "low", "mod", "high"],
                    help="force a component set")
    ap.add_argument("--base", default=None,
                    help="iteration JSON to start from instead of proposals")
    ap.add_argument("--from-store", default=None, metavar="STEM",
                    help="seed from the user's curated override for STEM in "
                         "ica/manual_fix_overrides.json (all five fields, pixels included)")
    ap.add_argument("--replot", default=None, metavar="RECORD.json",
                    help="re-render an existing iteration in place from its "
                         "recorded override; adds no new iteration")
    ap.add_argument("--grow-extra", type=int, default=0, metavar="PIXELS",
                    help="widen every mask range by this many 69 km/s pixels on "
                         "each side (the user judged a 2-pixel grow too small)")
    ap.add_argument("--grow-red", type=int, default=0, metavar="PIXELS",
                    help="additionally widen every mask range redward by this many pixels")
    ap.add_argument("--protect", action="append", default=[], type=parse_range,
                    metavar="LO:HI", help="remove any mask range overlapping this "
                         "window, e.g. to keep masks off the C III] anchor")
    ap.add_argument("--max-width", type=float, default=None, metavar="ANGSTROM",
                    help="drop proposal ranges wider than this; real absorption is "
                         "narrow, a broad deficit is the model being wrong")
    args = ap.parse_args()

    mask_ranges, unmask_ranges, comps = [], [], args.comps
    mask_pixels, unmask_pixels = [], []
    if args.from_store:
        # The user's curated masks are the best available prior for an object
        # they have already worked; start there rather than from a blank slate.
        # Ranges and pixels are rest-frame wavelengths, and the combined spectrum
        # shares the lattice of its constituents, so they carry straight across.
        store = json.load(open("/Users/gtr/Work/git/HST-Paper/ica/manual_fix_overrides.json"))
        e = store.get(args.from_store)
        if e is None:
            sys.exit("no curated override for %r" % args.from_store)
        mask_ranges += [list(r) for r in e.get("mask_ranges", [])]
        unmask_ranges += [list(r) for r in e.get("unmask_ranges", [])]
        mask_pixels += [float(x) for x in e.get("mask_pixels", [])]
        unmask_pixels += [float(x) for x in e.get("unmask_pixels", [])]
        comps = comps or e.get("forced_components") or None
        print("   seeded from curated override for %s: %d ranges, %d pixels, %d unmask ranges, %d unmask pixels"
              % (args.from_store, len(mask_ranges), len(mask_pixels), len(unmask_ranges), len(unmask_pixels)))
    if args.replot:
        # Re-running the identical override is deterministic through the shared
        # fit path, so this reproduces the fit exactly and only the figure and
        # the derived diagnostics change.
        args.base = args.replot
    if args.base:
        b = json.load(open(args.base))["override"]
        mask_ranges = list(b.get("mask_ranges", [])) + mask_ranges
        unmask_ranges = list(b.get("unmask_ranges", [])) + unmask_ranges
        mask_pixels = list(b.get("mask_pixels", [])) + mask_pixels
        unmask_pixels = list(b.get("unmask_pixels", [])) + unmask_pixels
        comps = comps or b.get("comps_use")
    if args.from_proposals and os.path.exists(PROPOSALS):
        p = json.load(open(PROPOSALS)).get(args.name, {})
        mask_ranges += [list(r) for r in p.get("mask_ranges", [])]
        if args.include_civ:
            mask_ranges += [list(r) for r in p.get("_civ_candidates", [])]
    if args.max_width is not None:
        dropped = [r for r in mask_ranges if r[1] - r[0] > args.max_width]
        mask_ranges = [r for r in mask_ranges if r[1] - r[0] <= args.max_width]
        if dropped:
            print("   dropped %d proposal ranges wider than %.1f A: %s"
                  % (len(dropped), args.max_width,
                     ", ".join("%.0f-%.0f" % (a, b) for a, b in dropped)))
    mask_ranges += args.mask
    unmask_ranges += args.unmask

    STEP = np.log(10) / 10000.0          # one rebinned pixel, 69.03 km/s
    if args.grow_extra or args.grow_red:
        g_lo = np.exp(-STEP * args.grow_extra)
        g_hi = np.exp(STEP * (args.grow_extra + args.grow_red))
        mask_ranges = [[lo * g_lo, hi * g_hi] for lo, hi in mask_ranges]
    for plo, phi in args.protect:
        before = len(mask_ranges)
        mask_ranges = [r for r in mask_ranges if r[1] < plo or r[0] > phi]
        print("   protected %.0f-%.0f A: removed %d mask ranges" % (plo, phi, before - len(mask_ranges)))

    path = WORKDIR if os.path.exists(os.path.join(WORKDIR, args.name + ".fits")) else REBIN

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ica.manual_fix import ICAManualFixProcessor

    proc = ICAManualFixProcessor(rebin_path=path, master_mode=True)
    res = proc.fit_with_overrides(args.name, mask_ranges=mask_ranges,
                                  mask_pixels=mask_pixels, unmask_ranges=unmask_ranges,
                                  unmask_pixels=unmask_pixels, comps_use=comps)

    from object_paths import folder_for
    d = os.path.join(ITERDIR, folder_for(args.name))
    os.makedirs(d, exist_ok=True)
    rec = os.path.join(d, "records")
    if args.replot:
        prev = json.load(open(args.replot))
        tag = os.path.basename(args.replot)[:-len(".json")]
        n = prev.get("iteration", 0)
    else:
        n = 1 + len([f for f in os.listdir(rec) if f.endswith(".json")]) if os.path.isdir(rec) else 1
        tag = "iter%02d%s" % (n, ("_" + args.label) if args.label else "")

    proc.create_diagnostic_plot(
        res["wave_arb"], res["flux_arb"], res["errs_arb"], res["mask_arb"],
        res["wave_ica"], res["flux_ica"], args.name, res["spec_name"],
        res["civ_blue"], res["civ_ew"], show=False, save=False)

    sub = civ_subcontinuum(res)

    # Break the plotted data line across wavelength gaps. A combined spectrum can
    # have a large hole -- Mrk 1044 has 3103 A of nothing between its UV and
    # optical halves -- and matplotlib joins the two sides with a straight line
    # that reads as data. Inserting NaN at the gap breaks the line and touches
    # nothing else; trimming the spectrum instead would tidy the plot but move
    # f2500, since the far side anchors the continuum fit behind the morphing.
    def _break_gaps(ax, gap=20.0):
        for ln in ax.get_lines():
            x, y = ln.get_xdata(), ln.get_ydata()
            if len(x) < 3 or not np.all(np.diff(x) >= 0):
                continue
            big = np.flatnonzero(np.diff(x) > gap)
            if big.size == 0:
                continue
            xi, yi = np.asarray(x, float).copy(), np.asarray(y, float).copy()
            xi = np.insert(xi, big + 1, (xi[big] + xi[big + 1]) / 2.0)
            yi = np.insert(yi, big + 1, np.nan)
            ln.set_data(xi, yi)


    # The shared plot sets the C IV panel's upper limit from the 99th percentile
    # of flux above 1400 A, which clips a sharp narrow peak off the top -- on
    # NGC 4395 the peak was not visible at all. Rescale to what is actually in
    # the window, and mark any sub-continuum stretch in red.
    fig = plt.gcf()
    for _ax in fig.axes:
        _break_gaps(_ax)
    wa, fa = np.asarray(res["wave_arb"]), np.asarray(res["flux_arb"])
    wi, fi = np.asarray(res["wave_ica"]), np.asarray(res["flux_ica"])
    ma = np.asarray(res["mask_arb"])
    win_a = (wa >= 1500) & (wa <= 1600) & np.isfinite(fa) & (ma == 0)
    win_i = (wi >= 1500) & (wi <= 1600) & np.isfinite(fi)
    if win_a.any() and win_i.any():
        top = max(np.nanmax(fa[win_a]), np.nanmax(fi[win_i]))
        bot = min(np.nanmin(fa[win_a]), np.nanmin(fi[win_i]), 0.0)
        pad = 0.06 * (top - bot)
        for ax in fig.axes:
            lo, hi = ax.get_xlim()
            if abs(lo - 1500) < 1 and abs(hi - 1600) < 1 and "Flux" in (ax.get_ylabel() or ""):
                ax.set_ylim(bot - pad, top + pad)
                for a, b in sub["ranges"]:
                    ax.axvspan(a, b, color="red", alpha=0.25, lw=0, zorder=0)
                if sub["n_px"]:
                    ax.text(0.01, 0.97,
                            "%d sub-continuum px (worst %.1f%%) -- REJECT"
                            % (sub["n_px"], 100 * sub["worst_frac"]),
                            transform=ax.transAxes, va="top", ha="left",
                            fontsize=11, color="red", weight="bold")
    # Figures sit at the top of the object's folder and the JSON records go
    # underneath in records/. The user browses these folders by flipping through
    # images, so the images are what the folder should show.
    os.makedirs(os.path.join(d, "records"), exist_ok=True)
    png = os.path.join(d, tag + ".png")
    plt.savefig(png, dpi=100, bbox_inches="tight")
    plt.close("all")

    # How well the model tracks the data in each diagnostic window. C III] and
    # Mg II are the anchors; C IV is reported but must not be optimised on.
    w, f, e, m = res["wave_arb"], res["flux_arb"], res["errs_arb"], np.asarray(res["mask_arb"])
    model = np.interp(w, res["wave_ica"], res["flux_ica"], left=np.nan, right=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = (f - model) / e
    windows = (("SiIV", 1360, 1440), ("CIV", 1500, 1600), ("CIII]", 1860, 1960), ("MgII", 2740, 2860))
    parts = []
    for lab, lo, hi in windows:
        sel = (w >= lo) & (w <= hi) & (m == 0) & np.isfinite(rs)
        if sel.sum() < 20:
            parts.append("%s: --" % lab)
        else:
            parts.append("%s: med %+.2f rms %.2f (n=%d)"
                         % (lab, np.median(rs[sel]), np.sqrt(np.mean(rs[sel] ** 2)), sel.sum()))
    record_windows = "  |  ".join(parts)

    # Redshift check, done on every fit rather than as a separate campaign.
    # Two estimators: the user's quick one -- the peak (not centroid) of C III],
    # smoothed first because a raw peak of noisy data is unstable -- and the
    # model-vs-data cross-correlation at the two anchors, which is the
    # differential test. Sign verified by applying a known shift: +ve means the
    # data sit redward of the model, i.e. the assumed redshift is too low by
    # that much. Caution: on NGC 3516, whose redshift NED knows to 11 km/s,
    # this reads -810 at C III] and -440 at Mg II -- the model's blend shapes
    # differ from the data's. Treat it as a flag to look, never as a correction
    # to apply, and prefer an external narrow-line redshift where one exists.
    from model_data_shift import xcorr_shift
    CIII = 1908.73

    # Pixels that are not measurements must not set an anchor. Some rebinned
    # spectra carry blocks the bad-pixel mask does not flag: Mrk 231_COS has
    # NEGATIVE errors across 1861-2341 A, which is 226 of the 228 pixels in the
    # C III] window, and NGC 985_COS and 2MASX-J00391586-5117013_COS carry
    # similar blocks just redward of it. A negative or zero error is not a
    # measurement at all, and an error orders of magnitude above the window's
    # own median is the detector falling off the end. Both are excluded here;
    # everything else the mask already handles. Note this is a change to a
    # DIAGNOSTIC only -- the fit never sees it, and no measurement moves.
    finite_e = np.isfinite(e) & (e > 0)
    usable = (m == 0) & np.isfinite(f) & finite_e

    def _anchor_ok(lo, hi):
        """Usable pixels in a window, minus any whose error is wild for it."""
        sel = usable & (w >= lo) & (w <= hi)
        if sel.sum() < 10:
            return sel
        med_e = float(np.median(e[sel]))
        return sel & (e <= 100.0 * med_e)

    zdiag = {}
    sel = _anchor_ok(1880, 1940)
    if sel.sum() >= 25:
        ww, ff = w[sel], f[sel]
        k = np.ones(5) / 5.0
        sm = np.convolve(ff, k, mode="same")
        pk = float(ww[np.argmax(sm[2:-2]) + 2])
        zdiag["ciii_peak_dv"] = round(299792.458 * (pk - CIII) / CIII, 0)
    for lab, lo, hi in (("ciii_xcorr_dv", 1860, 1960), ("mgii_xcorr_dv", 2740, 2860)):
        ok = _anchor_ok(lo, hi)
        v, n_ = xcorr_shift(w, np.where(ok, f, np.nan), model, lo, hi)
        zdiag[lab] = None if not np.isfinite(v) else round(float(v), 0)
        # Carry the health of the anchor with the anchor. NGC 985's C III]
        # window sits at the blue edge of the COS spectrum where the errors run
        # from 0.6 to 7.4 across the window, so its +2000 km/s is a low-S/N
        # number rather than a corrupted one -- no cut fixes that, and the only
        # honest remedy is to report the S/N beside the shift.
        if ok.sum():
            zdiag[lab.replace("_dv", "_snr")] = round(float(np.median(f[ok] / e[ok])), 1)
            zdiag[lab.replace("_dv", "_npx")] = int(ok.sum())
    def _zfmt(k_, v):
        if v is None:
            return "--"
        if k_.endswith("_dv"):
            return "%+.0f km/s" % v
        return ("%.1f" % v) if k_.endswith("_snr") else "%d" % v
    zline = "   redshift check   " + "  ".join(
        "%s=%s" % (k_, _zfmt(k_, v)) for k_, v in zdiag.items())

    record = {
        "name": args.name, "iteration": n, "label": args.label,
        "when_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "override": {"mask_ranges": mask_ranges, "unmask_ranges": unmask_ranges,
                     "mask_pixels": mask_pixels, "unmask_pixels": unmask_pixels,
                     "comps_use": comps},
        "result": {"civ_blue": res["civ_blue"], "civ_ew": res["civ_ew"],
                   "f2500": res["f2500"], "z": res["z"]},
        "plot": png,
        "windows": record_windows,
        "redshift_check_kms": zdiag,
        "civ_subcontinuum": sub,
    }
    json.dump(record, open(os.path.join(d, "records", tag + ".json"), "w"), indent=1)

    print("%s  %s" % (args.name, tag))
    print("   residual/sigma  " + record_windows)
    print(zline)
    print("   masks: %d ranges (%.1f A total) + %d pixels   unmask: %d ranges + %d pixels   comps: %s"
          % (len(mask_ranges), sum(hi - lo for lo, hi in mask_ranges), len(mask_pixels),
             len(unmask_ranges), len(unmask_pixels), comps or "auto"))
    if sub.get("no_civ_data"):
        veto_msg = "NO DATA IN 1500-1600 A -- blueshift is extrapolated  *** REJECT ***"
    elif sub["ok"]:
        veto_msg = "none"
    elif sub.get("no_line"):
        veto_msg = "EW = 0 -- NO MEASURABLE LINE  *** REJECT ***"
    else:
        veto_msg = ("%d px, worst %.1f%% below, at %s  *** REJECT ***"
                    % (sub["n_px"], 100 * sub["worst_frac"],
                       ", ".join("%.1f-%.1f" % (a, b) for a, b in sub["ranges"][:4])))
    print("   C IV sub-continuum: %s" % veto_msg)
    if sub.get("n_civ_px") is not None:
        span = sub.get("civ_span")
        print("   C IV window coverage: %d usable px%s"
              % (sub["n_civ_px"],
                 ("  (%.1f-%.1f A)" % tuple(span)) if span else ""))
    print("   blue=%9.1f   ew=%8.2f   f2500=%.4f" % (res["civ_blue"], res["civ_ew"], res["f2500"]))
    print("   -> %s" % png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
