"""A per-exposure recommendation for one object: keep it, drop it, or choose an epoch.

Intended to be read object by object with the DATA_archive_exposures figure open
beside it.  Nothing here acts: it proposes, GTR disposes, and the agreed answer
is what gets recorded.

For each archive exposure it reports the epoch it belongs to, its exposure time,
its signal-to-noise over its own coverage, its flux level relative to the other
exposures of the same epoch and grating, and how many of its pixels sit far from
what the other exposures say at the same wavelength.  From those it proposes one
of:

    keep            nothing wrong with it
    drop: no signal essentially nothing above the noise; the current reader only
                    catches an exposure that is zero *everywhere*, so exposures
                    that are merely consistent with zero are reaching the fits
    drop: level     flux far from the rest of its own epoch, which is a
                    different thing from the source having varied between epochs
    choose epoch    the object was caught in more than one flux state; this is
                    not a defect and the decision is which epoch to fit

    python exposure_decisions.py --index 40
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
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from rebinning import read_spec_data as R

MAST = os.path.join(HERE, "data_v23", "MAST_v23")
CIV = (1500.0, 1600.0)

# Thresholds, deliberately loose and all in one place so they can be tuned.
SNR_KEEP = 3.0          # GTR: "keep things with S/N >= 3 that are within 20% of the rest"
LEVEL_TOL = 0.20        # the 20%: agreement with the other exposures of the same grating
LEVEL_TOL_MANY = 0.10   # ...tightened where there are exposures to spare.  GTR: "maybe
MANY_EXPOSURES = 12     # within 10% is appropriate for objects with large numbers of
                        # spectra".  An object with four exposures needs all four; one
                        # with several hundred can afford to keep only the best.
SNR_DEAD = 3.0          # below this S/N...
LEVEL_DEAD = 5.0        # ...and this far below the same grating's other exposures: drop.
                        # Neither alone works: ARK 120's failed first visit sits at S/N 1-3,
                        # which a 0.5 threshold missed, and a low level alone is what a
                        # genuine faint state looks like.  Together they separate a failed
                        # visit from a real one.
# PRISM only.  I had G140L in here too, on my own initiative -- GTR agreed about
# PRISM and nothing else.  G140L is an ordinary first-order STIS grating covering
# 1150-1730 A and is entirely usable for an ICA fit whose components are
# themselves low resolution; excluding it left NGC 4593 with a single S/N 3.7
# visit while 78 sound exposures at S/N 15-42 sat unused.
LOWRES_GRATINGS = {"PRI", "PRISM"}
ALIGN_RMS = 0.25        # shape disagreement with the object's best data, after scaling
                        # out the flux level: GTR's test, that an exposure may pass
                        # every level and S/N check and still "not align well with the
                        # best single FOS or COS spectrum".
LEVEL_FACTOR = 3.0      # this far from its own epoch's level: an outlier, not variability
#: Skip the within-visit level comparison when the reference median flux is
#: not positive.  `level` is a whole-exposure median, so for an object whose
#: continuum is absorbed to zero over most of a grating's range it goes
#: negative and the ratio carries no information -- Mrk 231's COS exposures
#: sat at -9e-19 and were reported at -492x and -732x their visit's level.
#: Set False only to reproduce the pre-fix behaviour for impact assessment.
LEVEL_GUARD = True
EPOCH_SPAN = 1.5        # epochs differing by more than this need a choice
GAP_DAYS = 30.0
SNR_TARGET = 10.0       # the degrade-and-refit experiment: three of five spectra held
                        # their blueshift to within a quantum down to S/N 5-7, so 10 is
                        # sufficient with margin.  Combining past it buys nothing for
                        # C IV and costs coverage and epoch purity.
SNR_FLOOR = 5.0         # below this the degrade-and-refit experiment showed the
                        # blueshift coming apart, so a C III] anchor on such a
                        # spectrum is worth nothing: NGC 4395 was handed a STIS visit
                        # at S/N 1 with 45 usable C IV pixels, and the fit moved 49
                        # quanta.  An anchored visit only outranks an unanchored one
                        # if it is also usable.
CIII_ANCHOR = 1950.0    # reach past C III] 1909 and the visit can anchor its own fit
CIII_COVER = 0.5        # ...and actually have data there: half of 1895-1925 A filled
GRID = np.arange(1260.0, 3001.0, 1.0)
MIN_GAP = 30.0          # a hole narrower than this is not worth splicing across
MIN_OVERLAP = 50.0      # a join needs this much shared wavelength to set a scale on
SPLICE_TOL = 0.25       # and the ratio across that overlap must be this consistent
                        # after a single scale factor: a constant offset is
                        # calibration and can be scaled out, structure in the ratio
                        # means the two are not describing the same spectrum
BLUE_EDGE = 1950.0      # a visit must reach past C III] 1909 to stand on its own: the
                        # ICA fit is judged by its agreement at C III] and Mg II, so a
                        # visit stopping at 1790 gives C IV with no anchor.  GTR: "we certainly don't need all the COS
                        # spectra that only cover 1500A or less without any nearby in
                        # time counterpart at longer wavelength" -- the ICA fit is
                        # constrained by the whole UV range, so blue-only data can only
                        # be used by splicing it to another epoch's red data, which is
                        # the thing we are trying to avoid.


def read_exposures(path, inst, z):
    out = []
    if inst == "FOS":
        for p0 in sorted(glob.glob(os.path.join(glob.escape(path), "*_c0f.fits"))):
            p1, p2 = p0.replace("_c0f", "_c1f"), p0.replace("_c0f", "_c2f")
            if not os.path.exists(p1):
                continue
            try:
                w = np.atleast_2d(fits.getdata(p0).astype(float))
                f = np.atleast_2d(fits.getdata(p1).astype(float))
                e = np.atleast_2d(fits.getdata(p2).astype(float)) if os.path.exists(p2) else None
                hd = fits.getheader(p0)
            except Exception:
                continue
            for k in range(min(w.shape[0], f.shape[0])):
                ek = e[k] if (e is not None and k < e.shape[0]) else np.full_like(f[k], np.nan)
                out.append(dict(file=os.path.basename(p0), row=k, w=w[k] / (1 + z), f=f[k], e=ek,
                                mjd=hd.get("EXPSTART", np.nan), exptime=hd.get("EXPTIME", np.nan),
                                grating=hd.get("FGWA_ID", hd.get("OPT_ELEM", "FOS"))))
        return out
    pats = ("*_x1d.fits", "*_sx1.fits")
    for p in sorted({x for q in pats for x in glob.glob(os.path.join(glob.escape(path), q))}):
        try:
            with fits.open(p, memmap=False) as h:
                hd0, hd1, t = h[0].header, h[1].header, h[1].data
                # 27 STIS files across nine objects are filed in COS directories;
                # without this the optical G430M exposures appear as COS visits
                # covering 4800-5100 A.
                got = str(hd0.get("INSTRUME", "")).strip().upper()
                if got and got != inst.upper():
                    continue
                if t is None or "WAVELENGTH" not in getattr(t, "names", []):
                    continue
                for k, r in enumerate(np.atleast_1d(t)):
                    w = np.asarray(r["WAVELENGTH"], float)
                    f = np.asarray(r["FLUX"], float)
                    e = np.asarray(r["ERROR"], float) if "ERROR" in t.names else np.full_like(f, np.nan)
                    g = np.isfinite(w) & (w > 500)
                    if g.sum() > 20:
                        out.append(dict(file=os.path.basename(p), row=k, w=w[g] / (1 + z),
                                        f=f[g], e=e[g],
                                        mjd=hd1.get("EXPSTART", hd0.get("EXPSTART", np.nan)),
                                        exptime=hd1.get("EXPTIME", hd0.get("EXPTIME", np.nan)),
                                        grating=hd0.get("OPT_ELEM", hd0.get("FILTER", "?"))))
        except Exception:
            continue
    return out


def per_file(rows):
    """Collapse detector segments / echelle orders to one record per exposure."""
    byfile = {}
    for r in rows:
        byfile.setdefault(r["file"], []).append(r)
    out = []
    for fn, rs in byfile.items():
        f = np.concatenate([r["f"] for r in rs])
        e = np.concatenate([r["e"] for r in rs])
        w = np.concatenate([r["w"] for r in rs])
        g = np.isfinite(f) & np.isfinite(e) & (e > 0)
        snr = float(np.median(f[g] / e[g])) if g.any() else np.nan
        civ = (w >= CIV[0]) & (w <= CIV[1]) & np.isfinite(f)
        wv = w[np.isfinite(w) & np.isfinite(f) & (f != 0)]
        out.append(dict(file=fn, mjd=rs[0]["mjd"], exptime=rs[0]["exptime"],
                        grating=rs[0]["grating"],
                        wmin=float(wv.min()) if wv.size else np.nan,
                        wmax=float(wv.max()) if wv.size else np.nan,
                        level=float(np.median(f[np.isfinite(f)])) if np.isfinite(f).any() else np.nan,
                        civ_level=float(np.median(f[civ])) if civ.sum() > 20 else np.nan,
                        civ_px=int(civ.sum()), snr=snr))
    return sorted(out, key=lambda r: (r["mjd"] if np.isfinite(r["mjd"]) else 0, r["file"]))


def assign_epochs(recs):
    n, last = 0, None
    for r in recs:
        if last is None or not np.isfinite(r["mjd"]) or (r["mjd"] - last) > GAP_DAYS:
            n += 1
        r["epoch"] = n
        if np.isfinite(r["mjd"]):
            last = r["mjd"]
    return recs


def alignment(all_recs, rows_by_file):
    """How well each exposure's SHAPE matches the object's best data.

    Level tests cannot see this.  An echelle order normalised differently, or a
    spectrum with a calibration slope, sits at the right average flux and still
    disagrees pointwise -- GTR's observation that STIS "passes our current tests
    but I bet it doesn't align well with the best single FOS or COS spectrum".

    The reference is the highest signal-to-noise exposure of the object, from
    whichever instrument.  Each exposure and the reference are divided by their
    own median over the range they share, so a genuine change in brightness
    between epochs does not register; what is left is disagreement in shape.
    Returns the robust scatter of that ratio, so 0.1 means the two agree to ten
    per cent pointwise and 1.0 means they are not describing the same spectrum.
    """
    # The reference must cover the range we care about.  Choosing simply the
    # highest-S/N exposure picked a STIS G430M file at 4300 A that had been
    # misfiled into a COS directory: nothing in the UV overlapped it, so the
    # statistic came back undefined for 88 of 94 exposures.
    live = [r for r in all_recs
            if np.isfinite(r["snr"]) and r.get("civ_px", 0) > 100]
    if len(live) < 2:
        live = [r for r in all_recs if np.isfinite(r["snr"])]
    if len(live) < 2:
        return {}
    ref_rec = max(live, key=lambda r: r["snr"])
    ref_rows = rows_by_file.get(ref_rec["file"], [])
    if not ref_rows:
        return {}
    rw = np.concatenate([q["w"] for q in ref_rows])
    rf = np.concatenate([q["f"] for q in ref_rows])
    o = np.argsort(rw)
    rw, rf = rw[o], rf[o]
    out = {}
    for r in all_recs:
        rows = rows_by_file.get(r["file"], [])
        if not rows or r["file"] == ref_rec["file"]:
            out[r["file"]] = 0.0 if r["file"] == ref_rec["file"] else np.nan
            continue
        w = np.concatenate([q["w"] for q in rows])
        f = np.concatenate([q["f"] for q in rows])
        oo = np.argsort(w)
        w, f = w[oo], f[oo]
        lo, hi = max(w.min(), rw.min()), min(w.max(), rw.max())
        if not np.isfinite(lo) or hi - lo < 20:
            out[r["file"]] = np.nan
            continue
        grid = np.linspace(lo, hi, 400)
        a = np.interp(grid, w, f)
        b = np.interp(grid, rw, rf)
        g = np.isfinite(a) & np.isfinite(b)
        if g.sum() < 50:
            out[r["file"]] = np.nan
            continue
        ma, mb = np.median(a[g]), np.median(b[g])
        if not (np.isfinite(ma) and np.isfinite(mb)) or ma == 0 or mb == 0:
            out[r["file"]] = np.nan
            continue
        ratio = (a[g] / ma) / (b[g] / mb)
        out[r["file"]] = float(np.median(np.abs(ratio - 1.0)) / 0.6745)
    return out


def recommend(recs):
    """Attach a proposed action to each exposure, and an epoch verdict.

    The comparison is made within a grating and across epochs.  Comparing raw
    flux levels between gratings is meaningless -- they cover different
    wavelengths -- and comparing only within an epoch hides a whole failed
    visit, because such a visit is perfectly consistent with itself.  ARK 120's
    first FOS visit is the worked example: every exposure in it sits 20-50 times
    below the same grating's second visit at S/N 1-3.
    """
    by_ep = {}
    for r in recs:
        by_ep.setdefault((r["epoch"], r["grating"]), []).append(r)
    ep_level = {}
    for key, group in by_ep.items():
        lv = np.array([g["level"] for g in group], float)
        ep_level[key] = np.nanmedian(lv[np.isfinite(lv)]) if np.isfinite(lv).any() else np.nan
    best = {}
    for (ep, gr), lev in ep_level.items():
        if np.isfinite(lev) and (gr not in best or lev > best[gr]):
            best[gr] = lev
    # Two different questions, kept apart, because conflating them threw away
    # most of the data on every variable object: judging each exposure against
    # the single brightest epoch flagged 30 of NGC 3783's 49 COS exposures and
    # 730 of NGC 5548's 798, none of which is bad data.
    #
    #   1. Is this exposure sound?  Answered WITHIN its own visit -- does it
    #      agree with the exposures taken beside it -- plus a test against the
    #      best the object ever delivered, which only fires when an exposure has
    #      essentially no signal at all.
    #   2. Which epoch should be fitted?  A separate decision, reported once for
    #      the object rather than as a verdict on each exposure.
    by_grating = {}
    for r in recs:
        by_grating.setdefault(r["grating"], []).append(r)

    # First pass: exposures with no signal, judged against the best this object
    # and grating ever gave.  These are excluded from every level that follows,
    # so a half-dead visit does not define its own normal.
    for r in recs:
        ref_best = best.get(r["grating"], np.nan)
        r["grating_best_level"] = ref_best
        r["action"] = None
        if str(r["grating"]).upper() in LOWRES_GRATINGS and len(best) > 1:
            r["action"] = "drop: resolution too low for the ICA fit"
        elif (np.isfinite(r["snr"]) and r["snr"] < SNR_DEAD and np.isfinite(ref_best)
                and ref_best > 0 and np.isfinite(r["level"])
                and abs(r["level"]) < ref_best / LEVEL_DEAD):
            r["action"] = "drop: no signal"

    # Second pass: agreement with the rest of its own visit and grating.
    for key, group in by_ep.items():
        live = [g for g in group if g["action"] is None and np.isfinite(g["level"])]
        med = float(np.median([g["level"] for g in live])) if live else np.nan
        tol = LEVEL_TOL_MANY if len(live) >= MANY_EXPOSURES else LEVEL_TOL
        # `level` is the median flux over the whole exposure, so for an object
        # whose continuum is absorbed to zero over most of a grating's range the
        # reference median comes out NEGATIVE and the ratio is meaningless: on
        # Mrk 231 (index 54) the COS G140L exposures sit at about -9e-19 and the
        # test reported levels of -492x and -732x, keeping an arbitrary 5 of 22
        # exposures that were in truth indistinguishable.  A ratio of two numbers
        # straddling zero carries no information about agreement, so where the
        # reference is not positive the test is skipped rather than acted on.
        # The S/N and dead-exposure tests still apply.
        # LEVEL_GUARD exists so the change can be run both ways when assessing
        # what it alters for objects already processed; it is not a tuning knob.
        ratio_usable = np.isfinite(med) and (med > 0 or not LEVEL_GUARD)
        for g in group:
            g["epoch_level"] = med
            if LEVEL_GUARD:
                ratio = (g["level"] / med) if (ratio_usable and np.isfinite(g["level"])
                                               and g["level"] > 0) else np.nan
            else:
                ratio = (g["level"] / med) if (np.isfinite(med) and med != 0
                                               and np.isfinite(g["level"])) else np.nan
            g["level_ratio"] = ratio
            if g["action"] is not None:
                continue
            if not ratio_usable:
                g["action"] = "keep"
                g["level_test"] = "skipped: visit median flux %.3g is not positive" % med
            elif np.isfinite(ratio) and not ((1 - tol) <= ratio <= 1 / (1 - tol)):
                g["action"] = "review: level %.2fx the rest of this visit (tol %.0f%%)" % (
                    ratio, 100 * tol)
            else:
                g["action"] = "keep"

    # An exposure follows its visit -- but only if it is itself in question.  One
    # sitting at its visit's normal level stays: NGC 3783's fifth visit has eight
    # dead exposures and six sound ones at the usual flux, and dropping those six
    # for company would be indefensible.
    by_visit = {}
    for r in recs:
        by_visit.setdefault(r["epoch"], []).append(r)
    for key, group in by_visit.items():
        bad = [g for g in group if g["action"].startswith("drop")]
        if group and len(bad) >= max(2, (len(group) + 1) // 2):
            for g in group:
                if g["action"].startswith("drop"):
                    continue
                # The test is the level the OBJECT normally reaches, not the
                # level of the visit -- a mostly-dead visit defines a useless
                # normal.  ARK 120's lone survivor sits at 0.02x what the object
                # gives elsewhere and goes with its visit; NGC 3783's six sound
                # exposures sit at the usual flux beside eight dead ones and stay.
                # Use the same factor as the no-signal test.  A factor of two
                # was too tight: NGC 3783 varies by nearly three between epochs,
                # so four sound exposures at 0.47 of the object's best were being
                # dropped for keeping company with eight dead ones.
                ref = g.get("grating_best_level", np.nan)
                if np.isfinite(ref) and ref > 0 and np.isfinite(g["level"]) \
                        and abs(g["level"]) < ref / LEVEL_DEAD:
                    g["action"] = "drop: rest of this visit is being dropped"

    # Shape agreement is RECORDED but not acted on.  As built it compares raw
    # pixels between spectra of different resolution, so a COS G160M exposure
    # judged against a FOS reference scores 26-65% disagreement from resolution
    # and noise alone -- it flagged 38 of NGC 3783's 42 STIS exposures and
    # coloured the stacked figures misleadingly.  It needs both spectra smoothed
    # to a common resolution and the residual judged against what the errors
    # predict before it can carry a verdict.
    for r in recs:
        pass

    # epoch-level verdict, over what survives
    live = [r for r in recs if r["action"] == "keep" and np.isfinite(r["civ_level"])]
    ep_lev = {}
    for r in live:
        ep_lev.setdefault((r["epoch"], r["grating"]), []).append(r["civ_level"])
    levels = {k: float(np.median(v)) for k, v in ep_lev.items() if np.isfinite(np.median(v))}
    verdict = ""
    pos = [v for v in levels.values() if v > 0]
    if len(pos) >= 2 and max(pos) / min(pos) > EPOCH_SPAN:
        best = max(levels, key=lambda k: len([r for r in live if (r["epoch"], r["grating"]) == k]))
        verdict = ("epochs differ by %.1fx -- choose one (most exposures: epoch %d, %s)"
                   % (max(pos) / min(pos), best[0], best[1]))
    # Flag visits that cover C IV and nothing redward, with no contemporaneous
    # visit supplying the rest of the range.
    spans = {}
    for r in recs:
        e = spans.setdefault(r["epoch"], dict(hi=-np.inf, mjd=r["mjd"]))
        hi = r.get("wmax", np.nan)
        if np.isfinite(hi):
            e["hi"] = max(e["hi"], hi)
    for r in recs:
        e = spans.get(r["epoch"], {})
        hi, mjd = e.get("hi", np.nan), e.get("mjd", np.nan)
        partner = any(np.isfinite(v.get("hi", np.nan)) and v["hi"] > BLUE_EDGE
                      and np.isfinite(v.get("mjd", np.nan)) and np.isfinite(mjd)
                      and abs(v["mjd"] - mjd) <= GAP_DAYS
                      for k, v in spans.items() if k != r["epoch"])
        r["blue_only"] = int(np.isfinite(hi) and hi <= BLUE_EDGE and not partner)
    return recs, verdict



OVERRIDES = os.path.join(HERE, "pipeline_output", "final", "spectrum_overrides.csv")


def _overrides():
    """Hand choices of which spectrum to fit, recorded rather than done silently.

    The rules pick the visit for almost every object, but GTR will sometimes
    disagree with one, and a choice made by hand has to be reproducible and
    explainable to a referee.  One row per object: index, instrument, epoch, the
    reason, and when.  An override always wins, and the reason travels with the
    measurement into the provenance record.

    `donor_inst` / `donor_epoch` splice a second instrument's visit on below
    where the chosen one starts, for the object whose chosen visit is the only
    one carrying the red anchors while a different instrument is the only one
    carrying the blue end.  See the note on those fields below; the rule that
    makes it checkable is that the donor supplies only wavelengths the chosen
    visit does not cover.

    `drop_gratings` is a space-separated list of gratings to exclude from the
    chosen visit.  It exists because a visit can be the right visit and still
    carry the wrong data: Mrk 1044's STIS visit pairs two G140L exposures with
    eight narrow G140M strips, and co-adding resolutions that differ by a factor
    of ten puts a flux step across C IV rather than adding signal.  Dropping the
    strips is a choice about which data to trust, so it is recorded here with
    the rest of them rather than made by editing a co-add by hand.
    """
    out = {}
    if not os.path.exists(OVERRIDES):
        return out
    try:
        for r in csv.DictReader(open(OVERRIDES)):
            if not str(r.get("index", "")).strip():
                continue
            # `epoch` may name more than one, separated by spaces or commas, for
            # the case where a second epoch exists only to fill a hole the first
            # cannot cover.  Mrk 841's chosen COS visit has no exposure at all
            # across 1534-1542 A rest -- the detector segment gap, which its four
            # FP-POS offsets fail to close -- and the other epoch covers it.  The
            # co-adder continuum-normalises each exposure before combining, so
            # the second epoch contributes shape and not flux level, and its
            # inverse-variance weight is negligible wherever the first has data.
            # int(float(x)) rather than int(x): a pandas round-trip of this file
            # turns a column holding both blanks and numbers into floats, so
            # "3" is rewritten "3.0" and a bare int() raises.  The except below
            # then swallowed it and returned an EMPTY override table -- every
            # hand choice in this file silently stopped applying, with nothing
            # printed.  Found on 2026-10-06 when a splice reported "no donor
            # named" for a row that plainly named one.
            def _ints(v):
                return tuple(int(float(x)) for x in
                             str(v or "").replace(",", " ").split()
                             if x not in ("nan", "None"))
            _ep = _ints(r.get("epoch", ""))
            out[int(r["index"])] = dict(inst=str(r.get("inst", "")).strip(),
                                        epoch=_ep[0] if _ep else None,
                                        epochs=_ep,
                                        reason=str(r.get("reason", "")).strip(),
                                        drop_gratings=tuple(
                                            str(r.get("drop_gratings", "")).upper().split()),
                                        # `drop_files` names individual exposures to
                                        # exclude, by rootname, where neither the
                                        # instrument, the epoch nor the grating
                                        # separates the good from the bad.  Mrk 205's
                                        # two H27 exposures differ in flux level by a
                                        # factor of 430 -- 4.99e-12 against 1.16e-14,
                                        # where its other gratings sit at 1.6-2.0e-14 --
                                        # and the screening lost both because the level
                                        # test compared each against the median of that
                                        # two-element pair, which the bad one poisoned.
                                        drop_files=tuple(
                                            str(r.get("drop_files", "")).replace(",", " ").split()),
                                        # `keep_files` forces named exposures back in
                                        # after the screening has set them aside for
                                        # review.  The level test is a comparison with
                                        # the median of an exposure's own epoch-and-
                                        # grating group, so where that group has two
                                        # members and one is bad, BOTH are flagged and
                                        # the good one has to be reinstated by name.
                                        keep_files=tuple(
                                            str(r.get("keep_files", "")).replace(",", " ").split()),
                                        # `donor_inst` and `donor_epoch` name a SECOND
                                        # instrument's visit to splice on below where
                                        # the chosen one starts.  Not a co-addition: the
                                        # donor contributes only at wavelengths the
                                        # chosen visit does not cover, so no pixel is
                                        # ever averaged between instruments.  LEDA 50824
                                        # (index 90) is one STIS G230L exposure running
                                        # 1451-2899 A, so it carries C III] and Mg II and
                                        # no Si IV at all; COS epoch 3 supplies 1120-1447
                                        # and nothing above it.  Co-adding them instead
                                        # would mix R~700 with R~16000 across C IV, the
                                        # defect of index 17 and index 28, and measurably
                                        # did: 31 per cent off the EW.  The donor's
                                        # exposures are chosen by the same screening the
                                        # host's are, so the audit trail is the same;
                                        # `donor_files` overrides that by rootname only
                                        # where the screening is wrong about the donor,
                                        # exactly as keep_files does for the host.
                                        donor_inst=str(r.get("donor_inst", "")).strip(),
                                        donor_epochs=_ints(r.get("donor_epoch", "")),
                                        donor_files=tuple(
                                            str(r.get("donor_files", "")).replace(",", " ").split()),
                                        # The join is DERIVED -- the chosen visit's
                                        # bluest usable pixel -- so there is nothing to
                                        # get wrong.  `join` overrides it only where the
                                        # chosen visit's own blue edge is junk and the
                                        # donor should run further red.
                                        join=(float(r["join"])
                                              if str(r.get("join", "")).strip() else None),
                                        when=str(r.get("when", "")).strip())
    except Exception as exc:
        # Never fail silently here.  Returning {} means every hand choice in
        # spectrum_overrides.csv stops applying -- the wrong visit, the wrong
        # gratings, no splice -- and the pass carries on producing plausible
        # numbers for the wrong spectra.
        print("SPECTRUM OVERRIDES NOT LOADED: %s: %s -- every hand choice in %s "
              "is being IGNORED" % (type(exc).__name__, exc, OVERRIDES), flush=True)
        return {}
    return out


def choose_visit(per_inst, rows_by_file, index=None):
    """Which single visit should this object be fitted on?

    GTR's default is one self-sufficient visit rather than a combination across
    visits, and what makes a visit self-sufficient is coverage as much as depth:
    the fit is judged at C III] and Mg II, so a visit stopping short of C III]
    gives C IV with no anchor however deep it is.  Preference order is a visit
    covering C III] at sufficient S/N, then the best C IV visit recorded as
    having no anchor, and only then extending coverage from another epoch.

    Returns a list of candidate dicts, best first, each carrying why.
    """
    cands = []
    for inst, recs in per_inst.items():
        keep = [r for r in recs if str(r["action"]).startswith("keep")]
        by_ep = {}
        for r in keep:
            by_ep.setdefault(r["epoch"], []).append(r)
        for ep, g in by_ep.items():
            wmax = max((r.get("wmax", np.nan) for r in g), default=np.nan)
            wmin = min((r.get("wmin", np.nan) for r in g), default=np.nan)
            civ = max((r.get("civ_px", 0) for r in g), default=0)
            if not civ:
                continue                      # no C IV: cannot be the measurement

            # Coverage has to be what the visit actually fills, not the distance
            # from its bluest pixel to its reddest.  A STIS echelle visit is a
            # comb of short orders: NGC 4151's visit 11 spans 1137-3108 A and was
            # ranked as covering C III] when it is snippets with gaps between
            # them, while FOS visits with continuous coverage were passed over.
            occ = np.zeros(GRID.size, dtype=bool)
            for r in g:
                for q in rows_by_file.get(r["file"], []):
                    w = np.asarray(q["w"], float); f = np.asarray(q["f"], float)
                    ok = np.isfinite(w) & np.isfinite(f) & (f != 0)
                    if not ok.any():
                        continue
                    j = np.searchsorted(GRID, w[ok])
                    j = j[(j >= 0) & (j < GRID.size)]
                    occ[j] = True
            def frac(lo, hi):
                m = (GRID >= lo) & (GRID <= hi)
                return float(occ[m].mean()) if m.any() else 0.0
            civ_cov = frac(*CIV)
            ciii_cov = frac(1895.0, 1925.0)
            filled = frac(1260.0, min(3000.0, wmax if np.isfinite(wmax) else 3000.0))
            s1 = float(np.nanmedian([r["snr"] for r in g]))
            snr = s1 * np.sqrt(len(g))
            cands.append(dict(inst=inst, epoch=ep, n_exp=len(g),
                              mjd=float(np.nanmin([r["mjd"] for r in g])),
                              wmin=wmin, wmax=wmax, snr_single=s1, snr_combined=snr,
                              civ_cov=civ_cov, ciii_cov=ciii_cov, filled=filled,
                              has_ciii=bool(ciii_cov >= CIII_COVER),
                              meets_snr=bool(snr >= SNR_TARGET)))
    # anchored and deep enough first, then anchored, then deep enough, then the rest;
    # within each tier the fewest exposures that do the job, then the best S/N.
    def rank(c):
        usable = c["snr_combined"] >= SNR_FLOOR
        tier = (0 if (c["has_ciii"] and c["meets_snr"]) else
                1 if (c["has_ciii"] and usable) else
                2 if c["meets_snr"] else
                3 if usable else
                4 if c["has_ciii"] else 5)
        # Within a tier: better-filled coverage first, then the DEEPEST visit.
        # Ranking on fewest exposures was right for choosing a clean epoch and
        # plainly wrong here -- it handed NGC 4151 a 5-exposure visit at S/N 10
        # over an 18-exposure one at S/N 442 with identical coverage, and NGC 3516
        # a visit at S/N 26 over three others at S/N 33-40.
        return (tier, -round(c["filled"], 2), -c["snr_combined"])
    cands.sort(key=rank)
    ov = _overrides().get(index) if index is not None else None
    if ov and ov.get("inst"):
        want = ov.get("epochs") or ()
        hit = [c for c in cands
               if c["inst"] == ov["inst"] and (not want or c["epoch"] in want)]
        if hit:
            pick = hit[0]
            pick["overridden"] = True
            pick["override_reason"] = ov["reason"]
            cands = [pick] + [c for c in cands if c is not pick]
    for c in cands:
        if c.get("overridden"):
            c["why"] = "chosen by hand: %s" % (c.get("override_reason") or "no reason recorded")
            continue
        usable = c["snr_combined"] >= SNR_FLOOR
        c["why"] = ("covers C III] and reaches S/N %.0f" % c["snr_combined"]
                    if c["has_ciii"] and c["meets_snr"] else
                    "covers C III] at S/N %.0f" % c["snr_combined"]
                    if (c["has_ciii"] and usable) else
                    "S/N %.0f but stops at %.0f A -- no C III] anchor"
                    % (c["snr_combined"], c["wmax"]) if c["meets_snr"] else
                    "S/N %.0f, no C III] anchor" % c["snr_combined"] if usable else
                    "only S/N %.1f -- below the usable floor" % c["snr_combined"])
    return cands


def _occupancy(recs, rows_by_file):
    occ = np.zeros(GRID.size, dtype=bool)
    for r in recs:
        for q in rows_by_file.get(r["file"], []):
            w = np.asarray(q["w"], float); f = np.asarray(q["f"], float)
            ok = np.isfinite(w) & np.isfinite(f) & (f != 0)
            if not ok.any():
                continue
            j = np.searchsorted(GRID, w[ok])
            j = j[(j >= 0) & (j < GRID.size)]
            occ[j] = True
    return occ


def _interp(recs, rows_by_file):
    w = np.concatenate([np.asarray(q["w"], float)
                        for r in recs for q in rows_by_file.get(r["file"], [])] or [np.array([])])
    f = np.concatenate([np.asarray(q["f"], float)
                        for r in recs for q in rows_by_file.get(r["file"], [])] or [np.array([])])
    ok = np.isfinite(w) & np.isfinite(f) & (f != 0)
    if ok.sum() < 20:
        return None
    o = np.argsort(w[ok])
    return np.interp(GRID, w[ok][o], f[ok][o], left=np.nan, right=np.nan)


def plan_splice(primary, per_inst, rows_by_file):
    """What, if anything, should be added to the chosen visit to fill its gaps.

    GTR's third step, and the one the single-visit rule cannot do on its own:
    "fill in any wavelength gaps with any spectra covering those gaps so long as
    they don't corrupt the overlap regions."  A candidate may extend coverage
    only if it shares enough wavelength with the primary to set a scale on, and
    agrees with it across that overlap once a single scale factor is removed.
    The scale is recorded; a constant offset is calibration, structure in the
    ratio means the two are not the same spectrum and the join is refused.
    """
    prim = [r for r in per_inst[primary["inst"]]
            if r["epoch"] == primary["epoch"] and str(r["action"]).startswith("keep")]
    occ_p = _occupancy(prim, rows_by_file)
    f_p = _interp(prim, rows_by_file)
    if f_p is None:
        return []
    gaps, i = [], 0
    while i < GRID.size:
        if occ_p[i]:
            i += 1; continue
        j = i
        while j < GRID.size and not occ_p[j]:
            j += 1
        if GRID[j - 1] - GRID[i] >= MIN_GAP:
            gaps.append((GRID[i], GRID[j - 1]))
        i = j
    out = []
    for lo, hi in gaps:
        best = None
        for inst, recs in per_inst.items():
            for ep in sorted({r["epoch"] for r in recs}):
                g = [r for r in recs if r["epoch"] == ep
                     and str(r["action"]).startswith("keep")]
                if not g or (inst == primary["inst"] and ep == primary["epoch"]):
                    continue
                occ_c = _occupancy(g, rows_by_file)
                m = (GRID >= lo) & (GRID <= hi)
                fill = occ_c[m].mean() if m.any() else 0.0
                if fill < 0.6:
                    continue
                both = occ_c & occ_p
                if both.sum() < MIN_OVERLAP:
                    continue
                f_c = _interp(g, rows_by_file)
                if f_c is None:
                    continue
                ok = both & np.isfinite(f_c) & np.isfinite(f_p) & (f_p != 0)
                if ok.sum() < MIN_OVERLAP:
                    continue
                ratio = f_c[ok] / f_p[ok]
                scale = float(np.median(ratio))
                if not np.isfinite(scale) or scale == 0:
                    continue
                agree = float(np.median(np.abs(ratio / scale - 1.0)) / 0.6745)
                snr = float(np.nanmedian([r["snr"] for r in g])) * np.sqrt(len(g))
                cand = dict(inst=inst, epoch=ep, lo=lo, hi=hi, n_exp=len(g),
                            fill=float(fill), overlap=int(ok.sum()), scale=scale,
                            agreement=agree, snr=snr, ok=bool(agree <= SPLICE_TOL))
                if cand["ok"] and (best is None or snr > best["snr"]):
                    best = cand
        out.append(best if best else dict(lo=lo, hi=hi, inst=None,
                                          ok=False, why="nothing agrees across the overlap"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "pipeline_output", "final",
                                                  "exposure_decisions.csv"))
    a = ap.parse_args()
    queue = {int(r["order"]): r for r in csv.DictReader(
        open(os.path.join(HERE, "pipeline_output", "work_queue.csv")))}
    dec = pd.read_csv(os.path.join(HERE, "pipeline_output", "final",
                                   "pass_decisions.csv")).set_index("index")
    allrows = []
    for i in a.index:
        name = queue[i]["name_mast_key"]
        z = float(dec.loc[i, "z_adopted"])
        nm = dec.loc[i, "name_pub"] if i in dec.index else name
        print("\n%03d  row %s  %s   (z = %.5f)"
              % (i, dec.loc[i, "row_num"] if i in dec.index else "?", nm, z))
        base = os.path.join(MAST, name)
        # Read every instrument first: the alignment reference is the object's
        # best exposure wherever it came from, because GTR's question is whether
        # the STIS data agree with the best FOS or COS spectrum, not whether the
        # STIS exposures agree among themselves.
        per_inst, rows_all, recs_all = {}, {}, []
        for inst in sorted(os.listdir(base)):
            d = os.path.join(base, inst)
            if not os.path.isdir(d):
                continue
            raw = read_exposures(d, inst, z)
            recs = per_file(raw)
            if not recs:
                continue
            for q in raw:
                rows_all.setdefault(q["file"], []).append(q)
            recs = assign_epochs(recs)
            for r in recs:
                r["inst"] = inst
            per_inst[inst] = recs
            recs_all += recs
        al = alignment(recs_all, rows_all)
        for inst, recs in per_inst.items():
            for r in recs:
                r["align_rms"] = al.get(r["file"], np.nan)
            recs, verdict = recommend(recs)
            per_inst[inst] = recs
            keep = sum(1 for r in recs if r["action"] == "keep")
            print("  %-5s %d exposures, %d epoch(s): keep %d, drop %d%s"
                  % (inst, len(recs), max(r["epoch"] for r in recs), keep, len(recs) - keep,
                     ("   " + verdict) if verdict else ""))
            for r in recs:
                if r["action"] != "keep":
                    print("       %-22s epoch %d  %-6s  S/N %6.2f  level %9.3g (epoch %9.3g)  -> %s"
                          % (r["file"], r["epoch"], r["grating"], r["snr"], r["level"],
                             r["epoch_level"], r["action"]))
                r.update(index=i, inst=inst, name=name, verdict=verdict)
                allrows.append(r)
        cands = choose_visit(per_inst, rows_all, index=i)
        if cands:
            best = cands[0]
            print("  -> fit %s visit %d: %d exposure(s), MJD %.0f, rest %.0f-%.0f A, "
                  "S/N ~%.0f  (%s)"
                  % (best["inst"], best["epoch"], best["n_exp"], best["mjd"],
                     best["wmin"], best["wmax"], best["snr_combined"], best["why"]),
                  flush=True)
            for c in cands[1:4]:
                print("     next: %-5s v%-2d %3d exp  rest %.0f-%.0f  S/N ~%.0f  (%s)"
                      % (c["inst"], c["epoch"], c["n_exp"], c["wmin"], c["wmax"],
                         c["snr_combined"], c["why"]), flush=True)
            for r in allrows:
                if r.get("index") == i:
                    r["chosen"] = int(r.get("inst") == best["inst"]
                                      and r.get("epoch") == best["epoch"])

    pd.DataFrame(allrows).to_csv(a.out, index=False)
    print("\n-> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

