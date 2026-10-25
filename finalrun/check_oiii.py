"""Systemic redshift from [O III] 5007/4959 in the SDSS spectrum.

Why this exists: from index 286 upward every object has an SDSS spectrum that
reaches Mg II, and for many of them the HST spectrum cannot measure its own
frame -- the C III] cross-correlation is often at S/N 1-2, Mg II is frequently
uncovered, and "no redshift check is possible" has been the limiting caveat on
a long run of rows.  [O III] is a far better frame measurement than either.

Two lessons from earlier objects are built in.

A SINGLE GAUSSIAN IS NOT ENOUGH.  On index 197 a single Gaussian put the
catalogue redshift ~175 km/s too high; decomposing into core plus wing showed
the core at -5 +/- 12 km/s and the apparent offset was the outflow wing pulling
the centroid.  [O III] routinely has a blueshifted wing, so the core is what
measures the systemic frame and the wing is a separate physical quantity.

THE CORE MUST ACTUALLY BE A CORE.  A fitted "core" wider than CORE_SIGMA_MAX
is not a narrow-line region measurement and should not be used as a frame.
That threshold is reported, not enforced -- the caller decides.

Both lines are fitted together at the fixed 2.98 intensity ratio and a shared
velocity, on a local linear continuum from side bands either side of the pair.

Usage:  python check_oiii.py <index> [<index> ...]
        python check_oiii.py --all-from 286
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
SDSS_LITE = os.path.join(HERE, "data_v23", "SDSS_spec", "lite")
SDSS_TABLE = os.path.join(HERE, "pipeline_output", "final", "sdss_spectra.csv")
C = 299792.458
OIII_A, OIII_B = 5008.240, 4960.295          # vacuum

#: Widest fitted core still accepted as a narrow-line-region measurement, km/s.
#: RAISED FROM 150 TO 200 ON 2026-10-15, by GTR ("I think that 200 km/s is fine"),
#: after the 150 km/s value was measured against the pass rather than assumed.
#: Of 87 objects with an [O III] measurement over indices 244-330, 35 certified,
#: 31 failed for substantive reasons (a degenerate decomposition, a spurious
#: narrow spike, or an ill-determined centroid) and 21 failed ONLY on this
#: width test.  Those 21 were not errors the guard caught: their centroids are
#: determined to a median of +/-11 km/s, they sit a median of 14 km/s from the
#: catalogue redshift, and EVERY ONE of them agrees with the catalogue to within
#: two 69 km/s quanta.  18 of the 21 have sigma between 150 and 200, which is
#: still plausible narrow-line-region width.  Raising the threshold therefore
#: changes no adopted redshift in this range -- it stops the report calling 21
#: confirmed frames uncertified.  A core wider than this is still refused, and
#: the other three guards (degenerate_two, spurious_core, core_dv_err) are
#: untouched, so the failure modes that matter still fail.
CORE_SIGMA_MAX = 200.0

#: A core/wing split means nothing when the two components have similar widths
#: AND the narrower one is not the dominant one.  GTR caught this on index 344:
#: the fit returned sigma 147.0 and sigma 213.6 km/s -- a ratio of 1.45 -- with
#: the BROADER component carrying 77 per cent of the flux and 2.25 times the
#: amplitude, so the report labelled the weaker, narrower half the "core" and
#: the dominant half the "wing", and the figure looked flipped.  Nothing was
#: flipped: the narrower-is-core rule did exactly what it says.  The rule just
#: does not apply when the fit has split one line into two comparable halves
#: instead of separating a narrow core from a broad wing.  Distinct from the
#: index-296 defect, where plot() re-derived the assignment and genuinely
#: disagreed with measure().
#:
#: BOTH conditions are required, and the thresholds were calibrated against the
#: pass rather than assumed.  Width alone is not enough: index 289, whose core
#: was used to override a redshift (RM782), has a ratio of 1.43 -- but its core
#: carries 90 per cent of the flux, so "core" is exactly the right word for it.
#: A width-only guard at 1.8 would have stripped certification from 17 objects
#: including that one.  With the dominance condition added, ratio < 1.5 and core
#: flux < 50 per cent flags 5 objects over indices 244-352, NONE of which
#: certify on the other guards.  So this changes no verdict: it explains why a
#: decomposition that was already refused is not worth reading as core-plus-wing.
CORE_WING_RATIO_MIN = 1.5
CORE_WING_FRAC_MAX = 0.50
RATIO = 2.98                                  # 5007/4959


def _doublet(x, amp, sig, dv):
    la = OIII_A * (1.0 + dv / C)
    lb = OIII_B * (1.0 + dv / C)
    return (amp * np.exp(-0.5 * ((x - la) / sig) ** 2)
            + amp / RATIO * np.exp(-0.5 * ((x - lb) / sig) ** 2))


def measure(index, verbose=True):
    """Return a dict of [O III] results for this index, or None."""
    sd = pd.read_csv(SDSS_TABLE)
    row = sd[sd["index"].astype("Int64") == int(index)]
    if not len(row):
        return None
    row = row.iloc[0]
    #: Measure against the redshift ACTUALLY IN USE, not the catalogue value.
    #: sdss_spectra.csv carries z_queue, which is the pre-override catalogue
    #: number; sixteen objects now carry an override, and for those the dv this
    #: returns was being quoted against a frame the fit no longer uses.  On
    #: index 217 that showed up as a spurious +855 km/s when the adopted value
    #: is already 628 km/s above the catalogue.
    z_cat = float(row["z_queue"])
    try:
        import sys as _sys
        _sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
        from ica import manual_fix_store as _S
        _cv = pd.read_csv(os.path.join(HERE, "pipeline_output", "final",
                                       "chosen_visit_summary.csv"))
        _hit = _cv[_cv["index"].astype("Int64") == int(index)]
        if len(_hit):
            z_cat, _src = _S.resolve_redshift(str(_hit.iloc[0]["stem"]), z_cat)
    except Exception:
        pass
    fn = str(row.get("fn_sdss") or "").strip()
    path = os.path.join(SDSS_LITE, fn)
    if not fn or not os.path.exists(path):
        return None
    with fits.open(path) as h:
        d = h[1].data
        lam = 10.0 ** np.asarray(d["loglam"], float)
        fl = np.asarray(d["flux"], float)
        iv = np.asarray(d["ivar"], float)
    rest = lam / (1.0 + z_cat)
    fit = (rest > 4940) & (rest < 5080) & (iv > 0)
    side = (((rest > 4890) & (rest < 4940)) | ((rest > 5090) & (rest < 5140))) & (iv > 0)
    if fit.sum() < 25 or side.sum() < 8:
        return None
    coeff = np.polyfit(rest[side], fl[side], 1)
    x = rest[fit]
    y = fl[fit] - np.polyval(coeff, x)
    sg = 1.0 / np.sqrt(iv[fit])
    peak_snr = float(np.nanmax(y) / np.nanmedian(sg))
    out = dict(index=int(index), name=str(row.get("name_pub", "")), z_cat=z_cat,
               fn=fn, peak_snr=peak_snr, n_px=int(fit.sum()))
    try:
        p1, c1 = curve_fit(_doublet, x, y, p0=[max(y.max(), 1e-3), 3.0, 0.0],
                           sigma=sg, maxfev=60000)
        e1 = np.sqrt(np.diag(c1))
        out["single_dv"] = float(p1[2]); out["single_dv_err"] = float(e1[2])
        out["single_sigma"] = float(abs(p1[1]) / OIII_A * C)
        out["single_chi2"] = float(np.sum(((y - _doublet(x, *p1)) / sg) ** 2) / (len(x) - 3))
    except Exception:
        return out
    try:
        def two(xx, a1, s1, d1, a2, s2, d2):
            return _doublet(xx, a1, abs(s1), d1) + _doublet(xx, a2, abs(s2), d2)
        p2, c2 = curve_fit(two, x, y,
                           p0=[y.max(), 1.5, 0.0, y.max() * 0.3, 8.0, -300.0],
                           sigma=sg, maxfev=200000)
        e2 = np.sqrt(np.diag(c2))
        #: The core is the NARROWER component -- the standard [O III]
        #: decomposition, a narrow core plus a broader outflow wing -- with one
        #: guard, because the sort key alone is not enough.
        #:
        #: On index 245 the fit found a 21.7 km/s spike carrying 1.4 per cent of
        #: the flux at dv -355.7, and calling it the core put the reported
        #: frame 334 km/s from where the line actually is; the real narrow line
        #: (sigma 128, dv -21.2, 98.6 per cent of the flux) agrees with the
        #: single Gaussian to 1 km/s and sits at the catalogue redshift.  GTR
        #: caught that from the figure.
        #:
        #: Keying on integrated flux instead fixes 245 and breaks 244, where the
        #: broad sigma 385 wing carries more flux than the sigma 130 core but is
        #: obviously not the narrow line.  So: narrower component is the core,
        #: and if it carries less than 15 per cent of the flux it is a spurious
        #: spike -- the decomposition is flagged and the DOMINANT component is
        #: reported as the better estimate, since that is the line one would
        #: measure by eye.
        _a = (abs(p2[1]), p2[0], p2[2], e2[2])
        _b = (abs(p2[4]), p2[3], p2[5], e2[5])
        (sc, ac, dc, ec), (sw, aw, dw, ew) = ((_a, _b) if _a[0] <= _b[0] else (_b, _a))
        _fc, _fw = ac * sc, aw * sw
        #: A flux FRACTION only means anything while both components are
        #: emission.  Where the two-Gaussian fit is degenerate one amplitude
        #: goes negative and the ratio stops being a fraction at all: index 304
        #: ([HB89] 1543+489, peak S/N 9.2) reported "a spike carrying -19450.5%
        #: of the flux".  The guards rejected it anyway, for its 543 km/s core
        #: and its +/-4883 km/s dv, but the number printed was meaningless.
        #: Call the decomposition failed instead, and say so.
        #: Widths too close for the split to mean anything -- see
        #: CORE_WING_RATIO_MIN.  Recorded alongside the values rather than
        #: instead of them, so a reader can still see what the fit returned.
        #: Computed from the ORIGINAL narrower/broader assignment, before the
        #: spurious_core swap below may exchange them -- the question is whether
        #: the fit separated a narrow core from a broad wing at all, which is a
        #: property of the two fitted components and not of their labels.
        out["core_wing_ratio"] = float(abs(sw) / abs(sc)) if sc else float("nan")
        _ratio_frac = (_fc / (_fc + _fw)) if (_fc + _fw) else np.nan
        out["widths_too_close"] = bool(
            np.isfinite(out["core_wing_ratio"])
            and out["core_wing_ratio"] < CORE_WING_RATIO_MIN
            and np.isfinite(_ratio_frac) and _ratio_frac < CORE_WING_FRAC_MAX)
        out["degenerate_two"] = bool(ac <= 0 or aw <= 0)
        if out["degenerate_two"]:
            out["core_flux_frac"] = float("nan")
            out["spurious_core"] = False
            #: A degenerate fit has no meaningful widths either, but saying so
            #: twice helps nobody -- degeneracy is the more fundamental fault
            #: and is reported on its own.
            out["widths_too_close"] = False
        else:
            _frac = _fc / (_fc + _fw) if (_fc + _fw) else np.nan
            out["core_flux_frac"] = float(_frac)
            if np.isfinite(_frac) and _frac < 0.15:
                out["spurious_core"] = True
                (sc, ac, dc, ec), (sw, aw, dw, ew) = ((sw, aw, dw, ew), (sc, ac, dc, ec))
            else:
                out["spurious_core"] = False
        out.update(core_dv=float(dc), core_dv_err=float(ec),
                   core_sigma=float(sc / OIII_A * C), core_amp=float(ac),
                   wing_dv=float(dw), wing_dv_err=float(ew),
                   wing_sigma=float(sw / OIII_A * C), wing_amp=float(aw),
                   two_chi2=float(np.sum(((y - two(x, *p2)) / sg) ** 2) / (len(x) - 6)))
    except Exception:
        pass
    return out




FITS_DIR = os.path.join(HERE, "pipeline_output", "chosen_visit", "fits")


def plot(index, out_dir=None):
    """Render the [O III] fit beside the object's other diagnostic figures.

    Written into chosen_visit/fits/<folder>/ so it sits with the iter*.png the
    object is judged from, rather than somewhere separate -- the redshift is
    part of the same decision as the fit, and on a low-S/N object it can be the
    larger of the two uncertainties.  The panel is drawn in the rest frame at
    the CATALOGUE redshift, so an offset is visible as the lines sitting away
    from their dotted rest wavelengths rather than having to be read off a
    number.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = measure(index)
    if r is None or "core_dv" not in r:
        return None
    with fits.open(os.path.join(SDSS_LITE, r["fn"])) as h:
        d = h[1].data
        lam = 10.0 ** np.asarray(d["loglam"], float)
        fl = np.asarray(d["flux"], float)
        iv = np.asarray(d["ivar"], float)
    z = r["z_cat"]
    rest = lam / (1.0 + z)
    fitm = (rest > 4940) & (rest < 5080) & (iv > 0)
    side = (((rest > 4890) & (rest < 4940)) | ((rest > 5090) & (rest < 5140))) & (iv > 0)
    coeff = np.polyfit(rest[side], fl[side], 1)
    x = rest[fitm]; y = fl[fitm] - np.polyval(coeff, x); sg = 1.0 / np.sqrt(iv[fitm])

    def two(xx, a1, s1, d1, a2, s2, d2):
        return _doublet(xx, a1, abs(s1), d1) + _doublet(xx, a2, abs(s2), d2)

    p1, _ = curve_fit(_doublet, x, y, p0=[y.max(), 3.0, 0.0], sigma=sg, maxfev=60000)
    p2, _ = curve_fit(two, x, y, p0=[y.max(), 1.5, 0.0, y.max() * 0.3, 8.0, -300.0],
                      sigma=sg, maxfev=200000)
    #: Take the core/wing assignment from measure(), do NOT re-derive it here.
    #: This function used to sort the two components by width independently,
    #: so when measure() learned to demote a spurious narrow spike the figure
    #: kept the old labelling and drew the blue "core" curve on the wrong
    #: component -- correct numbers in the report, wrong line in the plot.
    #: GTR caught it on index 296, having already caught the underlying bug on
    #: index 245. One source of truth.
    _pairs = [(abs(p2[1]), p2[0], p2[2]), (abs(p2[4]), p2[3], p2[5])]
    _target = r["core_sigma"] * OIII_A / C
    _ci = int(np.argmin([abs(pp[0] - _target) for pp in _pairs]))
    (sc, ac, dc) = _pairs[_ci]
    (sw, aw, dw) = _pairs[1 - _ci]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.2),
                           gridspec_kw=dict(height_ratios=[2.4, 1]))
    xs = np.linspace(x.min(), x.max(), 2000)
    ax[0].fill_between(x, -sg, sg, color="0.85", zorder=0, label=r"$\pm1\sigma$")
    ax[0].step(x, y, where="mid", color="0.35", lw=0.9, label="SDSS (continuum removed)")
    ax[0].plot(xs, _doublet(xs, *p1), color="#d95f02", lw=1.4, ls="--",
               label=r"single  dv %+.1f, $\sigma$ %.0f" % (p1[2], abs(p1[1]) / OIII_A * C))
    ax[0].plot(xs, two(xs, *p2), color="#1b9e77", lw=2.0,
               label=r"core+wing  $\chi^2$/dof %.2f" % r["two_chi2"])
    ax[0].plot(xs, _doublet(xs, ac, sc, dc), color="#1f78b4", lw=1.2,
               label=r"core  dv %+.1f$\pm$%.1f, $\sigma$ %.0f"
                     % (dc, r["core_dv_err"], sc / OIII_A * C))
    ax[0].plot(xs, _doublet(xs, aw, sw, dw), color="#e7298a", lw=1.2,
               label=r"wing  dv %+.1f, $\sigma$ %.0f" % (dw, sw / OIII_A * C))
    #: Label both frames explicitly.  The x-axis is the rest frame AT THE
    #: CATALOGUE REDSHIFT, so the laboratory rest wavelength is where each line
    #: would peak if the catalogue value were right -- that is what the dotted
    #: marks mean, and it is not obvious from the axis label alone.
    for j, (l, lab) in enumerate(((OIII_A, "5007"), (OIII_B, "4959"))):
        ax[0].axvline(l, color="k", ls=":", lw=1.1,
                      label="catalogue z (lab rest wavelength)" if j == 0 else None)
        ax[0].axvline(l * (1 + dc / C), color="#1f78b4", lw=1.0, alpha=0.8,
                      label="[O III] core z" if j == 0 else None)
        ax[0].annotate("[O III] %s" % lab, xy=(l, 1.0), xycoords=("data", "axes fraction"),
                       xytext=(2, -3), textcoords="offset points",
                       fontsize=7, va="top", ha="left", rotation=90, color="0.3")
    ax[0].annotate("catalogue z = %.6f\n[O III] core  z = %.6f   (%+.1f km/s)"
                   % (z, z + (1 + z) * dc / C, dc),
                   xy=(0.995, 0.97), xycoords="axes fraction", ha="right", va="top",
                   fontsize=8, bbox=dict(fc="white", ec="0.7", alpha=0.9, pad=2.5))
    ax[0].set_xlim(4940, 5080)
    ax[0].set_ylabel("flux - continuum")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title("%s (index %d)   plate %s   rest frame at the CATALOGUE z = %.6f"
                    % (r["name"], r["index"], r["fn"].split("/")[0], z))
    ax[1].step(x, (y - two(x, *p2)) / sg, where="mid", color="#1b9e77", lw=0.9,
               label="core+wing")
    ax[1].step(x, (y - _doublet(x, *p1)) / sg, where="mid", color="#d95f02", lw=0.8,
               alpha=0.7, label="single")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylim(-6, 6); ax[1].set_xlim(4940, 5080)
    ax[1].set_xlabel("rest wavelength at the catalogue z (A)")
    ax[1].set_ylabel(r"residual / $\sigma$")
    ax[1].legend(fontsize=7)
    ax[1].text(0.995, 0.95,
               "a line ON its dotted mark means the catalogue redshift is correct",
               transform=ax[1].transAxes, ha="right", va="top", fontsize=7)
    plt.tight_layout()
    if out_dir is None:
        hits = sorted(glob.glob(os.path.join(FITS_DIR, "%03d_*" % int(index))))
        out_dir = hits[0] if hits else FITS_DIR
    out = os.path.join(out_dir, "OIII_redshift.png")
    plt.savefig(out, dpi=135); plt.close(fig)
    return out


def unavailable_reason(index):
    """Why measure() returned None for this index, in words.

    "no usable SDSS spectrum" was the only message, and it was wrong for the
    commonest case.  Above z ~= 0.84 [O III] 5007 redshifts past the red end of
    an SDSS spectrum, so every object from index 424 to the end of the queue has
    a perfectly good spectrum that simply does not reach the line -- 51 of them.
    Reporting that as "no usable spectrum" sends anyone reading
    prepared_flags.csv looking for a retrieval failure that is not there.
    """
    try:
        sd = pd.read_csv(SDSS_TABLE)
    except Exception:
        return "sdss_spectra.csv not readable"
    row = sd[sd["index"].astype("Int64") == int(index)]
    if not len(row):
        return "no SDSS row for this index in sdss_spectra.csv"
    row = row.iloc[0]
    fn = str(row.get("fn_sdss") or "").strip()
    if not fn:
        return "no SDSS spectrum retrieved (fn_sdss empty; status %s)" % row.get("status")
    if not os.path.exists(os.path.join(SDSS_LITE, fn)):
        return "SDSS file missing from disk: %s" % fn
    #: The redshift test, done from the table rather than by reopening the file:
    #: rest_max is the reddest rest wavelength the spectrum reaches.
    try:
        rest_max = float(row.get("rest_max"))
        rest_min = float(row.get("rest_min"))
    except (TypeError, ValueError):
        rest_max = rest_min = float("nan")
    if np.isfinite(rest_max) and rest_max < 5010.0:
        return ("[O III] 5007 is outside the SDSS spectrum at z = %.4f: coverage "
                "ends at %.1f A rest (%.0f A observed). Not a failure -- the line "
                "is past SDSS's red limit above z ~ 0.84."
                % (float(row.get("z_queue", float("nan"))), rest_max,
                   rest_max * (1.0 + float(row.get("z_queue", 0.0)))))
    if np.isfinite(rest_min) and rest_min > 4940.0:
        return ("[O III] 5007 is blueward of the SDSS coverage at z = %.4f: "
                "coverage starts at %.1f A rest." % (float(row.get("z_queue", float("nan"))), rest_min))
    return ("SDSS spectrum covers [O III] but too few good pixels to fit "
            "(needs 25 in 4940-5080 A and 8 in the side bands)")


def report(r, index=None):
    if r is None:
        return "  [O III] not measured: %s" % (unavailable_reason(index)
                                               if index is not None else
                                               "no usable SDSS spectrum")
    L = ["  %s (index %d), catalogue z = %.6f, %s" % (r["name"], r["index"], r["z_cat"], r["fn"]),
         "    [O III] peak S/N %.1f over %d px" % (r["peak_snr"], r["n_px"])]
    if "single_dv" in r:
        L.append("    single   : dv %+8.1f +- %5.1f km/s   sigma %6.1f   chi2/dof %.2f"
                 % (r["single_dv"], r["single_dv_err"], r["single_sigma"], r["single_chi2"]))
    if "core_dv" in r:
        L.append("    core     : dv %+8.1f +- %5.1f km/s   sigma %6.1f   chi2/dof %.2f"
                 % (r["core_dv"], r["core_dv_err"], r["core_sigma"], r["two_chi2"]))
        L.append("    wing     : dv %+8.1f +- %5.1f km/s   sigma %6.1f   amp/core %.2f"
                 % (r["wing_dv"], r["wing_dv_err"], r["wing_sigma"],
                    r["wing_amp"] / r["core_amp"] if r["core_amp"] else np.nan))
        #: Three ways the decomposition can fail, all met in the first batch it
        #: was run on and none caught by the sigma threshold alone.
        #:
        #: (a) sigma too wide -- not a narrow-line region measurement.
        #: (b) the velocity is unconstrained even though the fit converged:
        #:     index 293 returned a core at +26.4 +/- 304.5 km/s on a peak S/N
        #:     of 11.6, which is no measurement at all.
        #: (c) the two components have swapped roles.  Sorting by sigma calls
        #:     the narrower one the core, but on index 296 that picked out a
        #:     42.6 km/s feature carrying a twelfth of the flux while the real
        #:     narrow line, at sigma 126.5, was labelled the wing -- and the
        #:     reported "core" velocity was then -400 km/s instead of -51.
        #:     A core weaker than its own wing is a failed decomposition.
        bad = []
        if r["core_sigma"] > CORE_SIGMA_MAX:
            bad.append("core sigma %.0f > %.0f km/s, not a narrow-line core"
                       % (r["core_sigma"], CORE_SIGMA_MAX))
        if r["core_dv_err"] > 50:
            bad.append("core dv uncertain to +/-%.0f km/s" % r["core_dv_err"])
        #: With the core now chosen by flux, the failure to catch is the core
        #: coming out BROADER than its own wing -- the two components have not
        #: separated into a narrow line plus an outflow, so neither is a frame.
        if r.get("widths_too_close"):
            bad.append("the two components have widths of %.0f and %.0f km/s, a ratio of "
                       "%.2f, and the narrower one carries only %.0f%% of the flux, so this is "
                       "not a core plus a wing and the labels below carry no information"
                       % (r["core_sigma"], r["wing_sigma"], r["core_wing_ratio"],
                          100.0 * r.get("core_flux_frac", float("nan"))))
        if r.get("degenerate_two"):
            bad.append("the two-Gaussian decomposition is degenerate -- one "
                       "component came out in absorption, so there is no "
                       "core/wing split to report")
        if r.get("spurious_core"):
            bad.append("the narrow component was a spike carrying %.1f%% of the flux; "
                       "values below are the dominant component instead"
                       % (100.0 * r.get("core_flux_frac", float("nan"))))
        verdict = ("CORE IS USABLE as a frame" if not bad
                   else "DO NOT USE AS A FRAME: " + "; ".join(bad))
        L.append("    -> %s ; core implies z = %.6f"
                 % (verdict, r["z_cat"] + (1 + r["z_cat"]) * r["core_dv"] / C))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("index", nargs="*", type=int)
    ap.add_argument("--all-from", type=int, default=None)
    ap.add_argument("--plot", action="store_true",
                    help="also write OIII_redshift.png into chosen_visit/fits/<folder>/")
    a = ap.parse_args()
    idx = list(a.index)
    if a.all_from is not None:
        sd = pd.read_csv(SDSS_TABLE)
        idx += sorted(int(i) for i in sd["index"].dropna().astype(int) if i >= a.all_from)
    for i in idx:
        print(report(measure(i), index=i))
        if a.plot:
            out = plot(i)
            print("    figure -> %s" % (out or "not drawn"))
        print()


if __name__ == "__main__":
    main()
