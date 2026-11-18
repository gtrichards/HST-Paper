"""Regression tests on real spectra, so a change to the analysis code is visible.

GTR, 2026-11-13: "I'm losing confidence in the repeatability of this work.  I
think that we need to write some tests in order to have confidence in the final
product. ... create (or re-use) some example spectra with a variety of different
complications.  Then any change in the analysis code gets propagated to those
spectra and we make sure that nothing changes."

Each fixture is a real object's exposures, trimmed to what exercises one
complication, plus a frozen record of what the pipeline produced from it.  The
test rebuilds the product from the raw exposures and refits it, so it covers the
WHOLE path -- read_spec_data, coadd, the morph, maskNAL and the ICA fit -- not
just the arithmetic at the end.  That matters: every defect found on 2026-11-08
to 13 was in the reduction, and all three would have been caught here.

WHAT IS FROZEN, and why each entry is there:

    product-level
      sha256 of each column        any change at all, as a tripwire
      usable pixels per window     coverage and masking decisions
      min error / median error     the near-zero-error pixels of index 233
      scatter/error per window     catches an error array that moves the wrong
                                   way (index 70, 306); flux/error cannot,
                                   because both halves move
      flux stats per window        median, p05, p95 and ROUGHNESS of the flux
                                   array, to a 0.1 per cent relative tolerance.
                                   Until these existed the flux was watched
                                   only by sha_flux, a tripwire that says
                                   "changed" and nothing about how much or
                                   where -- and scatter/error, being a ratio,
                                   cannot see flux and error moving together.
      continuum stats per window   the other half of every EW and the whole of
                                   every luminosity: without it an EW change
                                   cannot be attributed to flux or continuum
      C IV data centroid           the half-flux blueshift of the DATA, by the
                                   measurement's own prescription applied to
                                   the co-added spectrum instead of to the ICA
                                   reconstruction.  The centroid is what the
                                   blueshift IS, and a fit that disagrees with
                                   its own data (index 70) shows up here
    fit-level
      blueshift                    EXACT: quantised at 69.03 km/s, so any real
                                   change is at least one quantum
      EW, log L1450, log L1700     to a tolerance -- see below
      f2500, veto pixels, usable C IV pixels, residual medians

TOLERANCES.  GTR: "We're not trying to prevent *any* changes, just changes that
matter."  EW to 0.1 A (under half a per cent of a typical 20-300 A measurement)
and luminosity to 0.01 dex (2.3 per cent).  For calibration, the errors this
project actually found were 0.502 dex (the morph coefficient), 0.19 dex (the
missing 1+z) and 5.9 dex (an amplitude collapse).  The blueshift is compared
exactly because quantisation means it cannot move by less than 69 km/s.

RE-BLESSING IS DELIBERATE.  --update rewrites the golden files and prints the
diff, which belongs in the commit message that changes them.  A test that is
easy to silence is worse than no test.

    python run_spectra_tests.py                 # check every fixture
    python run_spectra_tests.py --only fos_     # a subset, by name prefix
    python run_spectra_tests.py --update        # re-freeze, printing the diff
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PAPER)
sys.path.insert(0, os.path.join(PAPER, "finalrun"))

FIX = os.path.join(HERE, "fixtures")
GOLD = os.path.join(HERE, "golden")

CIV = 1549.48
#: (name, lo, hi) windows the frozen record reports on
WINDOWS = (("L1450", 1445.0, 1465.0), ("SiIV", 1360.0, 1440.0),
           ("CIV", 1500.0, 1600.0), ("linefree", 1620.0, 1700.0),
           ("CIII]", 1850.0, 1970.0), ("MgII", 2700.0, 2900.0))
TOL_EW = 0.1
TOL_LOGL = 0.01
#: relative tolerance on the flux and continuum statistics.  0.1 per cent: on a
#: 40 A line that is 0.04 A of EW, comfortably inside TOL_EW, so a flux change
#: large enough to trip this is a change that could matter to a measurement.
TOL_REL = 1e-3


def sha(a):
    b = np.ascontiguousarray(np.asarray(a, float))
    return hashlib.sha256(b.tobytes()).hexdigest()[:16]


def scatter_over_error(w, f, e, g, lo, hi):
    s = g & (w >= lo) & (w <= hi)
    if s.sum() < 40:
        return None
    x = f[s]
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    er = float(np.nanmedian(e[s]))
    if not (er > 0):
        return None
    return round(float((np.nanstd(d2) / np.sqrt(6.0)) / er), 3)


def array_stats(w, a, g, lo, hi):
    """Descriptive statistics of ONE array over one window.

    Why the roughness is here.  scatter_over_error is a RATIO, and a ratio is
    blind to its two halves moving together: double the flux noise and double
    the errors and it does not budge.  It was built to catch an error array
    moving the wrong way and it does that, but it cannot see the flux alone.
    This freezes the numerator by itself -- the rms second difference over
    sqrt(6), the pixel-to-pixel roughness of the array with no error in the
    denominator -- so flux and error are each pinned down independently.
    """
    s = g & (w >= lo) & (w <= hi)
    if s.sum() < 10:
        return None
    x = np.asarray(a, float)[s]
    x = x[np.isfinite(x)]
    if x.size < 10:
        return None
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    sig = lambda v: float("%.6g" % v)
    return dict(median=sig(np.median(x)), p05=sig(np.percentile(x, 5)),
                p95=sig(np.percentile(x, 95)),
                roughness=sig(np.std(d2) / np.sqrt(6.0)))


def data_centroid(w, f, g):
    """The DATA's own C IV half-flux blueshift, in km/s.

    A DIAGNOSTIC, not the published measurement.  It applies the measurement's
    own shape -- a straight continuum through the 1445-1465 and 1700-1705 A
    anchors, the clamped cumulative EW integrand, the half-flux point
    (ica/run_ica.py:140-153) -- to the CO-ADDED DATA instead of to the ICA
    reconstruction, with the window indexing done straightforwardly.

    It is here because the flux array can move in a way that no summary
    statistic catches but that the measurement depends on entirely: the
    centroid is what the blueshift IS.  It also makes a specific disagreement
    visible as a number.  On index 70 (Mrk 1502, I Zw 1) the data's centroid
    sits near +1716 km/s on both the old and the new product, agreeing with
    Amogh, Sulentic and Trevor, while every ICA fit on the corrected product
    returns +128 to +611 -- a gap between what the data say and what the fit
    returns that nothing frozen in this record would otherwise show.

    Returns None when the red anchor is empty, which it is for 104 of the 366
    adopted objects, so a None here is coverage and not a failure.
    """
    c1 = g & (w >= 1445) & (w <= 1465)
    c2 = g & (w >= 1700) & (w <= 1705)
    ew = g & (w >= 1500) & (w <= 1600)
    if c1.sum() < 3 or c2.sum() < 3 or ew.sum() < 40:
        return None
    from ica.run_ica import civ_continuum
    cont = civ_continuum(w, f)          # the measurement's power law (2026-11-17; a line before)
    ww, ff = w[ew], f[ew]
    cc = cont[ew]
    if not np.all(np.isfinite(cc)) or np.any(cc == 0):
        return None
    dw = np.diff(ww, append=ww[-1] + (ww[-1] - ww[-2]))
    integ = np.clip((ff - cc) / cc, 0, None) * dw
    cum = np.concatenate(([0.0], np.cumsum(integ)))
    total = cum[-1]
    if not (total > 0):
        return None
    # the pixel whose span contains the half-flux point (cum[j] is the integral
    # through pixel j-1), as run_ica.civ_measure does since 2026-11-17
    k = int(np.searchsorted(cum[1:], total / 2.0, side="left"))
    k = min(k, len(ww) - 1)
    return round(float((1549.48 - ww[k]) / 1549.48 * 299792.458), 1)


def measure(fixture, dump=None):
    """Rebuild the fixture's product and fit it; return the frozen record."""
    from astropy.table import Table
    meta = json.load(open(os.path.join(FIX, fixture, "fixture.json")))
    inst, z = meta["inst"], meta["z"]
    src = os.path.join(FIX, fixture, "exposures")
    work = tempfile.mkdtemp(prefix="spectest_")
    try:
        stage = os.path.join(work, "stage")
        reb = os.path.join(work, "rebin")
        os.makedirs(stage)
        os.makedirs(reb)
        for p in sorted(glob.glob(os.path.join(src, "*"))):
            shutil.copy2(p, stage)

        if inst == "IUE":
            sys.path.insert(0, os.path.join(PAPER, "finalrun", "iue"))
            from rebinning import coadd as CO
            name = meta["obj"]
            CO.rebin(name, z, "IUE", None, data_path=stage, output_dir=reb, flat=True)
        else:
            from rebinning import coadd as CO
            name = meta["obj"]
            if z is None:
                return dict(skipped="fixture has no redshift recorded")
            CO.rebin(name, z, inst, None, data_path=stage, output_dir=reb, flat=True)

        prods = sorted(glob.glob(os.path.join(reb, "*.fits")))
        if not prods:
            return dict(error="no product was written")
        t = Table.read(prods[0])
        w = np.asarray(t["Rest-frame Wavelength"], float)
        f = np.asarray(t["Coadded Flux (Arbitrary Units)"], float)
        e = np.asarray(t["Coadded Flux Errors"], float)
        m = np.asarray(t["Bad Pixel Mask"], float)
        cn = (np.asarray(t["Continuum Normalisation"], float)
              if "Continuum Normalisation" in t.colnames else np.full(len(w), np.nan))
        g = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)

        if dump:
            os.makedirs(dump, exist_ok=True)
            np.savez_compressed(os.path.join(dump, "%s.npz" % fixture),
                                wave=w, flux=f, err=e, mask=m, cont=cn)
        rec = dict(
            product=dict(
                n_rows=int(len(t)), n_usable=int(g.sum()),
                rest_lo=round(float(w[g].min()), 2) if g.any() else None,
                rest_hi=round(float(w[g].max()), 2) if g.any() else None,
                sha_wave=sha(w), sha_flux=sha(f), sha_err=sha(e),
                sha_mask=sha(m), sha_cont=sha(cn),
                min_over_median_error=(round(float(np.nanmin(e[g]) / np.nanmedian(e[g])), 6)
                                       if g.any() else None),
                usable=dict((n, int((g & (w >= lo) & (w <= hi)).sum()))
                            for n, lo, hi in WINDOWS),
                scatter_over_error=dict((n, scatter_over_error(w, f, e, g, lo, hi))
                                        for n, lo, hi in WINDOWS),
                # the FLUX array itself, which until now only sha_flux watched --
                # a tripwire that says "changed" and nothing about how much or where
                flux=dict((n, array_stats(w, f, g, lo, hi)) for n, lo, hi in WINDOWS),
                # the continuum is the other half of every EW and the whole of
                # every luminosity, so an EW change cannot be attributed without it
                continuum=dict((n, array_stats(w, cn, g, lo, hi))
                               for n, lo, hi in WINDOWS),
                civ_data_centroid=data_centroid(w, f, g)))

        # the fit, with no hand masks: the fixture tests the code, not a judgement
        from ica.manual_fix import ICAManualFixProcessor
        proc = ICAManualFixProcessor(rebin_path=reb, master_mode=True)
        stem = os.path.basename(prods[0])[:-5]
        try:
            r = proc.fit_with_overrides(stem, mask_ranges=[], mask_pixels=[],
                                        unmask_ranges=[], unmask_pixels=[], comps_use=None)
        except SystemExit as ex:
            rec["fit"] = dict(refused=str(ex)[:160])
            return rec
        except Exception as ex:
            rec["fit"] = dict(error="%s: %s" % (type(ex).__name__, str(ex)[:120]))
            return rec
        wi, fi = r["wave_ica"], r["flux_ica"]
        c1 = (wi >= 1445) & (wi <= 1465)
        c2 = (wi >= 1700) & (wi <= 1705)
        veto = None
        if c1.sum() > 3 and c2.sum() > 3:
            from ica.run_ica import civ_continuum
            win = (wi >= 1500) & (wi <= 1600)
            veto = int(((fi - civ_continuum(wi, fi)) < 0)[win].sum())
        # the monochromatic luminosities, by the same route iterate_fit uses:
        # the reconstruction's median in the window times the recorded continuum
        def _logl(lo, hi):
            kk = (wi >= lo) & (wi <= hi)
            jj = g & (w >= lo) & (w <= hi)
            if kk.sum() < 3 or jj.sum() < 3:
                return None
            f0 = float(np.nanmedian(fi[kk]))
            sc = float(np.nanmedian(cn[jj]))
            if not (np.isfinite(f0) and np.isfinite(sc)) or f0 * sc <= 0:
                return None
            from astropy.cosmology import FlatLambdaCDM
            dl = FlatLambdaCDM(H0=70, Om0=0.3).luminosity_distance(z).to("cm").value
            lam = 0.5 * (lo + hi)
            return round(float(np.log10(4 * np.pi * dl ** 2 * f0 * sc * lam * (1 + z))), 4)
        rec["fit"] = dict(blueshift=round(float(r["civ_blue"]), 3),
                          ew=round(float(r["civ_ew"]), 3),
                          logL1450=_logl(1445.0, 1465.0),
                          logL1700=_logl(1700.0, 1705.0),
                          f2500=round(float(r["f2500"]), 5),
                          veto_px=veto,
                          civ_usable=int((g & (w >= 1500) & (w <= 1600)).sum()))
        return rec
    finally:
        shutil.rmtree(work, ignore_errors=True)


def compare(name, old, new):
    """Differences that matter.  Returns a list of strings."""
    out = []
    if old is None:
        return ["NO GOLDEN RECORD -- run with --update to create it"]
    for section in ("product", "fit"):
        o, n = old.get(section) or {}, new.get(section) or {}
        for k in sorted(set(o) | set(n)):
            a, b = o.get(k), n.get(k)
            if isinstance(a, dict) or isinstance(b, dict):
                a = a or {}
                b = b or {}
                for kk in sorted(set(a) | set(b)):
                    x, y = a.get(kk), b.get(kk)
                    # the flux and continuum blocks are one level deeper --
                    # window -> statistic -- and are compared on a RELATIVE
                    # tolerance so the report says how far something moved and
                    # not merely that it did
                    if k in ("flux", "continuum"):
                        x = x or {}
                        y = y or {}
                        for st in sorted(set(x) | set(y)):
                            u, v = x.get(st), y.get(st)
                            if _num(u) and _num(v):
                                sc = max(abs(u), abs(v))
                                if sc > 0 and abs(u - v) / sc > TOL_REL:
                                    out.append("product.%s.%s.%s  %.6g -> %.6g"
                                               "  (%+.2f%%, tolerance %.1f%%)"
                                               % (k, kk, st, u, v,
                                                  100.0 * (v - u) / sc,
                                                  100.0 * TOL_REL))
                            elif u != v:
                                out.append("product.%s.%s.%s  %s -> %s"
                                           % (k, kk, st, u, v))
                        continue
                    if x != y:
                        out.append("%s.%s.%s  %s -> %s" % (section, k, kk, x, y))
                continue
            if k == "civ_data_centroid" and _num(a) and _num(b):
                # quantised on the same 69.03 km/s lattice as the measurement,
                # so compared exactly, like the blueshift
                if a != b:
                    out.append("product.civ_data_centroid  %.1f -> %.1f km/s"
                               "  (%+.1f, one lattice pixel is 69.03)"
                               % (a, b, b - a))
                continue
            if k == "ew" and _num(a) and _num(b):
                if abs(a - b) > TOL_EW:
                    out.append("fit.ew  %.3f -> %.3f  (tolerance %.2f A)" % (a, b, TOL_EW))
                continue
            if k.startswith("logL") and _num(a) and _num(b):
                if abs(a - b) > TOL_LOGL:
                    out.append("fit.%s  %.3f -> %.3f  (tolerance %.3f dex)" % (k, a, b, TOL_LOGL))
                continue
            if a != b:
                out.append("%s.%s  %s -> %s" % (section, k, a, b))
    return out


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="name prefix to restrict to")
    ap.add_argument("--update", action="store_true", help="re-freeze the golden records")
    ap.add_argument("--dir", default=None,
                    help="fixture directory to use instead of fixtures/ (e.g. cartoons)")
    ap.add_argument("--json", default=None, help="also write every result to this file")
    ap.add_argument("--dump", default=None, metavar="DIR",
                    help="also save each rebuilt product's arrays as <DIR>/<name>.npz, "
                         "so two versions of the code can be compared as SPECTRA and not "
                         "only as numbers")
    a = ap.parse_args()
    global FIX
    if a.dir:
        FIX = a.dir if os.path.isabs(a.dir) else os.path.join(HERE, a.dir)
    names = sorted(os.path.basename(d) for d in glob.glob(os.path.join(FIX, "*"))
                   if os.path.isdir(d))
    if a.only:
        names = [n for n in names if n.startswith(a.only)]
    if not names:
        sys.exit("no fixtures matched")
    os.makedirs(GOLD, exist_ok=True)

    failed, changed, skipped = [], [], []
    results = {}
    for n in names:
        gp = os.path.join(GOLD, "%s.json" % n)
        old = json.load(open(gp)) if os.path.exists(gp) else None
        try:
            new = measure(n, dump=a.dump)
        except Exception as ex:
            print("  %-30s ERROR  %s: %s" % (n, type(ex).__name__, str(ex)[:90]))
            failed.append(n)
            continue
        if new.get("skipped"):
            print("  %-30s skipped (%s)" % (n, new["skipped"]))
            skipped.append(n)
            continue
        results[n] = new
        diffs = compare(n, old, new)
        if a.update:
            json.dump(new, open(gp, "w"), indent=1)
            print("  %-30s %s" % (n, "frozen" if old is None
                                  else ("UPDATED, %d change(s)" % len(diffs) if diffs else "unchanged")))
            for d in diffs:
                print("        %s" % d)
        elif diffs:
            print("  %-30s %d CHANGE(S)" % (n, len(diffs)))
            for d in diffs:
                print("        %s" % d)
            changed.append(n)
        else:
            print("  %-30s ok" % n)
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
        print("  wrote %s" % a.json)
    print()
    if a.update:
        print("golden records written for %d fixture(s)" % (len(names) - len(failed) - len(skipped)))
        return 0
    print("%d fixture(s): %d ok, %d changed, %d errored, %d skipped"
          % (len(names), len(names) - len(changed) - len(failed) - len(skipped),
             len(changed), len(failed), len(skipped)))
    return 1 if (changed or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
