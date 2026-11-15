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
      scatter/error per window     the only thing that catches an error array
                                   that moves the wrong way (index 70, 306);
                                   flux/error cannot, because both halves move
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
                                        for n, lo, hi in WINDOWS)))

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
            sl, ic = np.polyfit(np.concatenate((wi[c1], wi[c2])),
                                np.concatenate((fi[c1], fi[c2])), 1)
            win = (wi >= 1500) & (wi <= 1600)
            veto = int(((fi - (wi * sl + ic)) < 0)[win].sum())
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
                    if x != y:
                        out.append("%s.%s.%s  %s -> %s" % (section, k, kk, x, y))
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
