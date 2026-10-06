"""Splice a second instrument's rebinned spectrum onto the chosen visit's, where
the chosen visit does not reach.

The single-visit policy picks one instrument and one epoch, which is right when
that visit is self-sufficient.  It is not right when the chosen visit is the
only spectrum carrying the red anchors and a different instrument is the only
one carrying the blue end: LEDA 50824 (index 90) is one STIS G230L exposure
running 1451-2899 A, so it has C III] and Mg II and NO Si IV at all, and only
two thirds of the 1445-1465 A window that L1450 is read from.

This is a SPLICE, not a co-addition, and the distinction is the whole point.
The donor contributes ONLY at wavelengths the chosen visit does not cover, so
no pixel is ever averaged between instruments.  Co-adding them instead would
mix STIS G230L (R~700) with COS G160M (R~16000) straight across C IV, a factor
of 23 -- the resolution mixing that displaced the line at index 17 and drove
maskNAL mad at index 28.  Measured on LEDA 50824, the co-add moved the EW from
49.1 to 33.9 and cost 15 usable C IV pixels; the splice leaves C IV at the full
280 pixels of exactly the spectrum the pass already fitted.

The donor is cross-normalised to the chosen visit on their overlap, in
continuum-normalised flux, the same way coadd_experiment.coadd does it.  Both
products are continuum-normalised by construction, so the scale is a genuine
shape mismatch rather than a unit difference; the recorded continuum is divided
by the same scale so that flux * continuum stays in erg/s/cm2/A and the
monochromatic luminosities remain measurable.  Carrying that column is why this
does not simply call coadd(): the COMBINED-era writer drops it, and the merge
came back with nan for L1450, L1700 and L2500 -- which on this object is the
entire reason for splicing.

Gaps are written as NaN flux with the bad-pixel flag set, on a contiguous
lattice, exactly as the single-spectrum rebin files carry their detector gaps.
A hole in the wavelength array itself breaks spec_morph.morph2 and
run_ica.get_ICA, both of which pair pixel i with component i (see the note in
coadd_experiment.coadd about Mrk 486).

Two rules, both GTR's, both enforced here rather than left to care:

  * The donor supplies ONLY wavelengths the chosen visit does not cover.  The
    moment it overlaps, we are co-adding across resolutions again and the rule
    stops being checkable, so an overlapping donor pixel is an error, not a
    judgement call.
  * A spliced object should carry manmask 2 unless there is a positive reason
    otherwise.  That is not enforced -- manmask is GTR's call on the figures --
    but it is printed as a reminder when a splice is built.

Which objects are spliced, and from what, is read from spectrum_overrides.csv
(`donor_inst`, `donor_epoch`, and `donor_files` / `join` as overrides), so a
splice sits in the same row as every other hand decision about which photons
enter the fit.  fit_chosen_visit.py calls build() after it rebins the chosen
visit, so there is one entry point and the spliced product cannot go stale
behind a regenerated host.

    python splice_spectra.py --index 90
"""
import argparse
import os
import sys
import glob
import shutil
import io
import contextlib
import warnings

warnings.filterwarnings("ignore")
import csv

import numpy as np
import pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from rebinning import coadd as CO
from exposure_decisions import (read_exposures, per_file, assign_epochs, recommend,
                                _overrides)

C_WAVE = "Rest-frame Wavelength"
C_FLUX = "Coadded Flux (Arbitrary Units)"
C_ERRS = "Coadded Flux Errors"
C_MASK = "Bad Pixel Mask"
C_ZED = "Redshift"
C_CONT = "Continuum Normalisation"
STEP = 1e-4 * np.log(10.0)

MAST = os.path.join(HERE, "data_v23", "MAST_v23")
REBIN = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
WORK = os.path.join(HERE, "pipeline_output", "chosen_visit", "splice_work")

#: Columns of the rebinned products, and the SDSS log lattice they share.
def read(path):
    d = fits.getdata(path, 1)
    out = {c: np.asarray(d[c], float) for c in (C_WAVE, C_FLUX, C_ERRS, C_MASK, C_ZED)}
    out[C_CONT] = np.asarray(d[C_CONT], float)
    out["good"] = (np.isfinite(out[C_FLUX]) & np.isfinite(out[C_ERRS])
                   & (out[C_ERRS] > 0) & (out[C_MASK] == 0))
    return out


def _donor_files(index, name, z, inst, want_epochs, forced):
    """The donor visit's exposures, chosen by the same screening the host gets."""
    d = os.path.join(MAST, name, inst)
    if not os.path.isdir(d):
        sys.exit("no %s data for %s" % (inst, name))
    with contextlib.redirect_stdout(io.StringIO()):
        recs = recommend(assign_epochs(per_file(read_exposures(d, inst, z))))[0]
    if not want_epochs:
        sys.exit("donor_inst given without donor_epoch for index %d" % index)
    kept = [r for r in recs
            if r["epoch"] in want_epochs
            and (str(r["action"]).startswith("keep")
                 or any(r["file"].startswith(x) for x in forced))]
    if forced:
        back = [r["file"] for r in kept if not str(r["action"]).startswith("keep")]
        print("   donor override: reinstating %d exposure(s): %s"
              % (len(back), ", ".join(sorted(back)) or "none matched"))
    return [r["file"] for r in kept]


def build(index, keep_work=False, quiet=False):
    """Splice the donor named in spectrum_overrides.csv onto the chosen visit.

    Returns the path written, or None when this index has no donor.
    """
    ov = _overrides().get(index, {})
    donor_inst = ov.get("donor_inst") or ""
    if not donor_inst:
        return None
    queue = {int(r["order"]): r for r in csv.DictReader(
        open(os.path.join(HERE, "pipeline_output", "work_queue.csv")))}
    name = queue[index]["name_mast_key"]
    dec = pd.read_csv(os.path.join(HERE, "pipeline_output", "final",
                                   "pass_decisions.csv")).set_index("index")
    z = float(dec.loc[index, "z_adopted"])

    host_inst = ov.get("inst") or ""
    host_path = None
    for cand in ([host_inst] if host_inst else []) + ["STIS", "COS", "FOS"]:
        p = os.path.join(REBIN, "%s_%s.fits" % (name, cand))
        if cand and cand != donor_inst and os.path.exists(p):
            host_path, host_inst = p, cand
            break
    if host_path is None:
        sys.exit("no chosen-visit spectrum for %s -- run fit_chosen_visit first" % name)

    stage = os.path.join(WORK, "%03d_stage" % index)
    out_rebin = os.path.join(WORK, "%03d_rebin" % index)
    for d in (stage, out_rebin):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    files = _donor_files(index, name, z, donor_inst,
                         ov.get("donor_epochs", ()), ov.get("donor_files", ()))
    n = 0
    src = os.path.join(MAST, name, donor_inst)
    for k in files:
        root = k.replace("_c0f.fits", "").replace(".fits", "")
        for p in glob.glob(os.path.join(glob.escape(src), root + "*")):
            shutil.copy2(p, stage)
            n += 1
    print("%03d %s: donor %s epoch(s) %s -- %d exposures (%d files)"
          % (index, name, donor_inst,
             " + ".join(str(x) for x in ov.get("donor_epochs", ())), len(files), n))
    with contextlib.redirect_stdout(io.StringIO()):
        CO.rebin(name, z, donor_inst, None, data_path=stage,
                 output_dir=out_rebin, flat=True)
    donor_path = sorted(glob.glob(os.path.join(out_rebin, "*.fits")),
                        key=os.path.getmtime)[-1]

    host, donor = read(host_path), read(donor_path)
    join = ov.get("join")
    derived = float(host[C_WAVE][host["good"]].min())
    if join is None:
        join = derived
    else:
        print("   join overridden: %.2f A (the chosen visit starts at %.2f)"
              % (join, derived))
    print("   chosen visit %s begins at %.2f A; the donor supplies below that only"
          % (host_inst, derived))

    ref = np.log(host[C_WAVE][0])
    kh = np.rint((np.log(host[C_WAVE]) - ref) / STEP).astype(int)
    kd = np.rint((np.log(donor[C_WAVE]) - ref) / STEP).astype(int)
    for k, w, lab in ((kh, host[C_WAVE], "chosen visit"), (kd, donor[C_WAVE], "donor")):
        resid = np.abs((np.log(w) - ref) / STEP - k)
        if resid.max() > 0.05:
            sys.exit("%s is off the shared lattice by %.3f px" % (lab, resid.max()))

    dh = {int(k): v for k, v, g in zip(kh, host[C_FLUX], host["good"]) if g}
    dd = {int(k): v for k, v, g in zip(kd, donor[C_FLUX], donor["good"]) if g}
    shared = set(dh) & set(dd)
    r = np.array([dh[k] / dd[k] for k in shared if dd[k] != 0])
    r = r[np.isfinite(r) & (r > 0)]
    if r.size < 20:
        sys.exit("only %d usable overlap pixels -- too few to cross-normalise" % r.size)
    scale = float(np.median(r))
    print("   overlap %d px; donor scaled by %.4f" % (r.size, scale))

    take = donor["good"] & (donor[C_WAVE] < join)
    if not take.any():
        sys.exit("the donor has no usable pixels blueward of the chosen visit")
    # GTR's rule, checked rather than trusted: the donor supplies only where the
    # chosen visit has nothing.  A donor pixel landing on a host pixel would mean
    # two instruments averaged at one wavelength, which is the resolution mixing
    # this whole arrangement exists to avoid, and it would do so silently.
    clash = set(kd[take].tolist()) & set(kh[host["good"]].tolist())
    if clash:
        sys.exit("donor and chosen visit both cover %d pixel(s) -- "
                 "the donor must supply only what the chosen visit does not" % len(clash))
    print("   donor contributes %d px over %.1f-%.1f A, none shared with the chosen visit"
          % (int(take.sum()), donor[C_WAVE][take].min(), donor[C_WAVE][take].max()))

    k_all = np.concatenate([kd[take], kh[host["good"]]])
    f_all = np.concatenate([donor[C_FLUX][take] * scale, host[C_FLUX][host["good"]]])
    e_all = np.concatenate([donor[C_ERRS][take] * scale, host[C_ERRS][host["good"]]])
    c_all = np.concatenate([donor[C_CONT][take] / scale, host[C_CONT][host["good"]]])
    o = np.argsort(k_all)
    k_all, f_all, e_all, c_all = k_all[o], f_all[o], e_all[o], c_all[o]

    full = np.arange(k_all.min(), k_all.max() + 1)
    wave = np.exp(ref + full * STEP)
    at = np.searchsorted(full, k_all)

    def lift(src_arr):
        out = np.full(full.size, np.nan)
        out[at] = src_arr
        return out

    flux, errs, cont = lift(f_all), lift(e_all), lift(c_all)
    mask = np.where(np.isfinite(flux), 0.0, 1.0)
    zz = float(host[C_ZED][0])

    cols = fits.ColDefs([
        fits.Column(name=C_WAVE, format="D", array=wave),
        fits.Column(name=C_FLUX, format="D", array=flux),
        fits.Column(name=C_ERRS, format="D", array=errs),
        fits.Column(name=C_MASK, format="D", array=mask),
        fits.Column(name=C_ZED, format="D", array=np.full(wave.size, zz)),
        fits.Column(name=C_CONT, format="D", array=cont)])
    out_path = os.path.join(REBIN, "%s_SPLICE.fits" % name)
    fits.HDUList([fits.PrimaryHDU(),
                  fits.BinTableHDU.from_columns(cols)]).writeto(out_path, overwrite=True)
    g = mask == 0
    print("   -> %s   %.1f-%.1f A, z = %s" % (os.path.basename(out_path),
                                              wave.min(), wave.max(), zz))
    if not quiet:
        for lab, lo, hi in (("SiIV", 1380, 1420), ("L1450", 1445, 1465),
                            ("CIV", 1500, 1600), ("L1700", 1700, 1705),
                            ("CIII]", 1860, 1960), ("L2500", 2490, 2510),
                            ("MgII", 2740, 2860)):
            sel = (wave >= lo) & (wave <= hi) & g
            src_lab = ("donor" if hi <= join else
                       ("chosen visit" if lo >= join else "both sides of the join"))
            print("      %-7s %4d usable px   (%s)" % (lab, int(sel.sum()), src_lab))
        # A hole can open at the join when the donor's own coverage stops short
        # of where the chosen visit begins: on index 90 COS's G160M segment gap
        # sits at 1447.0-1453.7 A, inside the L1450 window.
        hole = (wave >= donor[C_WAVE][take].max()) & (wave < derived) & (~g)
        if hole.any():
            print("      NOTE: %d px hole at the join, %.1f-%.1f A"
                  % (int(hole.sum()), wave[hole].min(), wave[hole].max()))
        print("      manmask: a spliced object should be 2 unless there is a "
              "positive reason otherwise (GTR, 2026-10-06).")
    if not keep_work:
        shutil.rmtree(WORK, ignore_errors=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, nargs="+", default=None)
    ap.add_argument("--keep-work", action="store_true",
                    help="leave the staged and rebinned donor files in place")
    a = ap.parse_args()
    spliced = sorted(i for i, v in _overrides().items() if v.get("donor_inst"))
    for i in (a.index if a.index is not None else spliced):
        if i not in spliced:
            print("%03d: no donor named in spectrum_overrides.csv" % i)
            continue
        build(i, keep_work=a.keep_work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
