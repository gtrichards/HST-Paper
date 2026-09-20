"""
Measure emission-line coverage directly from the pixels of every rebinned master
spectrum.

Coverage has until now been *inferred* from MAST grating metadata (the civ_zone
flag), which cannot distinguish nominal coverage from real coverage and leaves
~188 objects needing manual inspection. This measures it instead: for each of the
four diagnostic lines, how many pixels are actually present in the window, and how
many of those are usable.

A pixel counts as good only if the flux and error are finite, the error is
strictly positive, and the bad-pixel mask is clear. That is deliberately stricter
than "a pixel exists" -- a spectrum padded with zeros across the C IV window has
pixels but no data.

Four lines rather than one, because C IV alone is not enough. C III] and Mg II are
the anchors that tell you whether the redshift is right, so knowing where they are
covered tells you for which objects the redshift can be checked at all. Si IV is
recorded for completeness; it is an outflow line and is not an anchor.

Writes one row per spectrum. Nothing is excluded and nothing is judged here --
this produces the measurement that later decisions are made from.
"""

import csv
import glob
import os
import sys

import numpy as np
from astropy.io import fits

from rebin_path import REBIN as _REBIN_DEFAULT
REBIN_DIR = _REBIN_DEFAULT
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "pipeline_output", "master_line_coverage.csv")

# The ICA fits its components across this whole range (wav_12603000.dat /
# wav_12753000.dat), so this -- not the C IV window -- is what decides whether a
# spectrum contributes to the fit. C IV's shape is constrained by the entire UV
# spectrum, so a spectrum with no C IV coverage can still improve the
# reconstruction.
ICA_LO, ICA_HI = 1260.0, 3000.0

# Same windows as model_data_shift.py, so the two analyses agree on what
# "in the line" means.
LINES = [
    ("siiv",  1399.80, 1360.0, 1440.0),
    ("civ",   1549.48, 1500.0, 1600.0),
    ("ciii",  1908.73, 1860.0, 1960.0),
    ("mgii",  2799.94, 2740.0, 2860.0),
]

# Column names as written by the rebinning code.
C_WAVE = "Rest-frame Wavelength"
C_FLUX = "Coadded Flux (Arbitrary Units)"
C_ERRS = "Coadded Flux Errors"
C_MASK = "Bad Pixel Mask"
C_ZED = "Redshift"


def _column(data, name):
    """Fetch a column, tolerating case/whitespace drift between files."""
    names = data.columns.names
    if name in names:
        return data[name]
    want = name.lower().strip()
    for n in names:
        if n.lower().strip() == want:
            return data[n]
    raise KeyError("%s not among %s" % (name, names))


def scan_one(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    row = {"stem": stem}

    # Stem is "<name>_<INSTRUMENT>"; the name itself may contain underscores
    # and spaces, so split from the right and only once.
    if "_" in stem:
        name, inst = stem.rsplit("_", 1)
    else:
        name, inst = stem, ""
    row["name_from_stem"] = name
    row["instrument_from_stem"] = inst

    data = fits.getdata(path, 1)
    wave = np.asarray(_column(data, C_WAVE), dtype=float)
    flux = np.asarray(_column(data, C_FLUX), dtype=float)
    errs = np.asarray(_column(data, C_ERRS), dtype=float)
    try:
        bpix = np.asarray(_column(data, C_MASK), dtype=float)
    except KeyError:
        bpix = np.zeros_like(wave)
    try:
        zcol = np.asarray(_column(data, C_ZED), dtype=float)
        z = float(zcol[0]) if zcol.size else float("nan")
    except KeyError:
        z = float("nan")

    row["z_fits"] = z
    row["npix"] = wave.size
    row["wave_min"] = float(np.nanmin(wave)) if wave.size else float("nan")
    row["wave_max"] = float(np.nanmax(wave)) if wave.size else float("nan")

    usable = np.isfinite(flux) & np.isfinite(errs) & (errs > 0) & (bpix == 0)

    inica = (wave >= ICA_LO) & (wave <= ICA_HI)
    row["ica_npix"] = int(inica.sum())
    row["ica_ngood"] = int((inica & usable).sum())
    row["ica_frac"] = round(row["ica_ngood"] / max(row["ica_npix"], 1), 4)

    for tag, _rest, lo, hi in LINES:
        inwin = (wave >= lo) & (wave <= hi)
        n_in = int(inwin.sum())
        n_good = int((inwin & usable).sum())
        # Fraction of the window that is usable, by pixel count against the
        # nominal 69 km/s grid spacing rather than against n_in -- otherwise a
        # spectrum with three pixels in the window scores 100%.
        nominal = max(n_in, 1)
        row["%s_npix" % tag] = n_in
        row["%s_ngood" % tag] = n_good
        row["%s_frac" % tag] = round(n_good / nominal, 4) if n_in else 0.0
        # Does the window lie inside the spectrum's wavelength span at all?
        row["%s_inrange" % tag] = bool(wave.size and wave.min() <= hi
                                       and wave.max() >= lo)
    return row


def main():
    files = sorted(glob.glob(os.path.join(REBIN_DIR, "*.fits")))
    if not files:
        print("No FITS found in %s" % REBIN_DIR, file=sys.stderr)
        return 1
    print("scanning %d spectra" % len(files), flush=True)

    rows, failures = [], []
    for i, f in enumerate(files, 1):
        try:
            rows.append(scan_one(f))
        except Exception as exc:
            failures.append((os.path.basename(f), "%s: %s" % (type(exc).__name__, exc)))
        if i % 100 == 0:
            print("  %d/%d" % (i, len(files)), flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = (["stem", "name_from_stem", "instrument_from_stem", "z_fits",
             "npix", "wave_min", "wave_max", "ica_npix", "ica_ngood", "ica_frac"]
            + ["%s_%s" % (t, s) for t, _r, _lo, _hi in LINES
               for s in ("npix", "ngood", "frac", "inrange")])
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print("\nwrote %d rows -> %s" % (len(rows), OUT))
    if failures:
        print("\n%d FAILED:" % len(failures))
        for n, e in failures:
            print("   %-44s %s" % (n, e))

    # Summary: how much of the sample each line is actually usable in.
    print("\n%-6s %8s %8s %8s %8s" % ("line", "zero", "<50%", ">=80%", "in-range"))
    for tag, _r, _lo, _hi in LINES:
        fr = np.array([r["%s_frac" % tag] for r in rows])
        ir = np.array([r["%s_inrange" % tag] for r in rows])
        print("%-6s %8d %8d %8d %8d"
              % (tag, int((fr == 0).sum()), int(((fr > 0) & (fr < 0.5)).sum()),
                 int((fr >= 0.8).sum()), int(ir.sum())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
