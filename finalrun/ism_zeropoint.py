"""Measure each exposure's wavelength zero point against Galactic interstellar lines.

The problem this answers: NGC 7469's FOS spectrum puts C IV, C III] and Mg II
400-600 km/s redward of rest while COS and STIS, de-redshifted identically, sit
within 35 km/s.  Comparing an AGN's broad-line peaks between instruments cannot
say whether that is the instrument or the source, because the lines are broad,
structured and variable.

Galactic interstellar absorption can.  It sits at zero velocity in the observed
frame whatever the AGN does, so the position of Si II 1260, C II 1334, Si IV
1393/1402, Al II 1670 and Mg II 2796/2803 is a direct measure of the wavelength
zero point of the exposure that recorded it.

Because the low-dispersion gratings do not resolve these lines -- FOS G160L runs
about 1200 km/s per resolution element -- the shift is measured by
cross-correlating the continuum-normalised spectrum against a template of the
lines convolved to that grating's resolution, rather than by centroiding
individual features.  Blending is then part of the model instead of a bias, and
the width of the correlation peak gives the uncertainty.

    python ism_zeropoint.py --objects "NGC-7469"
    python ism_zeropoint.py --all --out pipeline_output/final/ism_zeropoint.csv
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from exposure_decisions import read_exposures, MAST

C = 299792.458

#: Galactic interstellar lines, with a rough relative strength.  These are the
#: transitions that are strong in essentially every sightline out of the Galaxy;
#: the strengths only have to be good enough to weight the template sensibly.
ISM = [
    (1190.42, 0.6), (1193.29, 0.7), (1199.55, 0.7),   # Si II, N I triplet
    (1260.42, 1.0),                                    # Si II
    (1302.17, 0.9), (1304.37, 0.6),                    # O I, Si II
    (1334.53, 1.0), (1335.71, 0.4),                    # C II, C II*
    (1393.76, 0.8), (1402.77, 0.5),                    # Si IV
    (1526.71, 0.8),                                    # Si II
    (1548.20, 0.6), (1550.77, 0.4),                    # C IV
    (1608.45, 0.5),                                    # Fe II
    (1670.79, 0.9),                                    # Al II
    (2344.21, 0.6), (2374.46, 0.4), (2382.76, 0.8),    # Fe II
    (2586.65, 0.6), (2600.17, 0.9),                    # Fe II
    (2796.35, 1.0), (2803.53, 0.9),                    # Mg II
    (2852.96, 0.7),                                    # Mg I
]

#: Rough resolving power per grating, used only to set the template's line width.
R_GRATING = {
    "G130M": 16000, "G160M": 16000, "G140L": 2000, "G230L": 700,
    "E140M": 45000, "E230M": 30000, "E140H": 114000, "E230H": 114000,
    "G140M": 10000, "G230M": 10000, "G230LB": 700, "G430L": 500, "G750L": 500,
    "G430M": 6000, "G750M": 5000,
    "H13": 1300, "H19": 1300, "H27": 1300, "H40": 1300, "H57": 1300,
    "L15": 250, "L65": 250,
    "G130H": 1300, "G190H": 1300, "G270H": 1300, "G400H": 1300, "G570H": 1300,
    "G160L": 250, "G650L": 250, "PRI": 100, "PRISM": 100,
}

#: AGN emission lines to keep out of the correlation, since a broad line drifting
#: under an interstellar one would pull the shift toward the source's velocity.
AGN_LINES = (1215.67, 1240.0, 1305.0, 1335.0, 1400.0, 1549.48, 1640.4, 1663.0,
             1857.4, 1892.0, 1908.73, 2326.0, 2423.0, 2799.12)
AGN_HALF = 4000.0   # km/s excluded either side of each AGN line


#: Width of the running-median continuum window, in Angstrom.  It must be set in
#: wavelength and not in pixels: the dispersions here span three orders of
#: magnitude, so a fixed pixel count that is sensible for FOS (about 1 A per
#: diode) spans a tenth of an Angstrom on COS and divides the lines out along
#: with the continuum.
CONT_WIN_AA = 25.0


def normalise(w, f, win_aa=CONT_WIN_AA):
    """Divide out a running-median continuum, so only the lines are left."""
    f = np.asarray(f, float)
    n = len(f)
    disp = np.nanmedian(np.abs(np.diff(np.asarray(w, float))))
    if not np.isfinite(disp) or disp <= 0:
        return None
    win = int(round(win_aa / disp))
    win = max(11, win + (win + 1) % 2)          # odd, and never degenerate
    if n < win + 10:
        return None
    pad = win // 2
    g = np.concatenate([f[:pad][::-1], f, f[-pad:][::-1]])
    med = np.array([np.nanmedian(g[i:i + win]) for i in range(n)])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = f / med
    out[~np.isfinite(out)] = np.nan
    return out


def template(w, R):
    """Absorption template: unit continuum minus Gaussians at the ISM lines."""
    t = np.ones_like(w, dtype=float)
    for lam, strength in ISM:
        if lam < w[0] or lam > w[-1]:
            continue
        sigma = lam / R / 2.355 * 2.0   # a resolution element, as a Gaussian sigma
        t -= strength * np.exp(-0.5 * ((w - lam) / sigma) ** 2)
    return t


def measure(w, f, z, R, max_kms=1500.0):
    """Velocity shift of the observed ISM lines from their laboratory positions."""
    order = np.argsort(w)
    w, f = np.asarray(w, float)[order], np.asarray(f, float)[order]
    good = np.isfinite(w) & np.isfinite(f) & (f != 0)
    if good.sum() < 400:
        return None
    w, f = w[good], f[good]
    norm = normalise(w, f)
    if norm is None:
        return None
    use = np.isfinite(norm)
    for lam in AGN_LINES:                      # observed position of each AGN line
        obs = lam * (1 + z)
        use &= np.abs(C * (w - obs) / obs) > AGN_HALF
    covered = [(lam, s) for lam, s in ISM if w[0] < lam < w[-1]]
    if len(covered) < 2:
        return None
    near = np.zeros_like(w, dtype=bool)        # only look near the lines themselves
    for lam, _ in covered:
        near |= np.abs(C * (w - lam) / lam) < 3 * max_kms
    use &= near
    # FOS spectra are 512 diodes, so a threshold sensible for a 16384-pixel COS
    # exposure would refuse the instrument this test was built to examine.  Scale
    # it to the spectrum, with a floor that still needs several resolution
    # elements of line.
    if use.sum() < max(40, 0.05 * len(w)):
        return None
    a = np.where(use, 1.0 - norm, 0.0)         # absorption as a positive signal
    a[~np.isfinite(a)] = 0.0
    if not np.any(a):
        return None
    dvs = np.arange(-max_kms, max_kms + 1, 10.0)
    cc = np.empty_like(dvs)
    for i, dv in enumerate(dvs):
        t = 1.0 - template(w / (1 + dv / C), R)
        t = np.where(use, t, 0.0)
        cc[i] = float(np.dot(a, t))
    k = int(np.argmax(cc))
    if k in (0, len(dvs) - 1):
        return None
    # parabolic interpolation on the correlation peak, and its half width as sigma
    y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
    denom = (y0 - 2 * y1 + y2)
    sub = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    dv = float(dvs[k] + sub * (dvs[1] - dvs[0]))
    half = cc >= (cc[k] + np.median(cc)) / 2.0
    width = float(dvs[half].max() - dvs[half].min()) if half.sum() > 1 else np.nan
    snr = float((cc[k] - np.median(cc)) / (np.std(cc) or 1))
    return dict(dv=dv, width=width, cc_snr=snr, n_lines=len(covered), n_px=int(use.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", nargs="*", default=None,
                    help="MAST directory names; default is every object")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "pipeline_output", "final",
                                                  "ism_zeropoint.csv"))
    a = ap.parse_args()
    names = a.objects
    if not names:
        names = sorted(os.listdir(MAST))
    zmap = {}
    import pandas as pd
    cat = pd.read_csv("/Users/gtr/Work/git/HST-Paper/Data/master_catalog_v22.csv")
    for col in ("common_name", "name_mast_key", "name_pub"):
        if col in cat.columns and "best_z" in cat.columns:
            for k, v in zip(cat[col], cat["best_z"]):
                if isinstance(k, str):
                    zmap.setdefault(k.strip(), v)
    rows = []
    for name in names:
        base = os.path.join(MAST, name)
        if not os.path.isdir(base):
            continue
        z = float(zmap.get(name, zmap.get(name.replace("-", " "), np.nan)))
        if not np.isfinite(z):
            z = 0.0
        for inst in sorted(os.listdir(base)):
            d = os.path.join(base, inst)
            if not os.path.isdir(d):
                continue
            try:
                raw = read_exposures(d, inst, 0.0)      # keep the observed frame
            except Exception:
                continue
            byfile = {}
            for r in raw:
                byfile.setdefault(r["file"], []).append(r)
            for fn, rs in byfile.items():
                g = str(rs[0].get("grating", "")).strip().upper()
                R = R_GRATING.get(g, 1000)
                # FOS writes one row per readout group on a shared wavelength
                # grid, so concatenating them stacks duplicate wavelengths and
                # makes the median dispersion zero -- which then asks for a
                # continuum window wider than the spectrum and rejects the file.
                # Average repeated groups; concatenate genuinely different
                # segments or echelle orders.
                same = (len({len(r["w"]) for r in rs}) == 1 and len(rs) > 1
                        and np.allclose(rs[0]["w"], rs[-1]["w"], equal_nan=True))
                if same:
                    w = np.asarray(rs[0]["w"], float)
                    f = np.nanmean(np.vstack([r["f"] for r in rs]), axis=0)
                else:
                    w = np.concatenate([r["w"] for r in rs])
                    f = np.concatenate([r["f"] for r in rs])
                res = measure(w, f, z, R)
                if res is None:
                    continue
                rows.append(dict(object=name, inst=inst, grating=g, file=fn, z=z,
                                 R=R, **{k: (round(v, 1) if isinstance(v, float) else v)
                                         for k, v in res.items()}))
                print("  %-22s %-5s %-7s dv %+7.0f +/- %5.0f km/s  (cc S/N %4.1f, %d lines)"
                      % (fn[:22], inst, g, res["dv"], res["width"] / 2, res["cc_snr"],
                         res["n_lines"]), flush=True)
    if rows:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        print("\nwrote %d rows -> %s" % (len(rows), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
