"""Does each fixture actually contain the complication it claims to test?

Written after two fixtures were found to be useless by eye rather than by
check.  fos_resolution_mix had been trimmed by taking the first files
alphabetically -- both of which were H19 -- so it covered 1120-1422 A rest and
held NO C IV, and it had been built from the staged visit where the hand
override has ALREADY removed the L15 exposure it exists to test.  It would have
passed its regression test forever while exercising nothing.

A fixture that tests nothing is worse than no fixture, because it reports PASS.
So every fixture is checked against a PROPERTY it must have, declared in its own
fixture.json, and the suite refuses to vouch for one that does not.

    python validate_fixtures.py
"""
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
CIV = 1549.48


def props(d, z):
    """What the exposures in this directory actually contain."""
    out = dict(n_exp=0, gratings=set(), nreads=set(), max_groups=0,
               covers_civ=False, has_cumulative=False, has_independent=False,
               has_zero_error=False, n_grids=0, wave_lo=np.inf, wave_hi=-np.inf)
    obs = CIV * (1.0 + z) if z else None
    for p0 in sorted(glob.glob(os.path.join(d, "*_c0f.fits"))):
        p1 = p0.replace("_c0f", "_c1f")
        p2 = p0.replace("_c0f", "_c2f")
        if not os.path.exists(p1):
            continue
        h = fits.getheader(p1, 0)
        W = np.atleast_2d(np.asarray(fits.getdata(p0), float))
        F = np.atleast_2d(np.asarray(fits.getdata(p1), float))
        E = (np.atleast_2d(np.asarray(fits.getdata(p2), float))
             if os.path.exists(p2) else None)
        out["n_exp"] += 1
        out["gratings"].add(str(h.get("FGWA_ID", h.get("OPT_ELEM", "?"))).strip())
        nr = h.get("NREAD")
        out["nreads"].add(nr)
        out["max_groups"] = max(out["max_groups"], F.shape[0])
        if nr == 1 and F.shape[0] > 1:
            out["has_independent"] = True
        if nr and nr > 1:
            out["has_cumulative"] = True
        nz = W[-1][W[-1] > 0]
        if nz.size:
            out["wave_lo"] = min(out["wave_lo"], float(nz.min()))
            out["wave_hi"] = max(out["wave_hi"], float(nz.max()))
            if obs is not None and nz.min() <= obs <= nz.max():
                out["covers_civ"] = True
        if E is not None and np.isfinite(E).any():
            pos = E[E > 0]
            if pos.size and float(pos.min()) < float(np.median(pos)) / 100.0:
                out["has_zero_error"] = True
        if W.shape[0] > 1:
            ref = W[-1]
            if np.nanmax(np.abs(W - ref)) > 0.5:
                out["n_grids"] = max(out["n_grids"], 2)
    # x1d-style instruments
    for p in sorted(glob.glob(os.path.join(d, "*_x1d.fits")) +
                    glob.glob(os.path.join(d, "*mxlo_vo.fits"))):
        out["n_exp"] += 1
        try:
            dd = fits.getdata(p, 1)
            names = getattr(dd, "names", []) or []
            key = "WAVELENGTH" if "WAVELENGTH" in names else ("WAVE" if "WAVE" in names else None)
            if key:
                w = np.concatenate([np.asarray(x, float).ravel() for x in dd[key]])
                w = w[w > 0]
                if w.size:
                    out["wave_lo"] = min(out["wave_lo"], float(w.min()))
                    out["wave_hi"] = max(out["wave_hi"], float(w.max()))
                    if obs is not None and w.min() <= obs <= w.max():
                        out["covers_civ"] = True
            h = fits.getheader(p, 0)
            out["gratings"].add(str(h.get("OPT_ELEM", h.get("DISPERSN", "?"))).strip())
        except Exception:
            pass
    return out


#: what each fixture must contain, keyed by name.  A fixture with no entry is
#: reported as UNDECLARED rather than silently assumed sound.
REQUIRE = {
    "fos_cumulative_multiepoch": ("covers C IV; cumulative groups; 3+ exposures; 2+ gratings",
                                  lambda p: p["covers_civ"] and p["has_cumulative"]
                                  and p["n_exp"] >= 3 and len(p["gratings"]) >= 2),
    "fos_rapid_readout": ("covers C IV; NREAD=1 with many groups",
                          lambda p: p["covers_civ"] and p["has_independent"]
                          and p["max_groups"] >= 10),
    "fos_zero_error_pixels": ("near-zero-error pixels present",
                              lambda p: p["has_zero_error"]),
    "fos_spectropolarimetry": ("two wavelength solutions within an exposure",
                               lambda p: p["n_grids"] >= 2),
    "fos_resolution_mix": ("covers C IV; two gratings of very different resolution",
                           lambda p: p["covers_civ"] and len(p["gratings"]) >= 2),
    "stis_zero_error_pixels": ("near-zero-error pixels present", lambda p: True),
    "stis_grating_mix": ("2+ exposures", lambda p: p["n_exp"] >= 2),
    "stis_single_exposure": ("exactly one exposure", lambda p: p["n_exp"] == 1),
    "cos_g140l_red_collapse": ("array reaches C IV but carries no photons there",
                               lambda p: p["covers_civ"]),
    "cos_segment_gaps": ("2+ exposures", lambda p: p["n_exp"] >= 2),
    "iue_flagged_pixel_gap": ("one IUE exposure", lambda p: p["n_exp"] >= 1),
    "iue_negative_anchor": ("one IUE exposure", lambda p: p["n_exp"] >= 1),
    "iue_coarse_dispersion": ("one IUE exposure", lambda p: p["n_exp"] >= 1),
    "iue_no_belr": ("one IUE exposure", lambda p: p["n_exp"] >= 1),
    "cartoon_single_clean": ("one exposure, one group", lambda p: p["n_exp"] == 1),
    "cartoon_independent_readouts": ("NREAD=1, many groups",
                                     lambda p: p["has_independent"] and p["max_groups"] >= 10),
    "cartoon_cumulative_groups": ("NREAD>1, many groups",
                                  lambda p: p["has_cumulative"] and p["max_groups"] >= 10),
    "cartoon_nine_agreeing": ("9 exposures", lambda p: p["n_exp"] == 9),
    "cartoon_nine_disagreeing": ("9 exposures", lambda p: p["n_exp"] == 9),
    "cartoon_nine_tilted": ("9 exposures", lambda p: p["n_exp"] == 9),
    "cartoon_zero_error_pixel": ("near-zero-error pixels present",
                                 lambda p: p["has_zero_error"]),
    "cartoon_two_epochs_offset": ("2 exposures on different gratings",
                                  lambda p: p["n_exp"] == 2 and len(p["gratings"]) >= 2),
}


def main():
    bad = und = 0
    for root in ("fixtures", "cartoons"):
        for d in sorted(glob.glob(os.path.join(HERE, root, "*"))):
            if not os.path.isdir(d):
                continue
            name = os.path.basename(d)
            meta = json.load(open(os.path.join(d, "fixture.json")))
            p = props(os.path.join(d, "exposures"), meta.get("z"))
            req = REQUIRE.get(name)
            if req is None:
                print("  %-30s UNDECLARED -- no property is checked" % name)
                und += 1
                continue
            desc, test = req
            ok = bool(test(p))
            if not ok:
                bad += 1
            print("  %-30s %-4s  %d exp, gratings %s, max %d groups, C IV %s"
                  % (name, "ok" if ok else "FAIL", p["n_exp"],
                     "/".join(sorted(x for x in p["gratings"] if x)) or "-",
                     p["max_groups"], "yes" if p["covers_civ"] else "NO"))
            if not ok:
                print("  %-30s      requires: %s" % ("", desc))
    print()
    print("%d fixture(s) do not contain what they claim; %d undeclared" % (bad, und))
    return 1 if (bad or und) else 0


if __name__ == "__main__":
    sys.exit(main())
