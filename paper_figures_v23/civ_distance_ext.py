"""
C IV distance, with the best-fit curve optionally extended past its high-EW end.

Trevor McCaffrey's curve (CIV_Distance/data/bestfit.npy) is fitted to SDSS
quasars and stops at log EW = 2.50 (EW = 316 A), blueshift 50 km/s, which is
D_CIV = 0.  Sixteen of our HST objects have higher EW; with the curve as it is
they all project onto that end and get D_CIV = 0, whatever their EW
(Mrk 1310 at 1139 A and Mrk 352 at 321 A come out the same).

With EXTEND_CURVE = True the curve is continued straight up from its end, at
blueshift 50 km/s (over its last 200 points the curve is within 1 degree of
vertical), to log EW = EXT_TOP, at the curve's own point spacing.  Distances
are then measured exactly as before -- Trevor's CIV_distance() on the
lengthened curve -- and the length of the extension is subtracted, so that
D_CIV = 0 stays at the old end of the curve:

  * every object that projects onto the original curve keeps its value
    (checked in civ_distance below);
  * objects beyond the end get D_CIV < 0, measured along the extension in the
    same scaled units.

REVERSIBLE: set EXTEND_CURVE = False to get Trevor's values back everywhere
(Table 1, Fig 4, Fig 6a all call civ_distance here).  Trevor's repository and
curve are not modified.  Table 1's CSV carries both values (CIV_dist and
CIV_dist_trevor).
"""

import os
import sys
import numpy as np

CIVDIST_DIR = os.path.expanduser("~/Work/git/CIV_Distance/")
sys.path.insert(0, CIVDIST_DIR)
from CIVfunctions import CIV_distance

EXTEND_CURVE = True     # False = Trevor's curve unchanged
EXT_TOP = 3.5           # log EW the extension reaches (sample max is 3.06)


def bestfit_curve():
    """Trevor's curve, (blueshift, log EW), from the high-EW end down."""
    return np.load(CIVDIST_DIR + "data/bestfit.npy")


def extension():
    """The vertical extension above the curve's end, top first.

    Returns (points, length): the extension's points (blueshift, log EW) and
    its length in the scaled units D_CIV is measured in.
    """
    import joblib
    bf = bestfit_curve()
    scaler = joblib.load(CIVDIST_DIR + "scalers/scaler_logEW.save")
    step = np.median(np.abs(np.diff(bf[:200, 1])))          # curve's own spacing
    y = np.arange(EXT_TOP, bf[0, 1], -step)                 # from the top down
    pts = np.column_stack([np.full(len(y), bf[0, 0]), y])
    s = scaler.transform(np.vstack([pts, bf[:1]]))
    length = float(np.sum(np.hypot(*np.diff(s, axis=0).T)))
    return pts, length


def civ_distance(blueshift, log_ew):
    """D_CIV for arrays of blueshift (km/s) and log10 EW (A).

    Uses the extended curve when EXTEND_CURVE is True, Trevor's otherwise.
    """
    data = np.column_stack([blueshift, log_ew])
    bf = bestfit_curve()
    d_trevor = CIV_distance(data, bf, path=CIVDIST_DIR)
    assert len(d_trevor) == len(data), "CIV_distance dropped a point"
    if not EXTEND_CURVE:
        return d_trevor
    pts, length = extension()
    d = CIV_distance(data, np.vstack([pts, bf]), path=CIVDIST_DIR)
    assert len(d) == len(data), "CIV_distance dropped a point"
    d = d - length
    # objects on the original curve must keep Trevor's value
    on_curve = d_trevor > 1e-3
    assert np.allclose(d[on_curve], d_trevor[on_curve], atol=2e-4), \
        "extension changed a value on the original curve"
    return d
