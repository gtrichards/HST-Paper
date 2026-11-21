"""Systemic velocity from the narrow optical lines in the SDSS spectrum,
relative to the redshift used (Table 1), for the redshift double-check.
[O III] 4959+5007: core + wing (two Gaussians each, 4959 = 5007/2.98 with the
same kinematics), linear continuum, fitted over 4925-5045 A rest.
[O II] 3726+3729: one velocity and width, amplitude ratio 0.35-1.5, linear
continuum, 3700-3760 A.  Vacuum wavelengths.  Errors from the covariance,
scaled by sqrt(chi2/dof) when that exceeds 1."""
import json, sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from astropy.io import fits
from scipy.optimize import curve_fit
# Usage: python fit_narrow_redshift.py WORKDIR  (WORKDIR/zcheck/zcheck_data.json lists
# the objects and the redshift used; writes WORKDIR/zfit/narrow_fits.csv)
S = sys.argv[1]; C = 299792.458
FR = "/Users/gtr/Work/projects/hstica/finalrun"
rows = {r["row"]: r for r in json.load(open(S + "/zcheck/zcheck_data.json"))}
man = pd.read_csv(FR + "/pipeline_output/final/sdss_spectra.csv").set_index("row_num")
O3, O3b, O2a, O2b = 5008.240, 4960.295, 3727.092, 3729.875
def g(x, a, mu, s): return a * np.exp(-0.5 * ((x - mu) / s) ** 2)
def o3model(v, c0, c1, a1, v1, s1, a2, v2, s2):
    # v = velocity grid relative to 5007 at z; 4959 at its own velocity offset
    off = C * (O3b / O3 - 1)
    return (c0 + c1 * v / 1e3 + g(v, a1, v1, s1) + g(v, a1 / 2.98, v1 + off, s1)
            + g(v, a2, v2, s2) + g(v, a2 / 2.98, v2 + off, s2))
def o2model(v, c0, c1, a, r, v0, s):
    off = C * (O2b / O2a - 1)
    return c0 + c1 * v / 1e3 + g(v, a, v0, s) + g(v, a * r, v0 + off, s)
out = []
for row, r in rows.items():
    if row not in man.index or not isinstance(man.loc[row, "fn_sdss"], str): continue
    p = FR + "/data_v23/SDSS_spec/lite/" + man.loc[row, "fn_sdss"]
    if not os.path.exists(p): continue
    d = fits.open(p)[1].data; lam = 10 ** np.asarray(d["loglam"], float); f = np.asarray(d["flux"], float); iv = np.asarray(d["ivar"], float)
    z = r["z"]; rec = {"row": row, "name": r["name"], "z": z, "zcode": r["zcode"] or "W"}
    rest = lam / (1 + z)
    # [O III]
    m = (rest > 4925) & (rest < 5045) & (iv > 0)
    if m.sum() > 60:
        v = C * (rest[m] / O3 - 1); y = f[m]; e = 1 / np.sqrt(iv[m])
        pk = np.nanmax(y[np.abs(v) < 1500]) - np.median(y)
        try:
            p0 = [np.median(y), 0, max(pk, 1e-3) * .7, 0, 150, max(pk, 1e-3) * .3, -150, 600]
            lo = [-np.inf, -np.inf, 0, -1500, 30, 0, -2500, 200]; hi = [np.inf, np.inf, np.inf, 1500, 450, np.inf, 2500, 2500]
            pp, cv = curve_fit(o3model, v, y, p0=p0, sigma=e, bounds=(lo, hi), maxfev=20000)
            res = (y - o3model(v, *pp)) / e; red = max(1, np.sum(res ** 2) / (len(y) - 8))
            err = np.sqrt(np.diag(cv)) * np.sqrt(red)
            snr = pp[2] / err[2] if err[2] > 0 else 0
            rec.update(o3_v=round(pp[3], 1), o3_err=round(err[3], 1), o3_sig=round(pp[4], 0), o3_wing_v=round(pp[6], 0),
                       o3_wing_frac=round(pp[5] * pp[7] / (pp[2] * pp[4] + pp[5] * pp[7] + 1e-30), 2), o3_snr=round(snr, 1))
        except Exception as ex:
            rec["o3_fail"] = str(ex)[:60]
    # [O II]
    m = (rest > 3700) & (rest < 3760) & (iv > 0)
    if m.sum() > 30:
        v = C * (rest[m] / O2a - 1); y = f[m]; e = 1 / np.sqrt(iv[m])
        try:
            pk = np.nanmax(y) - np.median(y)
            pp, cv = curve_fit(o2model, v, y, p0=[np.median(y), 0, max(pk, 1e-3) * .6, 1.0, 0, 150], sigma=e,
                               bounds=([-np.inf, -np.inf, 0, .35, -1500, 30], [np.inf, np.inf, np.inf, 1.5, 1500, 450]), maxfev=20000)
            res = (y - o2model(v, *pp)) / e; red = max(1, np.sum(res ** 2) / (len(y) - 6)); err = np.sqrt(np.diag(cv)) * np.sqrt(red)
            rec.update(o2_v=round(pp[4], 1), o2_err=round(err[4], 1), o2_snr=round(pp[2] / err[2], 1) if err[2] > 0 else 0)
        except Exception as ex:
            rec["o2_fail"] = str(ex)[:60]
    out.append(rec)
pd.DataFrame(out).to_csv(S + "/zfit/narrow_fits.csv", index=False)
print("fitted", len(out))
