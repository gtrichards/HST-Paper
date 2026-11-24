"""More optical spectra for the objects with neither SDSS nor a usable NED
spectrum: HST optical gratings already on disk, a fresh SDSS DR18 query,
LAMOST and the ESO archive.  Conventions per source (all converted to vacuum,
heliocentric/barycentric):
  SDSS, LAMOST   vacuum, already corrected (headers say VACUUM = True)
  HST STIS       vacuum (STIS convention); longest exposure per grating
  HST FOS        air above 2000 A -> vacuum; readouts averaged (NREAD 1) or
                 last readout (cumulative)
  ESO X-shooter, UVES, FORS2   air (TUCD1 'em.wl;obs.atmos') -> vacuum, and
                 topocentric (SPECSYS = TOPOCENT) -> barycentric, using
                 ESO QC VRAD BARYCOR when present, else computed for Paranal
Spectra finer than 20 km/s per pixel are binned to ~25 km/s."""
import glob, json, math, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
S = sys.argv[1]; C = 299792.458; O2 = S + "/optical2"
exec(open(S + "/zcheck/build_optical.py").read().split("log = []")[0].split("S = sys.argv[1]")[1].split("\n", 1)[1]
     .replace("spec = json.load", "_unused = json.load").replace("rows = {r", "rows = {r"), globals())
spec = json.load(open(S + "/zcheck/zcheck_spectra.json"))
rows = {r["row"]: r for r in json.load(open(S + "/zcheck/zcheck_data.json"))}
PARANAL = EarthLocation.from_geodetic(-70.4045 * u.deg, -24.6272 * u.deg, 2635 * u.m)
M = "/Users/gtr/Work/projects/hstica/finalrun/data_v23/MAST_v23"

def bin25(lam, f):
    pix = C * np.median(np.diff(lam) / lam[:-1])
    if pix >= 20: return lam, f
    k = int(round(25 / pix)); n = len(lam) // k * k
    return lam[:n].reshape(-1, k).mean(1), np.nanmean(f[:n].reshape(-1, k), 1)

def add(row, lam, f, label, log):
    o = np.argsort(lam); lam, f = bin25(np.asarray(lam, float)[o], np.asarray(f, float)[o])
    got = panels_from(lam, f, rows[row]["z"], label)
    if got:
        spec[str(row)]["panels"] = got + spec[str(row)]["panels"]
    log.append((row, label, len(got)))

log = []
T = json.load(open(O2 + "/targets.json"))
# --- HST optical gratings on disk
hst = json.load(open(O2 + "/hst_optical.json"))
import pandas as pd
q = pd.read_csv("/Users/gtr/Work/git/HST-Paper/archive/provenance/work_queue.csv").set_index("row_num")
best = {}
for (row, name, inst, oe, ex, lo, hi, cov, fn) in hst:
    if oe in ("PRI",): continue
    k = (row, inst, oe)
    if k not in best or ex > best[k][0]: best[k] = (ex, fn)
for (row, inst, oe), (ex, fn) in best.items():
    key = q.loc[row, "name_mast_key"]; d = os.path.join(M, str(key), inst)
    try:
        if inst == "STIS":
            t = fits.getdata(os.path.join(d, fn), 1)
            lam = np.concatenate([np.asarray(x, float) for x in t["WAVELENGTH"]]); f = np.concatenate([np.asarray(x, float) for x in t["FLUX"]])
            lab = "HST STIS %s (%.0f s), vacuum" % (oe, ex)
        else:
            root = fn.replace("_c0f.fits", "")
            w = np.asarray(fits.getdata(os.path.join(d, root + "_c0f.fits")), float)
            h1 = fits.open(os.path.join(d, root + "_c1f.fits"))[0]; fl = np.asarray(h1.data, float)
            w = w.reshape(-1, w.shape[-1])[0]; fl = fl.reshape(-1, fl.shape[-1])
            fl = fl.mean(0) if int(h1.header.get("NREAD", 1) or 1) == 1 else fl[-1]
            lam = np.where(w > 2000, air_to_vac(np.clip(w, 2000, None)), w); f = fl
            lab = "HST FOS %s, air→vac above 2000 A" % oe
        add(row, lam, f, lab, log)
    except Exception as ex_:
        log.append((row, "HST %s %s ERROR %s" % (inst, oe, str(ex_)[:50]), 0))
# --- downloaded SDSS / LAMOST / ESO
for m in json.load(open(O2 + "/downloads.json")):
    if not m["ok"]: continue
    row = m["row"]; fn = O2 + "/files/" + m["file"]
    try:
        h = fits.open(fn)
        if m["src"].startswith("SDSS"):
            d = h[1].data; lam = 10 ** np.asarray(d["loglam"], float); f = np.where(np.asarray(d["ivar"]) > 0, np.asarray(d["flux"], float), np.nan)
            add(row, lam, f, "SDSS DR18 %s, vacuum" % m["file"].split("_sdss_")[1].replace(".fits", ""), log)
        elif m["src"].startswith("LAMOST"):
            d = h[1].data; lam = np.asarray(d["WAVELENGTH"][0], float); f = np.where(np.asarray(d["IVAR"][0]) > 0, np.asarray(d["FLUX"][0], float), np.nan)
            add(row, lam, f, "%s obsid %s, vacuum" % (m["src"], m["meta"].get("catalogue_obsid")), log)
        else:
            hd0 = h[0].header; d = h[1].data
            unit = str(h[1].header.get("TUNIT1", "")).lower()
            lam = np.asarray(d["WAVE"][0], float) * (10.0 if unit.startswith("nm") else 1.0)
            col = "FLUX" if "FLUX" in d.columns.names else "FLUX_REDUCED"
            f = np.asarray(d[col][0], float)
            if "QUAL" in d.columns.names: f = np.where(np.asarray(d["QUAL"][0]) == 0, f, np.nan)
            # UVES/X-shooter merged spectra carry garbage at the ends of each
            # arm (+/-40000 against a median of ~10 for I Zw 1 and QSO
            # J0635-7516): drop the outer 20 A and anything beyond 50x the median.
            med = np.nanmedian(np.abs(f))
            f = np.where((lam < lam.min() + 20) | (lam > lam.max() - 20) | (np.abs(f) > 50 * med), np.nan, f)
            bc = hd0.get("ESO QC VRAD BARYCOR")
            if bc is None:
                sc = SkyCoord(hd0["RA"] * u.deg, hd0["DEC"] * u.deg)
                bc = sc.radial_velocity_correction(kind="barycentric", obstime=Time(hd0["MJD-OBS"], format="mjd"), location=PARANAL).to(u.km / u.s).value
            lam = air_to_vac(lam) * (1 + float(bc) / C)
            add(row, lam, f, "%s (%s), air→vac, barycentric %+.1f km/s, file: %s" % (m["src"], m["meta"].get("dp_id", "")[:24], float(bc), str(hd0.get("OBJECT", "")).strip()), log)
    except Exception as ex_:
        log.append((row, "%s ERROR %s" % (m["src"], str(ex_)[:60]), 0))
json.dump(spec, open(S + "/zcheck/zcheck_spectra.json", "w"), separators=(",", ":"), allow_nan=False)
json.dump(log, open(O2 + "/build2_log.json", "w"))
print("added panels:", sum(x[2] for x in log), " from", len(log), "spectra; errors:", [x for x in log if "ERROR" in x[1]][:6])
covered = [k for k, v in spec.items() if any(p["src"] == "SDSS" or "lam" in p or "v" in p for p in v["panels"])]
print("objects with any optical/HI spectrum now:", len(covered), "of", len(spec))
