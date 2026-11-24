"""Literature optical spectra (NED archive) as panels for the redshift-check
page, for the objects with no SDSS spectrum of their own.
  * wavelengths from the IRAF WCS (CRVAL1, CD1_1/CDELT1, CRPIX1; DC-FLAG 1 =
    log10 lambda); flux from the first row of a multi-row image;
  * air -> vacuum (Morton 1991) unless the file says VACUUM or is SDSS;
  * frame test: a narrow-line template ([O II], Hb, [O III], Ha, [N II]) is
    scored at observed (lambda0 (1+z)) and rest (lambda0) positions; rest is
    taken only if it scores clearly better, and the choice is recorded;
  * H I 21-cm profiles (VHELIO column, optical convention cz) become a
    velocity panel.
Infrared spectra (Herschel, ISO, wavelengths in microns) are skipped."""
import glob, json, math, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits
S = sys.argv[1]; C = 299792.458
spec = json.load(open(S + "/zcheck/zcheck_spectra.json"))
rows = {r["row"]: r for r in json.load(open(S + "/zcheck/zcheck_data.json"))}
REF = {"1992ApJS...80..109B": "Boroson & Green 1992", "2009MNRAS.399..683J": "6dFGS (Jones et al. 2009)", "2008ApJS..178..280B": "2008ApJS..178..280B",
       "2009A+A...495.1033B": "2009A&A...495.1033B", "1992ApJS...79..255K": "1992ApJS...79..255K",
       "1995ApJS...98..129K": "1995ApJS...98..129K", "2015ApJS..217...13M": "2015ApJS..217...13M"}
LINES = [("[O III] 5007", 5008.24, 3000), ("Hβ", 4862.68, 6000), ("Hα", 6564.61, 6000),
         ("[O II] 3727", 3728.48, 3000), ("Mg II", 2799.12, 6000)]
TEMPLATE = [2799.12, 3728.48, 4862.68, 4960.30, 5008.24, 6564.61, 6585.27]
# Bright night-sky lines (vacuum): a window on one of these proves nothing about
# the frame (PKS 0420-01's 6dF spectrum scored "rest" off 5577 and 6300 A).
SKY = [5578.9, 6302.0, 6365.5]

def air_to_vac(w):
    s2 = (1e4 / w) ** 2
    return w * (1 + 6.4328e-5 + 2.94981e-2 / (146 - s2) + 2.5540e-4 / (41 - s2))

def score(lam, f, z):
    t = 0.0
    for l0 in TEMPLATE:
        c = l0 * (1 + z)
        if any(abs(c - k) < 12 for k in SKY): continue
        s = (lam > c - 8) & (lam < c + 8); b = (np.abs(lam - c) > 15) & (np.abs(lam - c) < 60)
        if s.sum() > 2 and b.sum() > 4:
            # significance against the local noise, counting only real lines:
            # summing raw peak heights let noise at many template positions
            # outscore a real line (PKS 0420-01)
            med = np.nanmedian(f[b]); noise = 1.4826 * np.nanmedian(np.abs(f[b] - med)) + 1e-30
            t += max(0.0, (np.nanmax(f[s]) - med) / noise - 3.0)
    return t

def panels_from(lam, f, z, src):
    out = []
    for name, l0, hw in LINES:
        v = C * (lam / (l0 * (1 + z)) - 1); s = np.where(np.abs(v) <= hw)[0]
        if len(s) < 10: continue
        y = f[s]; ok = np.isfinite(y)
        if ok.sum() < 10 or (v[s][ok].max() - v[s][ok].min()) < 0.6 * 2 * hw: continue
        med = np.nanmedian(np.abs(y)) or 1.0
        out.append({"line": name, "lam0": l0, "hw": hw, "src": src, "lam": [round(float(x), 3) for x in lam[s]],
                    "f": [None if not math.isfinite(x) else round(float(x / med), 3) for x in y]})
    return out

from astropy.coordinates import SkyCoord
import astropy.units as u
def our_coord(row):
    des = rows[row]["des"][1:]; i = max(des.find("+"), des.find("-")); ra, dec = des[:i], des[i:]
    return SkyCoord("%s:%s:%s %s:%s:%s" % (ra[0:2], ra[2:4], ra[4:], dec[0:3], dec[3:5], dec[5:]), unit=(u.hourangle, u.deg))
def file_sep(hdr, row):
    """Separation in arcsec between the file's own position and ours, or None
    if it gives none.  Conventions met in NED's files: SDSS gives the target in
    RAOBJ/DECOBJ (RA/DEC there are the plate centre) or PLUG_RA/DEC; TNG files
    write RA in HOURS as a plain number; older files may be B1950."""
    for kr, kd in (("RAOBJ", "DECOBJ"), ("PLUG_RA", "PLUG_DEC"), ("RA", "DEC")):
        if kr in hdr and kd in hdr: ra, dec = hdr[kr], hdr[kd]; break
    else:
        return None
    try:
        eq = float(hdr.get("EQUINOX", hdr.get("EPOCH", 2000)) or 2000)
        fr = dict(frame="fk5", equinox="J%g" % eq) if eq > 1990 else dict(frame="fk4", equinox="B%g" % eq)
        if isinstance(ra, str) and ":" in ra:
            cands = [SkyCoord(ra, dec, unit=(u.hourangle, u.deg), **fr)]
        else:
            cands = [SkyCoord(float(ra), float(dec), unit=u.deg, **fr)]
            if 0 <= float(ra) <= 24:      # a plain number that may be hours
                cands.append(SkyCoord(float(ra) * 15.0, float(dec), unit=u.deg, **fr))
        return min(float(c.transform_to("icrs").separation(our_coord(row)).arcsec) for c in cands)
    except Exception:
        return None
log = []
for fj in sorted(glob.glob(S + "/optical/[0-9]*.json")):
    j = json.load(open(fj)); row = j["row"]; z = rows[row]["z"]; new = []
    for sp_ in j["spectra"]:
        fn = S + "/optical/fits/" + sp_["file"]
        # the bibcode is whichever URL path element looks like one
        bib = next((x for x in sp_["url"].split("/") if re.match(r"^(19|20)\d\d[A-Za-z&+.]", x)), sp_["url"].split("/")[-2])
        src = REF.get(bib, bib if "." in bib else "NED archive")
        # Boroson & Green (1992) spectra are de-redshifted with their own
        # redshifts (3 decimals, ~150 km/s), which NED does not list, so they
        # cannot test ours.  Exclude them outright (the frame test would too).
        if bib.startswith("1992ApJS...80..109B"):
            log.append((row, sp_["file"], "rest (excluded)", "Boroson & Green 1992 (de-redshifted)")); continue
        try:
            h = fits.open(fn)
            if isinstance(h[1] if len(h) > 1 else None, fits.BinTableHDU) and "VHELIO" in h[1].columns.names:
                d = h[1].data; vh = np.asarray(d["VHELIO"], float); fl = np.asarray(d["FLUX"], float)
                dv = C * (vh / C - z) / (1 + z); s = np.abs(dv) <= 1500
                if s.sum() > 10:
                    med = np.nanmedian(np.abs(fl[s])) or 1.0
                    new.append({"line": "H I 21 cm", "hw": 1500, "src": src + " (radio)", "v": [round(float(x), 1) for x in dv[s]],
                                "f": [None if not math.isfinite(x) else round(float(x / med), 3) for x in fl[s]]})
                    log.append((row, sp_["file"], "HI", src))
                continue
            hd = h[0]
            # 6dF files hold the V arm, the R arm and the two combined ("SPECTRUM
            # VR"); use the combined spectrum, or Hb/[O III] are lost at z~0.3.
            for ext in h:
                if "VR" in str(ext.header.get("EXTNAME", "")) and ext.data is not None: hd = ext; break
            hdr = hd.header
            if hd.data is None or "CRVAL1" not in hdr: continue
            tel = str(hdr.get("TELESCOP", "") or h[0].header.get("TELESCOP", ""))
            if "Herschel" in tel or "ISO" in tel: continue
            f = np.asarray(hd.data, float); f = f if f.ndim == 1 else f.reshape(-1, f.shape[-1])[0]
            # Dispersion: the original IRAF keywords first.  Boroson & Green
            # (1992) files carry W0/WPC = CRVAL1/CDELT1 (e.g. 2.4367 A/px) and a
            # rounded CD1_1 = 2.5 added later; trusting CD1_1 drifted ~50 A over
            # the spectrum and scattered [O III] by +/-1500 km/s.
            if hdr.get("WPC"):
                w0, d1, p = hdr.get("W0", hdr["CRVAL1"]), hdr["WPC"], 1.0
            else:
                w0, d1, p = hdr["CRVAL1"], (hdr.get("CDELT1") or hdr.get("CD1_1")), hdr.get("CRPIX1", 1)
            w = w0 + (np.arange(f.size) + 1 - p) * d1
            lam = 10 ** w if (hdr.get("DC-FLAG", 0) == 1 or "LOG" in str(hdr.get("CTYPE1", "")).upper()) else w
            if lam.max() < 3000 or lam.min() > 20000: continue
            # Whose spectrum is it?  NED can attach another object's spectrum
            # (Mrk 352 carried one of HCG 20A, NGC 4253 one of SS 433).  Where
            # the file records a position, require it within 120": older files
            # record telescope pointing, good only to ~1 arcmin (the Boroson &
            # Green 1992 files sit 30-90" from their objects).  Always show OBJECT.
            sep = file_sep(hdr, row)
            if sep is not None and sep > 120:
                log.append((row, sp_["file"], "wrong position (excluded)", "%s at %.0f arcsec" % (hdr.get("OBJECT", "?"), sep))); continue
            # Too coarse to test a redshift to a fraction of a 69 km/s pixel.
            pix = C * abs(d1) * (math.log(10) if (hdr.get("DC-FLAG", 0) == 1) else 1.0 / float(np.median(lam)))
            if pix > 250:
                log.append((row, sp_["file"], "too coarse (excluded)", "%.0f km/s per pixel" % pix)); continue
            vac = bool(hdr.get("VACUUM", False)) or "SDSS" in tel
            if not vac: lam = air_to_vac(lam)
            so, sr = score(lam, f, z), score(lam, f, 0.0)
            frame = "rest" if (sr > 2 * so and sr > 2.0) else "observed"
            # A file already shifted to the rest frame cannot test our redshift:
            # its lines sit where the AUTHOR's redshift put them, which we do not
            # know.  Leave it out and record it.
            if frame == "rest":
                log.append((row, sp_["file"], "rest (excluded)", src)); continue
            obj = str(hdr.get("OBJECT", "") or hdr.get("OBJCAT", "")).strip()
            label = ("SDSS (via NED)" if "SDSS" in tel else src) + ("" if vac else ", air→vac") + (", file: " + obj if obj else ", file names no object")
            got = panels_from(lam, f, z, label); new += got
            log.append((row, sp_["file"], frame, label, len(got)))
        except Exception as ex:
            log.append((row, sp_["file"], "ERROR", str(ex)[:60]))
    if new:
        spec[str(row)]["panels"] = new + [p for p in spec[str(row)]["panels"] if p["src"] != "SDSS" and "lam" not in p and "v" not in p]
json.dump(spec, open(S + "/zcheck/zcheck_spectra.json", "w"), separators=(",", ":"), allow_nan=False)
json.dump(log, open(S + "/optical/build_log.json", "w"))
from collections import Counter
print(Counter(x[2] for x in log))
lit = [k for k, v in spec.items() if any(("lam" in p) or ("v" in p) for p in v["panels"])]
print("objects with literature optical/HI panels:", len(lit))
print("errors:", [x for x in log if x[2] == "ERROR"][:5])
print("rest-frame files excluded:", [x[:2] for x in log if x[2].startswith("rest")])
