"""Line-region spectra for the redshift-check page: SDSS (optical) and the
HST/IUE co-add we fit (UV), as short windows on the 1e-4 dex log grid both use
(log10 lambda_obs of the first pixel + flux).  The page plots them against
v = c (lambda / (lambda0 (1+z)) - 1), the convention of the candidate offsets
dv = c (z_c - z)/(1+z).  Vacuum rest wavelengths.
GTR, 2026-11-19: in the UV only the C III] peak and Mg II can test the
redshift; Si IV, C IV and He II can be blueshifted, which is what we measure."""
import glob, json, math, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from astropy.io import fits
S = sys.argv[1]; C = 299792.458
FR = "/Users/gtr/Work/projects/hstica/finalrun"
LINES = [  # name, vacuum rest A, half-window km/s, source
    ("[O III] 5007", 5008.24, 3000, "sdss"),
    ("Hβ", 4862.68, 6000, "sdss"),
    ("Hα", 6564.61, 6000, "sdss"),
    ("[O II] 3727", 3728.48, 3000, "sdss"),
    ("Mg II", 2799.12, 6000, "both"),
    ("C III]", 1908.73, 6000, "uv"),
]
rows = json.load(open(S + "/zcheck_data.json"))
q = pd.read_csv("/Users/gtr/Work/git/HST-Paper/archive/provenance/work_queue.csv").set_index("order")
man = pd.read_csv(FR + "/pipeline_output/final/sdss_spectra.csv")
man = man[man.fn_sdss.notna()].set_index("row_num")

def uv_file(idx):
    for root, reb in (("pipeline_output/chosen_visit/fits", "pipeline_output/chosen_visit/rebin"), ("iue/fits", "iue/rebin")):
        for d in glob.glob("%s/%s/%03d_*" % (FR, root, idx)):
            if not glob.glob(d + "/records/BEST_*.json"): continue
            inst = d.rsplit("_", 1)[1]
            stems = [s for s in str(q.loc[idx, "stems"]).split(";") if s.endswith("_" + inst)] if idx in q.index else []
            cands = [os.path.join(FR, reb, s + ".fits") for s in stems]
            cands += glob.glob(os.path.join(FR, reb, "*_" + inst + ".fits")) if inst in ("IUE", "SPLICE") else []
            for c in cands:
                if os.path.exists(c) and (inst not in ("IUE", "SPLICE") or str(q.loc[idx, "name_mast_key"]) in os.path.basename(c)):
                    return c, inst
    return None, None

def windows(logl, flux, good, z, src):
    out = []
    for name, l0, hw, pref in LINES:
        if pref == "sdss" and src != "SDSS": continue
        if pref == "uv" and src == "SDSS": continue
        lam = 10 ** logl
        v = C * (lam / (l0 * (1 + z)) - 1)
        s = np.where(np.abs(v) <= hw)[0]
        if len(s) < 10: continue
        f = np.where(good[s], flux[s], np.nan)
        # the good data must span at least 60 per cent of the window, so a
        # spectrum that merely starts or ends inside it does not get a panel
        vg = v[s][np.isfinite(f)]
        if len(vg) < 10 or (vg.max() - vg.min()) < 0.6 * 2 * hw: continue
        med = np.nanmedian(np.abs(f)) or 1.0
        out.append({"line": name, "lam0": l0, "hw": hw, "src": src, "l0": round(float(logl[s[0]]), 6),
                    "step": round(float(np.median(np.diff(logl[s]))), 7),
                    "f": [None if not math.isfinite(x) else round(float(x / med), 3) for x in f]})
    return out

res = {}; nsd = nuv = 0
for r in rows:
    z = r["z"]; panels = []
    if r["row"] in man.index and isinstance(man.loc[r["row"], "fn_sdss"], str):
        p = FR + "/data_v23/SDSS_spec/lite/" + man.loc[r["row"], "fn_sdss"]
        if os.path.exists(p):
            d = fits.open(p)[1].data
            panels += windows(np.asarray(d["loglam"], float), np.asarray(d["flux"], float), np.asarray(d["ivar"], float) > 0, z, "SDSS"); nsd += 1
    fn, inst = uv_file(r["idx"]) if r.get("idx") else (None, None)
    if fn:
        t = fits.open(fn)[1].data
        zp = float(np.nanmedian(t["Redshift"]))
        lobs = np.log10(np.asarray(t["Rest-frame Wavelength"], float) * (1 + zp))
        panels += windows(lobs, np.asarray(t["Coadded Flux (Arbitrary Units)"], float), np.asarray(t["Bad Pixel Mask"]) == 0, z, inst); nuv += 1
    cands = [{"label": "z used in Table 1", "dv": 0.0, "kind": "ours"}]
    # a redshift revised during the check (override not yet in Table 1)
    if r.get("z_rev") is not None:
        cands.append({"label": "revised z", "dv": round(C * (r["z_rev"] - z) / (1 + z)), "kind": "rev"})
    if r.get("dv_ned") is not None: cands.append({"label": "NED preferred", "dv": r["dv_ned"], "kind": "ned"})
    if r["row"] in man.index and pd.notna(man.loc[r["row"], "z_sdss"]):
        cands.append({"label": "SDSS pipeline", "dv": round(C * (float(man.loc[r["row"], "z_sdss"]) - z) / (1 + z)), "kind": "sdss"})
    seen = []
    for p_ in sorted(r.get("pubs", []), key=lambda p: abs(p["dv"])):
        if any(abs(p_["dv"] - s) < 25 for s in seen): continue
        seen.append(p_["dv"]); cands.append({"label": "published " + str(p_.get("Refcode", "")), "dv": p_["dv"], "kind": "pub"})
    res[r["row"]] = {"panels": panels, "cands": cands}
json.dump(res, open(S + "/zcheck_spectra.json", "w"), separators=(",", ":"), allow_nan=False)
npan = sum(len(v["panels"]) for v in res.values())
print("objects %d, with SDSS %d, with UV co-add %d, panels %d, KB %d" % (len(res), nsd, nuv, npan, os.path.getsize(S + "/zcheck_spectra.json") // 1024))
