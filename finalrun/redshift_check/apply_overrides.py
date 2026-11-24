"""Apply the live redshift overrides to the preserved page data
(zcheck_data.json) without refetching NED.  The page's reference redshift is
ALWAYS the one in the current Table 1 (z, kept also as z_table); an override
that differs from it is stored as z_rev and drawn on the page as the amber
"revised" line (GTR, 2026-11-24: "all the solid black refer to what was used in
the current Table 1").  Every velocity offset is relative to the Table 1 z."""
import json, sys
import numpy as np, pandas as pd
S = sys.argv[1]; C = 299792.458
rows = json.load(open(S + "/zcheck_data.json"))
ov = pd.read_csv("/Users/gtr/Work/git/HST-Paper/ica/redshift_overrides.csv")
key = {str(o).strip(): float(z) for o, z in zip(ov.object, ov.z)}
n = 0
for r in rows:
    zt = r.get("z_table", r["z"]); r["z"] = zt; r["z_table"] = zt
    r.pop("fit_note", None); r.pop("z_rev", None)
    if r.get("zcode") == "O" and r.get("zcode_table") is not None: r["zcode"] = r["zcode_table"]
    def dv(zc): return round(C * (zc - zt) / (1 + zt))
    if (r.get("ned") or {}).get("z") is not None: r["dv_ned"] = dv(r["ned"]["z"])
    for p in r.get("pubs", []):
        try: p["dv"] = dv(float(p["Published Redshift"]))
        except Exception: pass
    if r.get("pubs"):
        a = np.array([p["dv"] for p in r["pubs"]]); r.update(pub_lo=float(a.min()), pub_hi=float(a.max()), pub_med=float(np.median(a)))
    z_new = key.get(str(r["name"]).strip(), key.get(str(r["name"]).replace(" ", "")))
    if z_new is not None and abs(z_new - zt) > 1e-7:
        r["z_rev"] = z_new; n += 1
json.dump(rows, open(S + "/zcheck_data.json", "w"), separators=(",", ":"), default=str, allow_nan=False)
print("objects with a revised z (amber line):", n)
