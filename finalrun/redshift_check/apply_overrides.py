"""Apply the live redshift overrides to the preserved page data
(zcheck_data.json), without refetching NED: for each object whose override
differs from the redshift now in the data, set the new z (code O), keep the
Table 1 value as z_table, and recompute every velocity offset (NED, each
published value, their range) relative to the new z."""
import json, sys
import numpy as np, pandas as pd
S = sys.argv[1]; C = 299792.458
rows = json.load(open(S + "/zcheck_data.json"))
ov = pd.read_csv("/Users/gtr/Work/git/HST-Paper/ica/redshift_overrides.csv")
key = {str(o).strip(): float(z) for o, z in zip(ov.object, ov.z)}
n = 0
for r in rows:
    z_new = key.get(str(r["name"]).strip(), key.get(str(r["name"]).replace(" ", "")))
    r.setdefault("z_table", r["z"])
    if z_new is None or abs(z_new - r["z"]) < 1e-7: continue
    z_old = r["z"]; r["z"] = z_new; r["zcode"] = "O"; n += 1
    r["fit_note"] = "measured at the previous z %.6f; refit pending" % r["z_table"]
    def dv(zc): return round(C * (zc - z_new) / (1 + z_new))
    if (r.get("ned") or {}).get("z") is not None: r["dv_ned"] = dv(r["ned"]["z"])
    for p in r.get("pubs", []):
        try: p["dv"] = dv(float(p["Published Redshift"]))
        except Exception: pass
    if r.get("pubs"):
        a = np.array([p["dv"] for p in r["pubs"]]); r.update(pub_lo=float(a.min()), pub_hi=float(a.max()), pub_med=float(np.median(a)))
json.dump(rows, open(S + "/zcheck_data.json", "w"), separators=(",", ":"), default=str, allow_nan=False)
print("updated", n, "objects; revised overall:", sum(1 for r in rows if abs(r["z_table"] - r["z"]) > 1e-7))
