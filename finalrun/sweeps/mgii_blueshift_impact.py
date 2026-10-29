"""If Mg II had chosen the component set, how much would the C IV blueshift move?

The sweep says which set reproduces the observed Mg II line best.  That is only
interesting if it would change the number the paper reports.  This refits each
object where the Mg II-preferred set differs from the adopted one and compares
the C IV blueshift and EW under both.
"""
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINALRUN = os.path.dirname(HERE)
sys.path.insert(0, FINALRUN)
import object_diagnostics as diag

FINAL = os.path.join(FINALRUN, "pipeline_output", "final")
FITS_DIR = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "fits")
REBIN = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "rebin")
QUANTUM = 69.03

rows = []
d = pd.read_csv(os.path.join(FINAL, "mgii_would_differ.csv"))
for _, r in d.iterrows():
    i = int(r["index"])
    rec = json.load(open(glob.glob(os.path.join(FITS_DIR, "%d_*" % i, "records", "BEST_*.json"))[0]))
    stem, ov = rec["name"], rec["override"]
    want = None if r["mgii_prefers"] == "auto" else r["mgii_prefers"]
    try:
        alt = diag.fit(stem, ov, want, REBIN)
    except Exception as exc:
        print("  %4d  refit failed: %s" % (i, type(exc).__name__), flush=True)
        continue
    b0, e0 = rec["result"]["civ_blue"], rec["result"]["civ_ew"]
    b1, e1 = alt["civ_blue"], alt["civ_ew"]
    rows.append(dict(index=i, folder=r["folder"], manmask=r["manmask"],
                     adopted_comps=r["adopted_comps"], mgii_prefers=r["mgii_prefers"],
                     bal_status=r["bal_status"],
                     blue_adopted=b0, blue_mgii=b1, dblue=b1 - b0,
                     ew_adopted=e0, ew_mgii=e1,
                     dlogew=np.log10(e1) - np.log10(e0) if e0 > 0 and e1 > 0 else np.nan,
                     ratio_adopted=r["ratio_adopted"]))
    print("  %4d %-38s %-5s -> %-5s   blue %8.1f -> %8.1f  (%+8.1f, %4.1f quanta)   logEW %+.3f"
          % (i, str(r["folder"])[:38], r["adopted_comps"], r["mgii_prefers"],
             b0, b1, b1 - b0, abs(b1 - b0) / QUANTUM, rows[-1]["dlogew"]), flush=True)

t = pd.DataFrame(rows)
t.to_csv(os.path.join(FINAL, "mgii_blueshift_impact.csv"), index=False)
print("\n%d objects refit -> final/mgii_blueshift_impact.csv" % len(t))
