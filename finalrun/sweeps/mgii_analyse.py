"""What the Mg II sweep says, and whether incorporating SDSS would change anything.

Reads final/mgii_sweep.csv (the measurements) and final/bal_status.csv (the
curated annotation) and answers the question GTR deferred to the end of the
pass: would using the SDSS Mg II line have changed the adopted component set,
and if so by how much would the C IV blueshift move?

Mg II is a CHECK here, never a constraint: these objects have no HST data at
2800 A, so the reconstruction's Mg II is an extrapolation from the C IV region.
A mismatch is evidence about the component library, not grounds to re-grade.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINALRUN = os.path.dirname(HERE)
FINAL = os.path.join(FINALRUN, "pipeline_output", "final")
SETS = ["auto", "low", "mod", "high"]


def main():
    t = pd.read_csv(os.path.join(FINAL, "mgii_sweep.csv"))
    bal = pd.read_csv(os.path.join(FINAL, "bal_status.csv"))
    t = t.merge(bal[["index", "bal_status"]], on="index", how="left")
    t["bal_status"] = t["bal_status"].fillna("none")

    print("=" * 78)
    print("SAMPLE")
    print("=" * 78)
    print("  adopted objects with SDSS Mg II coverage : %d" % len(t))
    print("  Mg II also in the HST spectrum (FITTED)  : %d  -- reported separately" % t["mgii_in_hst"].sum())
    g = t[t["guard"].notna()]
    print("  GUARDED (not measured)                   : %d" % len(g))
    for why, n in g["guard"].value_counts().items():
        print("        %-58s %d" % (str(why)[:58], n))

    ok = t[t["guard"].isna() & t["ratio_adopted"].notna() & ~t["mgii_in_hst"]].copy()
    print("  CHECK SAMPLE (SDSS-only Mg II, measured)  : %d" % len(ok))
    print("  of which BAL / not-BAL / unclear / none   : %s"
          % dict(ok["bal_status"].value_counts()))

    print()
    print("=" * 78)
    print("DOES THE C IV-DRIVEN RECONSTRUCTION REPRODUCE THE OBSERVED Mg II?")
    print("  ratio = model Mg II EW / SDSS Mg II EW, both against a local")
    print("  linear continuum from 2650-2700 and 2900-2950 A")
    print("=" * 78)

    def band(s, lab):
        if not len(s):
            print("  %-26s --" % lab); return
        q = np.percentile(s, [16, 50, 84])
        print("  %-26s n=%3d   median %.2f   16-84%% %.2f-%.2f   min %.2f max %.2f"
              % (lab, len(s), q[1], q[0], q[2], s.min(), s.max()))

    band(ok["ratio_adopted"], "all measured")
    for st in ("none", "bal", "not_bal", "unclear"):
        band(ok[ok["bal_status"] == st]["ratio_adopted"], "  bal_status = " + st)

    nb = ok[ok["bal_status"] == "none"]["ratio_adopted"]
    bb = ok[ok["bal_status"] == "bal"]["ratio_adopted"]
    print()
    print("  GTR's prediction at index 422 was that a BAL makes the model EXCEED the")
    print("  observed line, so BALs should sit at ratio > 1.")
    if len(bb) and len(nb):
        print("    BALs above 1.0      : %d of %d (%.0f%%)" % ((bb > 1).sum(), len(bb), 100*(bb > 1).mean()))
        print("    non-BALs above 1.0  : %d of %d (%.0f%%)" % ((nb > 1).sum(), len(nb), 100*(nb > 1).mean()))
        print("    median ratio, BAL %.2f vs non-BAL %.2f" % (bb.median(), nb.median()))

    print()
    print("=" * 78)
    print("WOULD Mg II HAVE CHOSEN A DIFFERENT COMPONENT SET?")
    print("=" * 78)
    cols = ["model_ew_" + s for s in SETS]
    sub = ok.dropna(subset=cols + ["sdss_ew"]).copy()
    pref, agree = [], 0
    for _, r in sub.iterrows():
        d = {s: abs(r["model_ew_" + s] - r["sdss_ew"]) for s in SETS if np.isfinite(r["model_ew_" + s])}
        if not d:
            pref.append(None); continue
        best = min(d, key=d.get)
        pref.append(best)
    sub["mgii_prefers"] = pref
    sub = sub[sub["mgii_prefers"].notna()]
    # 'auto' is whichever set auto resolved to, so compare EW values not labels
    same = []
    for _, r in sub.iterrows():
        a = r["model_ew_adopted"]; p = r["model_ew_" + r["mgii_prefers"]]
        same.append(np.isfinite(a) and np.isfinite(p) and abs(a - p) < 1e-6)
    sub["mgii_agrees_with_adopted"] = same
    print("  objects where Mg II's preferred set gives the SAME reconstruction")
    print("  as the adopted one                        : %d of %d (%.0f%%)"
          % (sub["mgii_agrees_with_adopted"].sum(), len(sub),
             100 * sub["mgii_agrees_with_adopted"].mean()))
    diff = sub[~sub["mgii_agrees_with_adopted"]]
    print("  objects where it would differ             : %d" % len(diff))
    print("    of those, BAL / not-BAL / unclear / none: %s" % dict(diff["bal_status"].value_counts()))
    print()
    print("  Those are listed in final/mgii_would_differ.csv for the blueshift")
    print("  follow-up, which refits them under the Mg II-preferred set.")
    diff[["index", "folder", "manmask", "adopted_comps", "mgii_prefers", "sdss_ew",
          "model_ew_adopted", "ratio_adopted", "civ_blue", "civ_ew", "bal_status"]
         ].sort_values("index").to_csv(os.path.join(FINAL, "mgii_would_differ.csv"), index=False)

    print()
    print("=" * 78)
    print("WORST MISMATCHES (non-BAL, where a high ratio is NOT expected)")
    print("=" * 78)
    w = ok[ok["bal_status"].isin(["none", "not_bal"])].copy()
    w["dev"] = (w["ratio_adopted"] - 1).abs()
    for _, r in w.nlargest(12, "dev").iterrows():
        print("  %4d %-38s ratio %5.2f  sdss %6.1f  model %6.1f  manmask %d  %s"
              % (r["index"], str(r["folder"])[:38], r["ratio_adopted"], r["sdss_ew"],
                 r["model_ew_adopted"], r["manmask"], r["bal_status"]))

    ok.to_csv(os.path.join(FINAL, "mgii_sweep_annotated.csv"), index=False)
    print("\n-> final/mgii_sweep_annotated.csv, final/mgii_would_differ.csv")


if __name__ == "__main__":
    main()
