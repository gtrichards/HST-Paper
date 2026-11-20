"""One review queue for the final lock-down, merging three lists:

  1. the sqrt(N) movers under the agreed rule (|d_quanta| big, or |dD_CIV| > 0.03)
     from pipeline_output/final/sqrtn_diff_graded.csv;
  2. the FOS products that were never rebuilt after the NREAD fix (their flux
     changed at the sqrt(N) rebuild), listed in flux_changed_fos.csv;
  3. whatever the Handbook-truncation + 1% guard rebuild moved
     (pipeline_output/final/hb_diff.csv, from run_hb_adoption.sh);
  4. whatever the one-pixel placement / echelle-splice fix (cbf3961) moved
     relative to the hb replay (pipeline_output/final/place_diff.csv);
  4b. whatever the fitter rulings (0bb4dac, d9c2701: chi^2 on the component grid,
     power-law continuum, interpolated lambda_half) moved relative to the place
     replay (pipeline_output/final/ica_diff.csv) -- note blueshifts are no longer
     quantised from this stage on, so "dq" is just the change in 69.03 km/s units;
  4c. whatever the final pass moved relative to the fitter replay
     (pipeline_output/final/final_diff.csv): c = 299792.458 km/s in blueshift()
     (e3c6791) and the 27 products re-staged under the current exposure rule
     (pipeline_output/final/restage_27.txt);
  5. the objects whose redshift override came from a C III]/Mg II anchor measured
     on a co-add that carried the placement shift (--flag-z), to be re-derived.

Every object not on the queue keeps its current adopted record.  Each row
carries index / v22 row / name_pub / folder, the adopted (BEST_) numbers, the
latest replayed numbers, D_CIV before and after, and WHY it is on the list.
Ranked by |dD_CIV| (adopted -> latest), then |d_blue|.

    python build_review_queue.py [--sqrtn CSV] [--hb CSV] [--stale CSV] [--out CSV]
"""
import argparse, glob, os, sys
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, "pipeline_output", "final")
FITS = os.path.join(HERE, "pipeline_output", "chosen_visit", "fits")
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper/paper_figures_v23")
from civ_distance_ext import civ_distance

D_CUT, Q_CUT = 0.03, 3.0          # the rule agreed for the sqrt(N) adoption: review if either trips


def dciv(blue, ew):
    try:
        if not (np.isfinite(blue) and np.isfinite(ew) and ew > 0): return np.nan
        return float(np.asarray(civ_distance(np.array([float(blue)]), np.array([float(np.log10(ew))])))[0])
    except Exception:
        return np.nan


def folder_of(index):
    d = glob.glob(os.path.join(FITS, "%03d_*" % index))
    return os.path.basename(d[0]) if d else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqrtn", default=os.path.join(FINAL, "sqrtn_diff_graded.csv"))
    ap.add_argument("--hb", default=os.path.join(FINAL, "hb_diff.csv"))
    ap.add_argument("--place", default=os.path.join(FINAL, "place_diff.csv"))
    ap.add_argument("--ica", default=os.path.join(FINAL, "ica_diff.csv"),
                    help="ledger of the fitter changes (chi^2, component-grid interpolation, power-law continuum, interpolated lambda_half)")
    ap.add_argument("--final", default=os.path.join(FINAL, "final_diff.csv"),
                    help="ledger of the final pass (c fix e3c6791 + the 27 re-staged products)")
    # the 7 redshift overrides derived from C III]/Mg II anchors on co-adds that carried the placement
    # shift (hstica-d8, 2026-11-17): UGC 12163, Z 291-51, ESO 113-45 (Fairall 9), Mrk 1502, LEDA 88835,
    # LEDA 52314, QSO B1626+5529.  The 13 from SDSS [O III] or published values are unaffected.
    ap.add_argument("--flag-z", default="28,32,60,70,79,101,106",
                    help="comma-separated indices whose UV-anchor redshift override must be re-derived")
    ap.add_argument("--stale", default=os.path.join(FINAL, "stale_fos_never_rebuilt_after_nread.csv"), help="flux_changed_fos.csv (stem,index); optional")
    ap.add_argument("--all", action="store_true",
                    help="review everything: list every adopted object, not only those a rule put on the queue (GTR, 2026-11-19)")
    ap.add_argument("--restaged", default=os.path.join(FINAL, "restage_27.txt"),
                    help="indices re-staged under the current exposure rule; flagged, and ranked directly after --first")
    ap.add_argument("--out", default=os.path.join(FINAL, "review_queue.csv"))
    ap.add_argument("--first", default="70", help="comma-separated indices to pin to the top of the queue (default: 70, the archetype)")
    a = ap.parse_args()

    Q = pd.read_csv(os.path.join(HERE, "pipeline_output", "work_queue.csv")).set_index("order")
    s = pd.read_csv(a.sqrtn).drop_duplicates("spectrum").set_index("order")
    if a.all:
        # objects adopted after the sqrt(N) ledger was graded are in the later ledgers only: take their adopted
        # (BEST_) numbers from the first ledger that has them; they have no sqrt(N) stage of their own
        for path in (a.hb, a.place, a.ica, a.final):
            if not os.path.exists(path): continue
            h0 = pd.read_csv(path).drop_duplicates("spectrum").set_index("order")
            for i in h0.index.difference(s.index):
                r0 = h0.loc[i]
                s.loc[i, ["spectrum", "blue_before", "ew_before", "veto_before", "blue_after", "ew_after", "veto_after", "d_quanta"]] = \
                    [r0.spectrum, r0.blue_before, r0.ew_before, r0.veto_before, r0.blue_before, r0.ew_before, r0.veto_before, 0.0]
    rows = {}

    def row(i):
        if i not in rows:
            q = Q.loc[i]
            rows[i] = dict(index=i, v22_row=int(q["row_num"]) if pd.notna(q["row_num"]) else -1,
                           name_pub=q["name_pub"], folder=folder_of(i), mm=int(s.loc[i, "mm"]) if (i in s.index and pd.notna(s.loc[i, "mm"])) else -1,
                           reasons=set())
        return rows[i]

    # 1. sqrt(N) rule
    for i, r in s.iterrows():
        big = (pd.notna(r.get("big")) and bool(r.get("big"))) or abs(r.d_quanta) >= Q_CUT
        dD = abs(r.dD) if pd.notna(r.dD) else 0.0
        if big or dD > D_CUT:
            x = row(i); x["reasons"].add("sqrtN: %s" % ", ".join(
                (["|dq| %.1f" % abs(r.d_quanta)] if big else []) + (["|dD| %.3f" % dD] if dD > D_CUT else [])))
        # keep the adopted and sqrtN numbers for every object we end up listing
        rows.setdefault(i, None)
    rows = {k: v for k, v in rows.items() if v is not None}

    # 2. stale FOS (never rebuilt after NREAD; flux changed at the sqrt(N) rebuild)
    if a.stale and os.path.exists(a.stale):
        st = pd.read_csv(a.stale)
        for i in st["index"].astype(int):
            if i in s.index:
                row(i)["reasons"].add("stale FOS: flux changed at rebuild (NREAD never applied)")

    # 3./4. successive rebuilds: each ledger's 'after' compared with the previous stage's 'after'
    stages = [("sqrtN", s)]
    for tag, path in (("Handbook/guard", a.hb), ("placement", a.place), ("fitter", a.ica), ("final", a.final)):
        if not os.path.exists(path): continue
        h = pd.read_csv(path).drop_duplicates("spectrum").set_index("order")
        prev = stages[-1][1]
        for i, r in h.iterrows():
            if i not in s.index or i not in prev.index: continue
            d_blue = r.blue_after - prev.loc[i, "blue_after"]
            D_p = dciv(prev.loc[i, "blue_after"], prev.loc[i, "ew_after"]); D_h = dciv(r.blue_after, r.ew_after)
            dq = abs(d_blue) / 69.03
            dD = (D_h - D_p) if (np.isfinite(D_h) and np.isfinite(D_p)) else np.nan
            if dq >= 1.0 or (np.isfinite(dD) and abs(dD) > D_CUT):
                row(i)["reasons"].add("%s: dq %+.1f, dD %+.3f" % (tag, d_blue / 69.03, dD))
        stages.append((tag, h))
    latest_tag, h = stages[-1]

    # 5. redshift overrides from UV anchors measured on shifted co-adds
    for i in [int(x) for x in a.flag_z.split(",") if x.strip()]:
        if i in s.index:
            row(i)["reasons"].add("z override from C III]/Mg II anchor on a shifted co-add: re-derive")

    # 6. products re-staged under the current exposure rule: the replayed masks/component set were chosen on other exposures
    restaged = [int(t) for t in open(a.restaged).read().split()] if (a.restaged and os.path.exists(a.restaged)) else []
    for i in restaged:
        if i in s.index:
            row(i)["reasons"].add("re-staged: refit by hand")
    if a.all:
        for i in s.index:
            row(i)

    out = []
    for i, x in rows.items():
        r = s.loc[i]
        latest = h.loc[i] if (h is not None and i in h.index) else r
        D0 = dciv(r.blue_before, r.ew_before); D1 = dciv(latest.blue_after, latest.ew_after)
        out.append(dict(index=i, v22_row=x["v22_row"], name_pub=x["name_pub"], folder=x["folder"], mm=x["mm"],
                        blue_adopted=r.blue_before, ew_adopted=r.ew_before, D_adopted=D0,
                        blue_sqrtn=r.blue_after, ew_sqrtn=r.ew_after,
                        blue_latest=latest.blue_after, ew_latest=latest.ew_after, D_latest=D1,
                        dD=D1 - D0, d_quanta=(latest.blue_after - r.blue_before) / 69.03,
                        veto_adopted=r.veto_before, veto_latest=latest.veto_after,
                        latest_label=latest_tag if (h is not None and i in h.index) else "sqrtN",
                        reasons="; ".join(sorted(x["reasons"]))))
    d = pd.DataFrame(out)
    d["rank_key"] = d.dD.abs().fillna(9) * 100 + d.d_quanta.abs().fillna(0)
    pin = [int(x) for x in a.first.split(",") if x.strip()]
    d["pin"] = [pin.index(i) if i in pin else (len(pin) if i in restaged else len(pin) + 1) for i in d["index"]]
    d = d.sort_values(["pin", "rank_key"], ascending=[True, False]).drop(columns=["rank_key", "pin"]).reset_index(drop=True)
    d.insert(0, "rank", np.arange(1, len(d) + 1))
    d.to_csv(a.out, index=False)

    cnt = lambda key: sum(key in r for r in d.reasons)
    print("REVIEW QUEUE: %d objects (sqrtN rule %d, stale FOS %d, Handbook/guard %d, placement %d, fitter %d, final %d, re-staged %d, z-override %d; %d adopted objects auto-keep; latest replay = %s)"
          % (len(d), cnt("sqrtN"), cnt("stale"), cnt("Handbook"), cnt("placement"), cnt("fitter"), cnt("final:"), cnt("re-staged"), cnt("z override"), len(s) - len(d), latest_tag))
    print("  rank idx  v22  %-24s mm  blue_adopted -> latest   EW_adopted -> latest   D_adopted -> latest   dD     reasons" % "name_pub")
    for _, r in d.iterrows():
        print("  %4d %3d %4d  %-24s %2d  %+8.1f -> %+8.1f   %6.1f -> %6.1f   %6.3f -> %6.3f  %+.3f  %s"
              % (r["rank"], r["index"], r.v22_row, str(r.name_pub)[:24], r.mm, r.blue_adopted, r.blue_latest,
                 r.ew_adopted, r.ew_latest, r.D_adopted, r.D_latest, r.dD, r.reasons))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
