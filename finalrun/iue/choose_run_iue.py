"""Pick ONE observing run per IUE object, the analogue of the chosen-visit
policy the HST pass used.

GTR: "There are lots of IUE spectra, but we don't need to use all of them (in
the same way that we are not using all the HST data)", and, asked whether one
epoch at S/N 19 beats all epochs at S/N 37 for 3C 120, "yes".

WHY.  3C 120 has 41 SWP exposures spanning swp1357 to swp50192, roughly 1978 to
1994.  Its C IV-region flux runs from 1.2e-14 to 3.35e-14 between runs -- a
factor of three on a variable Seyfert.  Co-adding all of them averages sixteen
years of a source that changed, which is exactly what the single-visit policy
exists to prevent for HST, and it buys depth we do not need: the best single run
reaches S/N 19 at C IV, which is better than most of the HST spectra in the
adopted sample.

HOW A RUN IS DEFINED.  IUE image numbers are chronological within a camera, so
exposures whose numbers are close were taken together.  A gap of more than
`GAP` image numbers starts a new run.  Runs are formed within a camera, never
across, so SWP and LWP exposures are never mixed.

HOW ONE IS CHOSEN.  The run whose inverse-variance co-add has the highest
median signal-to-noise in the rest-frame 1500-1600 A window, which is the
quantity every C IV measurement depends on.  Ties go to the run with more
exposures.

The choice is written to iue_chosen_run.csv and can be overridden per object in
iue_run_overrides.csv (columns: common_name, camera, swp_lo, swp_hi, reason),
the same discipline as spectrum_overrides.csv on the HST side -- so which
photons entered a measurement is a recorded decision rather than a default.
"""
import os, re, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = "/Users/gtr/Work/git/HST-Paper/Data/CIV_measurements_v23.csv"
OUT = os.path.join(HERE, "iue_chosen_run.csv")
OVERRIDES = os.path.join(HERE, "iue_run_overrides.csv")
GAP = 300
CIV = 1549.48


def safe(n):
    return "".join(c if (c.isalnum() or c in "+-._") else "_" for c in str(n))


def exposures(name, z):
    """Every MXLO exposure for one object, with its camera, number and C IV S/N."""
    d = os.path.join(HERE, "spectra", safe(name))
    out = []
    for p in sorted(set(glob.glob(os.path.join(d, "**", "*mxlo_vo.fits"), recursive=True))):
        m = re.search(r"(swp|lwp|lwr)(\d+)", os.path.basename(p), re.I)
        if not m:
            continue
        with fits.open(p) as h:
            dd = h[1].data
            w = np.asarray(dd["WAVE"][0], float)
            f = np.asarray(dd["FLUX"][0], float)
            s = np.asarray(dd["SIGMA"][0], float)
            q = np.asarray(dd["QUALITY"][0], int)
        g = (q == 0) & np.isfinite(f) & np.isfinite(s) & (s > 0)
        k = g & (w / (1 + z) >= 1500) & (w / (1 + z) <= 1600)
        out.append(dict(path=p, camera=m.group(1).upper(), num=int(m.group(2)),
                        sn=float(np.median(f[k] / s[k])) if k.sum() > 5 else np.nan,
                        flux=float(np.median(f[k])) if k.sum() > 5 else np.nan,
                        n_civ=int(k.sum())))
    return pd.DataFrame(out)


def runs(df):
    """Label contiguous image numbers within a camera as one run."""
    df = df.sort_values(["camera", "num"]).reset_index(drop=True)
    lab, cur = [], 0
    for i, r in df.iterrows():
        if i and (r["camera"] != df.loc[i - 1, "camera"]
                  or r["num"] - df.loc[i - 1, "num"] > GAP):
            cur += 1
        lab.append(cur)
    df["run"] = lab
    return df


def main():
    tab = pd.read_csv(TABLE)
    m1 = tab[tab["CIVgood_manmask"] == 1].sort_values("row_num")
    ov = pd.read_csv(OVERRIDES) if os.path.exists(OVERRIDES) else pd.DataFrame()
    rows = []
    for _, r in m1.iterrows():
        name = str(r["common_name"]); z = float(r["z"])
        e = exposures(name, z)
        if not len(e):
            print("  %-14s no exposures" % name, flush=True); continue
        e = runs(e)
        # inverse-variance co-add S/N of a run goes as the quadrature sum
        agg = (e.groupby("run")
                 .apply(lambda g: pd.Series(dict(
                     camera=g["camera"].iloc[0], n=len(g),
                     lo=int(g["num"].min()), hi=int(g["num"].max()),
                     sn=float(np.sqrt(np.nansum(np.clip(g["sn"], 0, None) ** 2))),
                     flux=float(np.nanmedian(g["flux"])))))
                 .reset_index())
        pick = None
        if len(ov):
            o = ov[ov["common_name"].astype(str) == name]
            if len(o):
                o = o.iloc[0]
                sel = agg[(agg["camera"] == o["camera"]) & (agg["lo"] >= int(o["swp_lo"]))
                          & (agg["hi"] <= int(o["swp_hi"]))]
                if len(sel):
                    pick = sel.sort_values(["sn", "n"], ascending=False).iloc[0]
                    src = "override"
        if pick is None:
            pick = agg.sort_values(["sn", "n"], ascending=False).iloc[0]
            src = "automatic"
        files = e[(e["camera"] == pick["camera"]) & (e["run"] == pick["run"])]["path"].tolist()
        rows.append(dict(row_num=int(r["row_num"]), common_name=name, z=z,
                         camera=pick["camera"], run_lo=int(pick["lo"]), run_hi=int(pick["hi"]),
                         n_exp=int(pick["n"]), run_sn=round(float(pick["sn"]), 2),
                         n_runs=int(len(agg)), n_exp_total=int(len(e)),
                         chosen_by=src, files=";".join(files)))
        print("  %-14s %s %d-%d | %d of %d exposures, %d runs | run S/N %5.1f (%s)"
              % (name, pick["camera"], pick["lo"], pick["hi"], pick["n"], len(e),
                 len(agg), pick["sn"], src), flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("\n%d objects -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
