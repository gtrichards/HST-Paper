"""
Table 1 (v23): the master sample, one row per object in the v22 spine (474).

Rebuilds table1_master_sample_v22.tex (made by HST_SDSS_Master_Pipeline_v22.ipynb,
Cell 14) from CIV_measurements_v23.csv.  Rows, numbering (# = RA order, the
v22 row_num), designations, instruments and the SDSS / Sul / Flags columns
are unchanged.  What changed:
  * common names: the three overrides in final/name_overrides.csv
    (I Zw 1, PG 0804+761, PG 1211+143);
  * z: the redshift actually used in the fits (v23 table).  This equals the
    master catalogue's best_z except for the 20 objects revised during the
    pass (ica/redshift_overrides.csv) and 4th-decimal rounding;
  * the z superscript now names the source actually used (see footnote a);
  * new column Q = CIVgood_manmask (temporary: the final table will drop Q=0).

MERGED WITH TABLE 2 (GTR, 2026-10-30).  Table 2 was a one-column
deluxetable declaring seven columns, so it overlapped itself on the page.
Rather than widen it, its four measurement columns -- Spec., Blueshift, EW
and D_CIV -- now sit in this table, and the whole thing is rotated to
landscape.  make_table2.py is left working, so the split version can be
restored by reverting this file and re-adding \input{table2_civ_v23}.

Rows stay at 474, one per object in the spine, so the 125 objects with no
measurement (Q=0: 103, Q=1: 22) carry blank measurement columns.  That is
the point of a master table: it shows what was looked at, not only what
was measured.

Usage:   PYTHONPATH=~/Work/git python make_table1.py
Output:  Tables/table1_master_sample_v23.tex
         Tables/table1_master_sample_v23.csv  (same rows, with the full
                                               literature source string for z)
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from hst_sample import designation_to_coord

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Data")
OUT_DIR = os.path.join(HERE, "..", "Tables")
OVERRIDES = os.path.join(HERE, "..", "ica", "redshift_overrides.csv")

#: Trevor McCaffrey's C IV distance, as used in make_table2.py and Fig 6a.
CIVDIST_DIR = os.path.expanduser("~/Work/git/CIV_Distance/")
sys.path.insert(0, CIVDIST_DIR)
from CIVfunctions import CIV_distance


def norm(name):
    """Name key for matching: lower case, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


#: The legacy compilation (HST_CIV_Sulentic2007_HSLA2018.csv, column z_src)
#: records most individual literature redshifts as arXiv links.  Each was
#: looked up (arXiv + doi.org) and is cited here by its refs.bib key.
#: Zheng et al. (2019) supplies 16 objects and gets its own superscript (Z);
#: the rest are listed object by object under L.
ZHENG = "https://arxiv.org/pdf/1710.10703.pdf"
LIT_CITE = {
    "https://arxiv.org/pdf/1506.00635.pdf": r"\citealt{Koch2015}",
    "https://arxiv.org/pdf/1507.06317.pdf": r"\citealt{Richter2016}",
    "https://arxiv.org/pdf/2108.11958.pdf": r"\citealt{Fox2021}",
    "Muzahid+2018,MNRAS": r"\citealt{Muzahid2018}",   # COS-Weak: likely, not confirmed
    "DR7": r"SDSS DR7, \citealt{Schneider2010}",
    "Borisov, Voikhanskaya & Fabrika 1991": r"Borisov, Voikhanskaya \& Fabrika (1991)",
    "http://quasar.square7.ch/fqm/1411+442.html": r"web compilation, not a refereed source",
    "nan": r"no source recorded",
}


def z_code(row):
    """Superscript code for the source of the redshift used (footnote a)."""
    if row["z_revised"]:
        return "O"                       # revised during the pass
    src = row["z_source"]
    if src == "W":
        return ""                        # Wu & Shen (2022): no superscript
    if src in ("B", "K"):
        return src                       # RM compilations / Kaspi+2000
    # src == '*': the legacy compilation; name its own source
    leg = str(row["z_src_legacy"])
    if leg == "Sul":
        return "S"
    if leg == "NED":
        return "N"
    if leg == ZHENG:
        return "Z"
    return "L"                           # an individual literature value


def main():
    t = pd.read_csv(os.path.join(DATA, "CIV_measurements_v23.csv"))

    # master catalogue, matched by position, for the redshift source columns
    mc = pd.read_csv(os.path.join(DATA, "master_catalog_v22.csv"))
    c_t = SkyCoord([designation_to_coord(d) for d in t["designation"]])
    c_m = SkyCoord(mc["ra_deg"].values * u.deg, mc["dec_deg"].values * u.deg)
    idx, sep, _ = c_t.match_to_catalog_sky(c_m)
    assert sep.arcsec.max() < 1 and len(set(idx)) == len(t)
    for col in ["z_source", "z_src_legacy", "best_z"]:
        t[col] = mc[col].values[idx]
    t["name_v22"] = mc["common_name"].values[idx]   # before the name overrides

    # objects whose redshift was revised during the pass
    ov = pd.read_csv(OVERRIDES)
    ov_keys = set(norm(n) for n in ov["object"])
    t["z_revised"] = t["common_name"].map(norm).isin(ov_keys) | \
        t["name_v22"].map(norm).isin(ov_keys)
    assert t["z_revised"].sum() == len(ov), "an override did not match an object"
    # every other redshift must be the catalogue value (to 4 decimals)
    other = ~t["z_revised"]
    assert np.allclose(t.loc[other, "z"], t.loc[other, "best_z"], atol=5e-5)

    t["z_code"] = t.apply(z_code, axis=1)
    t["z_src_text"] = np.where(t["z_code"].isin(["L", "Z"]), t["z_src_legacy"], "")
    lit = t[t["z_code"] == "L"].sort_values("row_num")
    lit_note = "; ".join("\\#%d %s" % (r["row_num"], LIT_CITE[str(r["z_src_legacy"])])
                         for _, r in lit.iterrows())

    # ---- the measurement columns, from Table 2 ----
    meas = t["CIVgood_manmask"].isin([2, 3]).values
    data = np.array([t.loc[meas, "CIV_blue"].values,
                     np.log10(t.loc[meas, "CIV_EW"].values)]).T
    bestfit = np.load(CIVDIST_DIR + "data/bestfit.npy")
    d = CIV_distance(data, bestfit, path=CIVDIST_DIR)
    # CIV_distance silently skips points it cannot place; check none was lost
    assert len(d) == meas.sum(), "CIV_distance dropped a point"
    t["CIV_dist"] = np.nan
    t.loc[meas, "CIV_dist"] = d
    # Spec. is the spectrum the measurement came from, which is NOT the Inst.
    # column: that lists every instrument the archive holds, and the two differ
    # for 39 per cent of measured objects because of the single-visit policy.
    t["spectrum"] = np.where(meas, t["adopted"].astype(str),
                             np.where(t["CIVgood_manmask"] == 1, "IUE", ""))

    # ---- LaTeX ----
    def fmt_int(v):
        return "" if pd.isna(v) else "%d" % int(v)

    lines = []
    for _, r in t.sort_values("row_num").iterrows():
        des = r["designation"].replace("-", "$-$")
        z = "%.4f" % r["z"]
        z = "$%s^{\\rm %s}$" % (z, r["z_code"]) if r["z_code"] else "$%s$" % z
        if r["CIVgood_manmask"] in (2, 3):
            vals = "%d & %.1f & %.2f" % (round(r["CIV_blue"]), r["CIV_EW"],
                                         r["CIV_dist"])
        else:
            vals = " & & "
        lines.append("  %d & %s & %s & %s & %s & %s & %s & %s & %s & %s & %d \\\\"
                     % (r["row_num"], des, r["common_name"], z, r["instrument"],
                        r["spectrum"], fmt_int(r["sdss_flag"]),
                        fmt_int(r["sul_flag"]), fmt_int(r["cat_flags"]), vals,
                        r["CIVgood_manmask"]))

    n = t["z_code"].value_counts().to_dict()
    head = r"""%% Table generated by HST-Paper/paper_figures_v23/make_table1.py
%% from Data/CIV_measurements_v23.csv

\startlongtable
\begin{deluxetable*}{rllcccccrrrrc}
\rotate
\tabletypesize{\scriptsize}
\tablecaption{Master AGN Sample with HST/IUE Rest-Frame \civ\ Coverage and Adopted \civ\ Measurements\label{tab:master_sample}}
\tablewidth{0pt}
\tablehead{
  \colhead{\#} &
  \colhead{Designation} &
  \colhead{Common Name} &
  \colhead{$z$\tablenotemark{a}} &
  \colhead{Inst.\tablenotemark{b}} &
  \colhead{Spec.\tablenotemark{c}} &
  \colhead{SDSS\tablenotemark{d}} &
  \colhead{Sul\tablenotemark{e}} &
  \colhead{Flags\tablenotemark{f}} &
  \colhead{Blueshift\tablenotemark{g}} &
  \colhead{EW} &
  \colhead{$D_{\rm C\,IV}$\tablenotemark{h}} &
  \colhead{Q\tablenotemark{i}} \\
  \colhead{} & \colhead{} & \colhead{} & \colhead{} & \colhead{} &
  \colhead{} & \colhead{} & \colhead{} & \colhead{} &
  \colhead{(km\,s$^{-1}$)} & \colhead{(\AA)} & \colhead{} & \colhead{}
}
\startdata
"""
    foot = r"""\enddata
\tablenotetext{a}{Systemic redshift used in the fits. No superscript: \citet{WuShen2022} $Z_{\rm SYS}$ (%d objects); $^{\rm S}$: \citet{Sulentic2007} (%d); $^{\rm N}$: NED (%d); $^{\rm B}$: \citet{Barth2015} or \citet{Bentz2015} (%d); $^{\rm K}$: \citet{Kaspi2000} (%d); $^{\rm Z}$: \citet{Zheng2019} (%d); $^{\rm L}$: individual literature values (%d: %s); $^{\rm O}$: revised in this work, see \S2 (%d).}
\tablenotetext{b}{Every instrument the archive holds for this object.}
\tablenotetext{c}{Spectrum the measurement was actually made from, under the single-visit policy (\S2); SPLICE: two instruments joined. This differs from Inst.\ for 39\%% of measured objects. IUE objects are not yet measured.}
\tablenotetext{d}{SDSS spectroscopic coverage: 1\,=\,SDSS spectrum available; 2\,=\,SDSS spectrum covers rest-frame Mg\,\textsc{ii}\,$\lambda2798$ [$3800\,\text{\AA}/(1+z) \leq 2798\,\text{\AA}$].}
\tablenotetext{e}{In Sulentic+2007 (DOI: 10.1086/519916); 1\,=\,yes.}
\tablenotetext{f}{Bitmask of catalog memberships: Sul\,=\,1 (Sulentic+2007); K00\,=\,2 (Kaspi+2000); B15\,=\,4 (Barth+2015); D15\,=\,8 (Du+2015); Be15\,=\,16 (Bentz+2015); S16\,=\,32 (Shen+2016 SDSS-RM); M21\,=\,64 (Matthews+2021); NLS1\,=\,128 (Rakshit+2017); RK20\,=\,256 (Rakshit+2020 SDSS AGN catalog).}
\tablenotetext{g}{Positive values are blueshifted relative to systemic. Quantised at 69.03\,km\,s$^{-1}$ by construction.}
\tablenotetext{h}{\civ\ distance: position along the best-fit curve through the SDSS quasars in the blueshift--$\log$\,EW plane \citep{Rivera2020,Richards2021}, computed with the code of \citet{CIVDistance}.}
\tablenotetext{i}{\civ\ status: 3\,=\,measured; 2\,=\,measured, with a reservation; 1\,=\,IUE only, not yet measured; 0\,=\,excluded (see \S2).}
\end{deluxetable*}
""" % (n.get("", 0), n.get("S", 0), n.get("N", 0), n.get("B", 0),
       n.get("K", 0), n.get("Z", 0), n.get("L", 0), lit_note, n.get("O", 0))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "table1_master_sample_v23.tex")
    open(out, "w").write(head + "\n".join(lines) + "\n" + foot)
    t[["row_num", "designation", "common_name", "z", "z_code", "z_src_text",
       "instrument", "spectrum", "sdss_flag", "sul_flag", "cat_flags",
       "CIV_blue", "CIV_EW", "CIV_dist", "CIVgood_manmask"]
      ].to_csv(out.replace(".tex", ".csv"), index=False)
    print("wrote", out)
    print("z sources:", n)
    print("Q:", t["CIVgood_manmask"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
