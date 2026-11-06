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

from hst_sample import designation_to_coord, add_logL1450

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Data")
OUT_DIR = os.path.join(HERE, "..", "Tables")
OVERRIDES = os.path.join(HERE, "..", "ica", "redshift_overrides.csv")

#: Trevor McCaffrey's C IV distance, as used in make_table2.py and Fig 6a.
CIVDIST_DIR = os.path.expanduser("~/Work/git/CIV_Distance/")
sys.path.insert(0, CIVDIST_DIR)
from CIVfunctions import CIV_distance
from civ_distance_ext import civ_distance, EXTEND_CURVE


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
    # D_CIV from civ_distance_ext: Trevor's curve, extended past its high-EW
    # end when EXTEND_CURVE is True (16 objects get D < 0 instead of all 0).
    # CIV_dist_trevor keeps Trevor's unextended value for comparison.
    t["CIV_dist"] = np.nan
    t.loc[meas, "CIV_dist"] = civ_distance(data[:, 0], data[:, 1])
    bestfit = np.load(CIVDIST_DIR + "data/bestfit.npy")
    d_trevor = CIV_distance(data, bestfit, path=CIVDIST_DIR)
    assert len(d_trevor) == meas.sum(), "CIV_distance dropped a point"
    t["CIV_dist_trevor"] = np.nan
    t.loc[meas, "CIV_dist_trevor"] = d_trevor
    # log L_nu(1450 A), erg/s/Hz -- the same calculation as Figure 1's y axis
    # (hst_sample.add_logL1450), so the table and the figure cannot disagree.
    # F1450_est = 1 marks the four objects whose 1450 A continuum is
    # extrapolated (extract_f1450.py); tested on the 345 measured objects that
    # extrapolation is good to 0.02-0.03 dex.  Measured objects only.
    t = add_logL1450(t, DATA)
    t.loc[~meas, "logLnu1450"] = np.nan
    t["L_est"] = meas & (t["F1450_est"].fillna(0).values == 1)
    assert t.loc[meas, "logLnu1450"].notna().all(), "a measured object has no L1450"

    # Spec. is the spectrum the measurement came from, which is NOT the Inst.
    # column: that lists every instrument the archive holds, and the two differ
    # for 39 per cent of measured objects because of the single-visit policy.
    # Before the IUE addendum was merged (2026-11-06) an unmeasured manmask-1
    # object was labelled "IUE" here to show that IUE data existed but had not
    # been used.  manmask 1 is now empty -- every one of those 22 objects was
    # measured or graded 0 -- and `adopted` carries "IUE" for the ten measured
    # from it, so the fallback is gone and IUE is treated exactly as COS, FOS
    # and STIS are (GTR, 2026-11-06: "Table 1 should treat IUE data the same
    # way that HST data are treated").
    t["spectrum"] = np.where(meas, t["adopted"].astype(str), "")

    # Inst. carries the adopted spectrum by BOLDING it in the list, instead of
    # a separate Spec. column (GTR, 2026-10-30: "we could save some space by
    # boldfacing the appropriate instrument in the Inst column").  Where the
    # adopted spectrum is not one of the listed instruments it is appended in
    # parentheses rather than dropped, and reported below -- that mismatch is
    # a real defect in the instrument listing, not a formatting problem.
    # Instruments as single letters to save width (GTR 2026-11-03, option 2):
    # C = COS, F = FOS, S = STIS, I = IUE (any camera), in the archive's order.
    # The adopted spectrum is bold; a splice (two instruments joined) has both
    # bold; an adopted spectrum missing from the listing is added in brackets.
    # The camera names are what the instrument column lists; "IUE" is what the
    # adopted column says, and both must map to I or the adopted letter cannot
    # be matched against the list and nothing is bolded.
    LETTER = {"COS": "C", "FOS": "F", "STIS": "S",
              "SWP": "I", "LWP": "I", "LWR": "I", "IUE": "I"}

    def inst_cell(r):
        toks = []
        for x in str(r["instrument"]).split("/"):
            if LETTER.get(x, x) not in toks:
                toks.append(LETTER.get(x, x))
        spec = str(r["spectrum"])
        bold = set(toks) if spec == "SPLICE" else {LETTER.get(spec, "")}
        cell = "".join(r"\textbf{%s}" % x if x in bold and spec else x
                       for x in toks)
        if spec not in ("", "SPLICE") and LETTER.get(spec) not in toks:
            cell += r" (\textbf{%s})" % LETTER.get(spec, spec)
        return cell

    t["inst_cell"] = t.apply(inst_cell, axis=1)
    # Compare LETTERS, not raw names.  The instrument column lists IUE by its
    # cameras (LWP/LWR/SWP) while the adopted column says "IUE", so a raw string
    # comparison called all ten IUE objects anomalies; they are not, their
    # camera name simply is not spelled "IUE".  LETTER maps both to I.
    def _listed(r):
        toks = [LETTER.get(x, x) for x in str(r["instrument"]).split("/")]
        return LETTER.get(str(r["spectrum"]), str(r["spectrum"])) in toks
    odd = t[meas & ~t.apply(_listed, axis=1)]
    if len(odd):
        print("NOTE: %d measured object(s) whose adopted spectrum is not in the "
              "instrument list; shown in parentheses:" % len(odd))
        for _, r in odd.iterrows():
            print("   row %-4s %-28s instrument=%-14s adopted=%s"
                  % (r["row_num"], str(r["common_name"])[:28], r["instrument"],
                     r["spectrum"]))

    # ---- LaTeX ----
    def fmt_int(v):
        return "" if pd.isna(v) else "%d" % int(v)

    # The PRINTED Flags bitmask puts Sulentic+2007 membership in bit 1 (GTR
    # 2026-11-03: "that way they are easy to recognize as objects with
    # odd-numbered bitmasks"), replacing the separate Sul column, so every
    # catalogue bit of cat_flags moves up one place (cat_flags << 1).  Only the
    # printed column changes: cat_flags itself, which hst_sample decodes for the
    # figures' RM groups, keeps the sample-construction bits.
    t["flags_out"] = 2 * t["cat_flags"].fillna(0).astype(int) + \
        (t["sul_flag"].fillna(0) == 1).astype(int)      # shift up one bit, Sul in bit 1

    # FWHM(Hbeta) and R(Fe II): the values plotted in Figure 2 (make_fig2.py:
    # Rakshit+20 where available, else Sulentic+07)
    opt = pd.read_csv(os.path.join(DATA, "HST_optical_EV1_v23.csv"))
    opt = opt[opt["opt_source"].fillna("") != ""].set_index("row_num")

    def opt_cells(row_num):
        if row_num not in opt.index:
            return " & "
        o = opt.loc[row_num]
        return "%d%s & %s%.2f" % (round(o["Hb_fwhm"]),
                                  r"$^{\rm S}$" if o["opt_source"] == "Sulentic07" else "",
                                  "$<$" if o["RFeII_upper"] else "", o["RFeII"])

    lines = []
    for _, r in t.sort_values("row_num").iterrows():
        des = r["designation"].replace("-", "$-$")
        # FIVE decimals, not four.  This column is the redshift USED IN THE FIT,
        # an input to our velocities rather than a measurement we are claiming,
        # so it needs enough digits to reproduce them.  Rounding to 4 dp costs
        # up to 14.6 km/s at z = 0.03 (peak-to-peak ~29 km/s, 0.42 of the
        # 69.03 km/s quantisation); 5 dp costs 1.5 km/s.  GTR, 2026-10-30:
        # "a 30 km/s error is half a quantum".  340 of the 474 source values
        # carry 5 or more decimals; the rest gain a trailing zero, which is
        # honest here because the value shown is the one that was used.
        z = "%.5f" % r["z"]
        z = "$%s^{\\rm %s}$" % (z, r["z_code"]) if r["z_code"] else "$%s$" % z
        if r["CIVgood_manmask"] in (2, 3):
            vals = "%d & %.1f & %.2f & %.2f%s" % (
                round(r["CIV_blue"]), r["CIV_EW"], r["CIV_dist"], r["logLnu1450"],
                r"$^{\dagger}$" if r["L_est"] else "")
        else:
            vals = " & & & "
        flags = "%d" % r["flags_out"] if r["flags_out"] else ""
        lines.append("  %d & %s & %s & %s & %s & %s & %s & %s & %d & %s \\\\"
                     % (r["row_num"], des, r["common_name"], z, r["inst_cell"],
                        fmt_int(r["sdss_flag"]), flags, vals,
                        r["CIVgood_manmask"], opt_cells(r["row_num"])))

    n = t["z_code"].value_counts().to_dict()
    head = r"""%% Table generated by HST-Paper/paper_figures_v23/make_table1.py
%% from Data/CIV_measurements_v23.csv

\begin{longrotatetable}
\startlongtable
\setlength{\tabcolsep}{3.5pt}   % 14 columns: tighter than the 6pt default to fit the page
\begin{deluxetable*}{rllcccrrrrrcrr}
\tabletypesize{\scriptsize}
\tablecaption{Master AGN Sample with HST/IUE Rest-Frame \civ\ Coverage and Adopted \civ\ Measurements\label{tab:master_sample}}
\tablewidth{0pt}
\tablehead{
  \colhead{\#} &
  \colhead{Designation} &
  \colhead{Common Name} &
  \colhead{$z$\tablenotemark{a}} &
  \colhead{Inst.\tablenotemark{b}} &
  \colhead{SDSS\tablenotemark{c}} &
  \colhead{Flags\tablenotemark{d}} &
  \colhead{Blueshift\tablenotemark{e}} &
  \colhead{EW} &
  \colhead{$D_{\rm C\,IV}$\tablenotemark{f}} &
  \colhead{$\log L_\nu(1450\,\mathrm{\AA})$\tablenotemark{g}} &
  \colhead{Q\tablenotemark{h}} &
  \colhead{FWHM(H$\beta$)\tablenotemark{i}} &
  \colhead{R(Fe\,{\sc ii})\tablenotemark{i}} \\
  \colhead{} & \colhead{} & \colhead{} & \colhead{} & \colhead{} &
  \colhead{} & \colhead{} &
  \colhead{(km\,s$^{-1}$)} & \colhead{(\AA)} & \colhead{} &
  \colhead{(erg\,s$^{-1}$\,Hz$^{-1}$)} & \colhead{} & \colhead{(km\,s$^{-1}$)} & \colhead{}
}
\startdata
"""
    # Footnote d: cat_flags carries the v22 notebook's CATALOG_BITS (K00 = 1,
    # B15 = 2, D15 = 4, Be15 = 8, S16 = 16, M21 = 32, R20 = 64, NLS1 = 128,
    # RK20 = 256), checked against known objects (PG 0804+761 = 9, Kaspi +
    # Bentz; NGC 5548 = 8, Bentz).  The PRINTED bitmask shifts all of them up
    # one place and puts Sulentic+2007 in bit 1 (see flags_out above).
    # footnote f: say so when the curve is extended (civ_distance_ext.py)
    EXT_NOTE = (r" Above $\log$\,EW\,=\,2.50 (EW\,=\,316\,\AA), where the curve ends, it is"
                r" continued vertically at its end point; objects beyond the end therefore have"
                r" $D_{\rm C\,IV}<0$ (16 objects).") if EXTEND_CURVE else ""
    foot = r"""\enddata
\tablenotetext{a}{Systemic redshift used in the fits, quoted to five decimals so that the \civ\ velocities are reproducible to $\sim$1.5\,km\,s$^{-1}$, well inside the 69\,km\,s$^{-1}$ quantisation; four decimals would cost up to 15\,km\,s$^{-1}$ at the low-redshift end. No superscript: \citet{WuShen2022} $Z_{\rm SYS}$ (%d objects); $^{\rm S}$: \citet{Sulentic2007} (%d); $^{\rm N}$: NED (%d); $^{\rm B}$: \citet{Barth2015} or \citet{Bentz2015} (%d); $^{\rm K}$: \citet{Kaspi2000} (%d); $^{\rm Z}$: \citet{Zheng2019} (%d); $^{\rm L}$: individual literature values (%d: %s); $^{\rm O}$: revised in this work, see \S2 (%d).}
\tablenotetext{b}{Every instrument the archive holds for this object: C\,=\,COS, F\,=\,FOS, S\,=\,STIS, I\,=\,IUE. \textbf{Bold} marks the one the measurement was actually made from, under the single-visit policy (\S2); 135 objects have more than one, and the measured spectrum is not the only one available for 39\%% of them. Two bold letters denote two instruments joined (\S2). Where the adopted spectrum is not among those listed it is given in parentheses (\#452, an instrument-listing error). IUE data are treated exactly as the HST data are: \textbf{I} marks the ten objects measured from IUE, and an unbolded I marks the other objects for which IUE data exist but the measurement came from elsewhere or could not be made.}
\tablenotetext{c}{SDSS spectroscopic coverage: 1\,=\,SDSS spectrum available; 2\,=\,SDSS spectrum covers rest-frame Mg\,\textsc{ii}\,$\lambda2798$ [$3800\,\text{\AA}/(1+z) \leq 2798\,\text{\AA}$].}
\tablenotetext{d}{Bitmask of catalog memberships Sul\,=\,1 (\citealt{Sulentic2007}; odd values are Sulentic objects); K00\,=\,2 (Kaspi+2000); B15\,=\,4 (Barth+2015); D15\,=\,8 (Du+2015); Be15\,=\,16 (Bentz \& Katz 2015); S16\,=\,32 (Shen+2016 SDSS-RM); NLS1\,=\,256 (Rakshit+2017); RK20\,=\,512 (Rakshit+2020 SDSS AGN catalog). Bits 64 (Matthews+2021) and 128 (Rankine+2020) are defined but set for no object here.}
\tablenotetext{e}{Positive values are blueshifted relative to systemic. Quantised at 69.03\,km\,s$^{-1}$ by construction.}
\tablenotetext{f}{\civ\ distance: position along the best-fit curve through the SDSS quasars in the blueshift--$\log$\,EW plane \citep{Rivera2020,Richards2021}, computed with the code of \citet{CIVDistance}.%s}
\tablenotetext{g}{Specific luminosity at rest-frame 1450\,\AA, from the calibrated continuum under the adopted fit, as plotted in Figure~1. $^{\dagger}$: the continuum calibration does not reach 1450\,\AA\ and is extrapolated along the object's own power law (4 objects); tested on the measured objects, this is accurate to 0.02--0.03\,dex.}
\tablenotetext{h}{\civ\ status: 3\,=\,measured; 2\,=\,measured, with a reservation; 1\,=\,IUE only, not yet measured; 0\,=\,excluded (see \S2).}
\tablenotetext{i}{Broad H$\beta$ FWHM and R(Fe\,{\sc ii})\,=\,EW(Fe\,{\sc ii}\,$\lambda\lambda$4434--4684)/EW(H$\beta$), as plotted in Figure~2: from \citet{Rakshit+2020} where available, otherwise from \citet{Sulentic2007} ($^{\rm S}$); $<$: upper limit.}
\end{deluxetable*}
\end{longrotatetable}
""" % ((n.get("", 0), n.get("S", 0), n.get("N", 0), n.get("B", 0),
       n.get("K", 0), n.get("Z", 0), n.get("L", 0), lit_note, n.get("O", 0)) + (EXT_NOTE,))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "table1_master_sample_v23.tex")
    open(out, "w").write(head + "\n".join(lines) + "\n" + foot)
    t[["row_num", "designation", "common_name", "z", "z_code", "z_src_text",
       "instrument", "spectrum", "sdss_flag", "sul_flag", "cat_flags",
       "flags_out", "CIV_blue", "CIV_EW", "CIV_dist", "CIV_dist_trevor", "logLnu1450", "L_est", "CIVgood_manmask"]
      ].to_csv(out.replace(".tex", ".csv"), index=False)
    print("wrote", out)
    print("z sources:", n)
    print("Q:", t["CIVgood_manmask"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
