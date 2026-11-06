"""Tabulate the overlap between the student Amogh's IUE list and our sample,
WITHOUT depending on how either side spells a name.

WHY THIS EXISTS.  Four successive name-based attempts gave four different
answers -- 25, 26, then 33 matched, and "31 not in the catalogue", which was
wrong.  Every failure was an alias, not an absence.  GTR caught one directly:
"why would you say that 'I Zw 1' isn't in the HST sample; it is" -- it is our
070_Mrk_1502_COMBINED.  No amount of string normalisation bridges "I Zw 1" and
"Mrk 1502".  GTR: "we need to have a clear tabulation of the overlap (or lack
thereof) that is not sensitive to name variations."

SIMBAD resolution was tried first and is the right tool, but this environment has
no network access (the sim-tap capabilities endpoint is unreachable), so the
overlap is established three ways instead, in order of objectivity, and every
object records WHICH way settled it:

  1. POSITION, from the designation itself.  31 of Amogh's 67 names encode
     coordinates (PG1211+143, Q_1302-102, X_1821+643, ...).  Those are parsed to
     RA/Dec and matched against the ra_deg/dec_deg already in
     master_catalog_v22.csv within MATCH_ARCSEC.  Both the B1950 and the J2000
     reading of the digits is tried, since PG/Q/PKS designations are B1950 and
     RXJ/HS are J2000, and whichever lands inside the radius is taken -- a match
     at a few arcsec in either frame is the same object.  This path uses no names
     at all beyond the digits.
  2. NAME, against every name column the catalogue carries (mast_target_name,
     common_name, rankine_name, wushen_name, name_legacy, name_simbad,
     designation).  Catalogues disagree about punctuation, not about objects, so
     this is normalised hard: case, spaces, underscores, hyphens and leading
     zeros removed, and MRK/MARK/MK folded to MKN.
  3. AN EXPLICIT ALIAS TABLE, below.  Only for identifications that need
     astronomical knowledge rather than string work -- "I Zw 1 is Mrk 1502".
     Each entry is hand-entered and auditable; nothing here is inferred by the
     code.  ADD TO IT RATHER THAN LOOSENING THE MATCHER.

Anything still unmatched is reported as UNRESOLVED, never as absent.  That is
the distinction the name-based attempts kept losing, and it is why this script
separates "we checked and it is not in the sample" from "we could not tell".

The catalogue row index is NOT the ledger folder index.  The folder number is the
work-queue `order`, and the two are different orderings of the same 474 objects;
conflating them produced wrong folder attributions on an intermediate pass, so
the mapping is made through work_queue.csv here.

    python crossmatch_amogh.py
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINALRUN = os.path.dirname(HERE)
AMOGH = "/Users/gtr/Dropbox/HST/Amogh/IUE good/IUE_good_EW_blueshift.csv"
CAT = "/Users/gtr/Work/git/HST-Paper/Data/master_catalog_v22.csv"
LEDGER = "/Users/gtr/Work/git/HST-Paper/Data/decisions.csv"
QUEUE = os.path.join(FINALRUN, "pipeline_output", "work_queue.csv")
FITS = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "fits")
IUE_FITS = os.path.join(HERE, "fits")
OUT = os.path.join(HERE, "amogh_overlap.csv")

#: A HHMM+DDM designation carries no seconds of RA, so it localises an object
#: only to about +-8 arcmin in RA and +-3 arcmin in Dec.  20 arcsec was tried
#: first and matched NOTHING even where the nearest catalogue object was plainly
#: the right one (PG 1049-005 came out at 419 arcsec, PG 1211+143 at 662).  600
#: arcsec is the honest radius for this kind of identification.  It is safe here
#: because the catalogue is 474 objects spread over the whole sky, so the chance
#: of a wrong object inside 10 arcmin is negligible -- and AMBIGUOUS matches,
#: where a second catalogue object also falls inside the radius, are flagged
#: rather than silently taken.
MATCH_ARCSEC = 600.0
QUANTUM = 69.03

#: Identifications that string matching cannot reach.  Each maps one of Amogh's
#: names to a name the v22 catalogue carries.  Hand-entered; the comment gives
#: why.  Extend this rather than weakening the normaliser.
ALIASES = {
    "I_ZW_1":        "Mrk 1502",      # I Zw 1 = Mrk 1502 = PG 0050+124; GTR confirmed it is in the sample
    "IIZW_136":      "Mrk 1513",      # II Zw 136 = PG 2130+099 = Mrk 1513
    "IIIZW_2":       "Mrk 1501",      # III Zw 2 = PG 0007+106 = Mrk 1501
    "MKN_590":       "NGC 863",       # Mrk 590 = NGC 863
    "Q_1229+204":    "Mrk 771",       # Q 1229+204 = PG 1229+204 = Mrk 771
    "ARK_120":       "AKN120",        # Arakelian 120, spelled AKN in the catalogue
    "FAIRALL_0009":  "ESO 113-45",    # Fairall 9 = ESO 113-45
    "ESO_141-_G_055": "ESO 141-55",   # punctuation only, but the digits differ enough to miss
    "HB89_2251-178": "MR 2251-178",   # [HB89] 2251-178 = MR 2251-178
    "PG1211+143":    "PB 3894",       # PG 1211+143 carries a PB designation here
    "PG_0947+396":   "LEDA 2149945",
    "PG_1001+054":   "LEDA 29208",
    "PG_1115+407":   "LEDA 34570",
    "3C_263":        "ICRF J113957.0+654749",
    "NGC_1566":      "NGC1566-NUC",   # the catalogue names the nucleus pointing
    "X_1821+643":    "H1821+643",
    "Q_1302-102":    "PG1302-102",
    "1001+291":      "PG1001+291",
    "PG_1552+08":    "PG1552+085",
    "PG0026+12":     "PG0026+129",
    "PG1435-0645":   "QSO-B1435-0645",
    "HS0624+690":    "QSO B0624+6907",
    "Q1512+37":      "B21512+37",
}


def norm(s):
    s = str(s).upper()
    for a in ("MARK", "MRK", "MK"):
        s = re.sub(r"\b%s\b" % a, "MKN", s)
        s = re.sub(r"^%s(?=[_ 0-9])" % a, "MKN", s)
    s = s.replace("MKNN", "MKN")
    s = re.sub(r"[^A-Z0-9+]", "", s)
    return re.sub(r"(?<=[A-Z])0+(?=\d)", "", s)


_DESIG = re.compile(r"(\d{2})(\d{2})([+-])(\d{2,4})(?!\d)")


def coords_from_designation(name):
    """[(ra_deg, dec_deg, frame)] readings of a HHMM+DDMM / HHMM+DD.M designation."""
    m = _DESIG.search(str(name).replace("_", "").replace(" ", ""))
    if not m:
        return []
    hh, mm, sign, dd = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    if hh > 23 or mm > 59:
        return []
    ra = (hh + mm / 60.0) * 15.0
    decs = []
    if len(dd) == 4:                       # DDMM -> degrees + arcmin
        decs.append(int(dd[:2]) + int(dd[2:]) / 60.0)
    elif len(dd) == 3:                     # DD.M -> degrees + tenths
        decs.append(int(dd[:2]) + int(dd[2]) / 10.0)
    else:                                  # DD
        decs.append(float(dd))
    out = []
    for dv in decs:
        dec = dv if sign == "+" else -dv
        out.append((ra, dec, "as written (J2000 reading)"))
        try:
            from astropy.coordinates import SkyCoord, FK4, FK5
            import astropy.units as u
            c = SkyCoord(ra * u.deg, dec * u.deg, frame=FK4(equinox="B1950"))
            j = c.transform_to(FK5(equinox="J2000"))
            out.append((float(j.ra.deg), float(j.dec.deg), "B1950 -> J2000"))
        except Exception:
            pass
    return out


def sep_arcsec(ra1, dec1, ra2, dec2):
    d2r = np.pi / 180.0
    dra = (ra1 - ra2) * np.cos(0.5 * (dec1 + dec2) * d2r)
    return np.hypot(dra, dec1 - dec2) * 3600.0


def main():
    am = pd.read_csv(AMOGH)
    names = [str(x).strip() for x in am.iloc[:, 0]]
    his = {str(r.iloc[0]).strip(): (float(r.iloc[2]), float(r.iloc[1]))
           for _, r in am.iterrows()}

    cat = pd.read_csv(CAT)
    q = pd.read_csv(QUEUE)
    led = pd.read_csv(LEDGER)
    led["order"] = led.folder.astype(str).str.extract(r"^(\d+)").astype(int)
    cra = cat["ra_deg"].values.astype(float)
    cdec = cat["dec_deg"].values.astype(float)
    ncol = [c for c in cat.columns if "name" in c.lower()]

    # catalogue name -> row, and work-queue name -> order
    cat_by_name = {}
    for i, r in cat.iterrows():
        for c in ncol:
            v = r[c]
            if isinstance(v, str) and v.strip():
                cat_by_name.setdefault(norm(v), i)
    q_by_name = {}
    for _, r in q.iterrows():
        for c in ("name_pub", "name_mast_key", "name_simbad", "designation", "sulentic"):
            v = r.get(c)
            if isinstance(v, str) and v.strip():
                q_by_name.setdefault(norm(v), int(r["order"]))
    # catalogue row -> work-queue order, via any shared name
    row_to_order = {}
    for i, r in cat.iterrows():
        for c in ncol:
            v = r[c]
            if isinstance(v, str) and norm(v) in q_by_name:
                row_to_order[i] = q_by_name[norm(v)]
                break

    adopted = {}
    for rec in glob.glob(os.path.join(FITS, "*", "records", "BEST_*.json")):
        i = int(re.search(r"fits/(\d+)_", rec).group(1))
        d = json.load(open(rec))
        adopted[i] = (d["result"]["civ_blue"], d["result"]["civ_ew"], "HST")
    iue_adopted = {}
    for rec in glob.glob(os.path.join(IUE_FITS, "*", "records", "BEST_*.json")):
        d = json.load(open(rec))
        iue_adopted[norm(d.get("name", "").replace("_IUE", ""))] = \
            (d["result"]["civ_blue"], d["result"]["civ_ew"], "IUE")

    rows = []
    for n in names:
        rec = dict(amogh_name=n, his_blueshift=his[n][0], his_ew=his[n][1],
                   matched_by="", sep_arcsec=np.nan, cat_row=np.nan, order=np.nan,
                   folder="", our_manmask="", our_source="",
                   our_blueshift=np.nan, our_ew=np.nan, d_quanta=np.nan,
                   status="UNRESOLVED -- could not identify")
        row = None
        # 1. position from the designation -- take the closest reading over both frames
        best = None
        for ra, dec, frame in coords_from_designation(n):
            s = sep_arcsec(ra, dec, cra, cdec)
            j = int(np.nanargmin(s))
            if s[j] <= MATCH_ARCSEC and (best is None or s[j] < best[1]):
                nearby = int((s <= MATCH_ARCSEC).sum())
                best = (j, float(s[j]), frame, nearby)
        if best is not None:
            row = best[0]
            rec["matched_by"] = "position (%s)" % best[2]
            rec["sep_arcsec"] = round(best[1], 1)
            if best[3] > 1:
                rec["matched_by"] += " AMBIGUOUS: %d objects within radius" % best[3]
        # 2. name against every catalogue name column
        if row is None and norm(n) in cat_by_name:
            row, rec["matched_by"] = cat_by_name[norm(n)], "name"
        # 3. explicit alias table
        if row is None and n in ALIASES and norm(ALIASES[n]) in cat_by_name:
            row, rec["matched_by"] = cat_by_name[norm(ALIASES[n])], "alias table (%s)" % ALIASES[n]
        # 3b. the IUE-only set, which is keyed by its own folder names
        if row is None and norm(n) in iue_adopted:
            b, e, src = iue_adopted[norm(n)]
            rec.update(matched_by="name (IUE set)", our_blueshift=b, our_ew=e,
                       our_source=src, d_quanta=round((b - his[n][0]) / QUANTUM, 2),
                       status="IN SAMPLE, measured from IUE (our 22-object set)")
            rows.append(rec)
            continue

        if row is None:
            rows.append(rec)
            continue

        rec["cat_row"] = row
        o = row_to_order.get(row)
        if o is not None:
            rec["order"] = o
            lr = led[led.order == o]
            if len(lr):
                rec["folder"] = lr.iloc[0]["folder"]
                rec["our_manmask"] = lr.iloc[0]["manmask"]
            if o in adopted:
                b, e, src = adopted[o]
                rec.update(our_blueshift=b, our_ew=e, our_source=src,
                           d_quanta=round((b - his[n][0]) / QUANTUM, 2),
                           status="IN SAMPLE, measured from HST")
            else:
                # No HST record.  These are the v22 manmask-1 objects -- IUE-only
                # -- so look for one of OUR IUE measurements before concluding
                # there is nothing.  Checking the IUE set only when the catalogue
                # match failed (as an earlier version did) miscounted the three
                # objects that are in both Amogh's list and our 22-object IUE set.
                ik = None
                for k in iue_adopted:
                    fol = norm(str(rec.get("folder", "")))
                    if k and (k in fol or k in norm(n)):
                        ik = k
                        break
                if ik:
                    b, e, src = iue_adopted[ik]
                    rec.update(our_blueshift=b, our_ew=e, our_source=src,
                               d_quanta=round((b - his[n][0]) / QUANTUM, 2),
                               status="IN SAMPLE, measured from IUE (our 22-object set)")
                else:
                    rec["status"] = "IN SAMPLE, no adopted measurement (manmask 0)"
        else:
            rec["status"] = "IN SAMPLE (catalogue row %d), queue order unresolved" % row
        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    print("AMOGH'S IUE LIST vs OUR SAMPLE -- identified by POSITION where the designation")
    print("allows it, then by catalogue name, then by an explicit alias table.")
    print("Nothing is called absent unless it was actually checked.")
    print()
    for st in sorted(d.status.unique()):
        print("   %-52s %3d" % (st, int((d.status == st).sum())))
    print("   %-52s %3d" % ("TOTAL in Amogh's table", len(d)))
    print()
    print("how each identification was made:")
    for mb in sorted(x for x in d.matched_by.unique() if x):
        print("   %-52s %3d" % (mb, int((d.matched_by == mb).sum())))
    if (d.matched_by == "").any():
        print("   %-52s %3d" % ("(not identified)", int((d.matched_by == "").sum())))

    m = d[d.d_quanta.notna()]
    if len(m):
        print()
        print("THE CROSS-CHECK -- %d objects measured on both sides:" % len(m))
        print("   blueshift, ours minus his: median %+.2f quanta (%+.0f km/s), sd %.1f quanta"
              % (m.d_quanta.median(), m.d_quanta.median() * QUANTUM, m.d_quanta.std()))
        print("   within 2 quanta: %d of %d;  within 5: %d"
              % (int((m.d_quanta.abs() < 2).sum()), len(m), int((m.d_quanta.abs() < 5).sum())))
        print("   EW ratio ours/his: median %.2f" % float((m.our_ew / m.his_ew).median()))
        print()
        print("   largest disagreements:")
        for _, r in m.reindex(m.d_quanta.abs().sort_values(ascending=False).index).head(4).iterrows():
            print("      %-18s %-26s his %+8.1f  ours %+8.1f  %+6.1f quanta"
                  % (r.amogh_name, str(r.folder)[:26], r.his_blueshift, r.our_blueshift, r.d_quanta))
    un = d[d.matched_by == ""]
    if len(un):
        print()
        print("NOT IDENTIFIED (%d) -- status unknown, NOT evidence of absence:" % len(un))
        for _, r in un.iterrows():
            print("   %s" % r.amogh_name)
    print()
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
