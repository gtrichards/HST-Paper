"""Does each of the 474 objects actually have C IV coverage?  Measured, not assumed.

This exists because the sample needs a reproducible definition.  GTR, 2026-11-06:
"we need a final query that produces *exactly* the objects in Table 1 -- after we
drop the rows that are manmask=0.  That is the query should recognize when there
is no CIV coverage.  Noting that we might want to keep the manmask=0 rows where
there is indeed CIV coverage, but we have deemed it unusable."

So the sample the query must reproduce is

    manmask 2 or 3                    the adopted measurements
    manmask 0 WITH C IV coverage      looked at, judged unusable
    -- excluding --
    manmask 0 with NO C IV coverage   nothing was ever there to judge

and that split has to rest on a coverage test that is actually right.

WHY NOT work_queue.civ_frac_best, THE OBVIOUS CANDIDATE.  It is wrong for at
least twelve objects.  Ten are the IUE additions, which it cannot see because it
was computed from HST spectra alone.  TWO ARE HST OBJECTS IT SIMPLY HAS WRONG:
order 164 (LEDA 138607) and order 220 (2MASS J10215235+4645158) both read
civ_frac_best = 0 while their adopted products carry 272 and 240 usable C IV
pixels and both are measured at manmask 2.  A stale column cannot define a
published sample.

WHAT "COVERED" MEANS HERE, and each clause was learned from a specific failure:

    unmasked              Bad Pixel Mask == 0
    finite flux           not NaN
    POSITIVE ERROR        err > 0

The error clause is the one that does the work.  A negative error is what a
negative continuum produces -- coadd divides flux and errors by a continuum that
cont_filtered interpolated across the C IV exclusion window from an anchor below
zero -- and it is also what COS G140L's dead red edge produces, where the
extracted array runs on past 2400 A with ZERO GROSS COUNTS.  Orders 307 and 334
have 140 unmasked finite-flux pixels each at C IV and no photons behind any of
them; MKN 6 and 3C 227 have the same on the IUE side.  Requiring err > 0 rejects
all four, which is the right answer, and it is the same criterion the fitter
itself applies, so this measures what the fitter would actually have to work
with.

Products are searched in the order chosen_visit, IUE, v23: the chosen visit is
what a measurement was made from, the v23 tree holds every object including those
never staged for fitting, and an object is COVERED if ANY of its products has
usable C IV pixels.  That is deliberately generous -- the question is whether the
data exist, not whether the adopted visit happened to include them.

    python civ_coverage_audit.py            # report
    python civ_coverage_audit.py --csv      # write pipeline_output/final/civ_coverage.csv
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")

from astropy.table import Table

QUEUE = os.path.join(HERE, "pipeline_output", "work_queue.csv")
LEDGER = "/Users/gtr/Work/git/HST-Paper/Data/decisions.csv"
TREES = [os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin"),
         os.path.join(HERE, "iue", "rebin"),
         os.path.join(HERE, "data_v23", "RebinnedSpec_v23")]
OUT = os.path.join(HERE, "pipeline_output", "final", "civ_coverage.csv")

CIV = (1500.0, 1600.0)
#: A line needs more than a handful of pixels to be called covered.  On the
#: 69.03 km/s lattice the 100 A window is about 280 pixels, so 10 is a very low
#: bar -- it is there to reject edge slivers, not to make a quality judgement.
MIN_PX = 10


def usable_civ(path):
    """(usable C IV pixels, rest range) for one product, or (0, None)."""
    try:
        t = Table.read(path)
    except Exception:
        return 0, None
    cols = t.colnames
    w = np.asarray(t[cols[0]], float)
    f = np.asarray(t[cols[1]], float)
    e = (np.asarray(t["Coadded Flux Errors"], float)
         if "Coadded Flux Errors" in cols else np.full(len(w), np.nan))
    m = (np.asarray(t["Bad Pixel Mask"], float)
         if "Bad Pixel Mask" in cols else np.zeros(len(w)))
    g = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    civ = g & (w >= CIV[0]) & (w <= CIV[1])
    rng = (float(w[g].min()), float(w[g].max())) if g.any() else None
    return int(civ.sum()), rng


def products_for(stems, name_keys):
    """Every rebinned product that could belong to this object."""
    out = []
    pats = [s.strip() for s in str(stems).split(";") if s.strip()]
    pats += ["%s_IUE" % k for k in name_keys if k]
    for tree in TREES:
        for pat in pats:
            # glob.escape is essential: many names are bracketed, e.g.
            # "[VV2006] J123734.4+444731", and an unescaped "[VV2006]" is a
            # CHARACTER CLASS matching one of V,2,0,6 -- so the pattern matched
            # nothing and thirteen measured objects were reported as having no
            # C IV coverage when the real problem was that no product was found.
            for p in glob.glob(os.path.join(glob.escape(tree),
                                            "%s*.fits" % glob.escape(pat))):
                if p not in out:
                    out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()

    q = pd.read_csv(QUEUE)
    led = pd.read_csv(LEDGER)
    led["order"] = led.folder.astype(str).str.extract(r"^(\d+)").astype(int)
    led["mm"] = pd.to_numeric(led.manmask, errors="coerce")
    mm = dict(zip(led.order, led.mm))

    rows = []
    for _, r in q.iterrows():
        o = int(r["order"])
        keys = [str(r.get("name_mast_key") or ""), str(r.get("name_pub") or "")]
        best_px, best_p, rng = 0, "", None
        for p in products_for(r.get("stems"), keys):
            n, rr = usable_civ(p)
            if n > best_px:
                best_px, best_p, rng = n, os.path.basename(p), rr
        rows.append(dict(order=o, row_num=int(r["row_num"]),
                         name_pub=str(r.get("name_pub") or "").strip(),
                         name_mast_key=str(r.get("name_mast_key") or "").strip(),
                         z=float(r["z"]) if pd.notna(r["z"]) else np.nan,
                         manmask=mm.get(o, np.nan),
                         civ_px=best_px, covered=bool(best_px >= MIN_PX),
                         product=best_p,
                         rest_lo=rng[0] if rng else np.nan,
                         rest_hi=rng[1] if rng else np.nan,
                         queue_civ_frac=pd.to_numeric(r.get("civ_frac_best"),
                                                      errors="coerce")))
    d = pd.DataFrame(rows)
    d["in_sample"] = d.manmask.isin([2, 3]) | ((d.manmask == 0) & d.covered)

    print("C IV COVERAGE, measured from the rebinned products (%d objects)" % len(d))
    print("  covered = at least %d pixels in 1500-1600 A that are unmasked, finite"
          % MIN_PX)
    print("            and have a POSITIVE error -- the fitter's own criterion")
    print()
    print("  %-10s %9s %9s %9s" % ("manmask", "covered", "not", "total"))
    for g in sorted(x for x in d.manmask.dropna().unique()):
        s = d[d.manmask == g]
        print("  %-10d %9d %9d %9d" % (g, s.covered.sum(), (~s.covered).sum(), len(s)))
    print("  %-10s %9d %9d %9d" % ("ALL", d.covered.sum(), (~d.covered).sum(), len(d)))
    print()
    print("  *** THE SAMPLE THE QUERY MUST REPRODUCE: %d objects ***" % d.in_sample.sum())
    print("        manmask 2 or 3 (adopted)                  %d" % d.manmask.isin([2, 3]).sum())
    print("        manmask 0 with C IV coverage (unusable)   %d"
          % int(((d.manmask == 0) & d.covered).sum()))
    print("  and it must EXCLUDE %d objects with no C IV coverage"
          % int(((d.manmask == 0) & ~d.covered).sum()))

    # where the queue column disagrees -- the reason this audit exists
    dis = d[(d.covered) & (d.queue_civ_frac == 0) |
            (~d.covered) & (d.queue_civ_frac > 0)]
    if len(dis):
        print()
        print("  work_queue.civ_frac_best disagrees for %d objects:" % len(dis))
        for _, r in dis.iterrows():
            print("     %-5d %-28s mm=%-3s queue=%-5s measured %d px  %s"
                  % (r.order, r.name_pub[:28], r.manmask, r.queue_civ_frac,
                     r.civ_px, r["product"][:34]))

    odd = d[d.manmask.isin([2, 3]) & ~d.covered]
    if len(odd):
        print()
        print("  *** %d MEASURED objects fail the coverage test -- investigate:" % len(odd))
        for _, r in odd.iterrows():
            print("     %-5d %-28s mm=%s  %d px  %s"
                  % (r.order, r.name_pub[:28], r.manmask, r.civ_px, r["product"][:34]))

    if a.csv:
        d.to_csv(OUT, index=False)
        print()
        print("  wrote %s" % OUT)


if __name__ == "__main__":
    main()
