"""Audit: which spectra the rebin master is missing, and what that costs.

The rebin master holds one file per object *and instrument*
(`<common_name>_<INSTRUMENT>.fits`).  For a number of objects some of those
files were never produced, so the pipeline fits fewer spectra than the sample
is supposed to have.  This was found when 3C 273 -- 56 HST observations in the
v22 catalogue -- came through the pass with a single COS/G130M file covering
1120-1278 A rest and was provisionally excluded for having no C IV data.

The test applied here is deliberately NOT "does the missing spectrum cover
C IV".  GTR's rule: "It only matters if there is coverage of any parts of the
ICA components (so long as there is some coverage of C IV in other spectra)."
The ICA fits its components over 1260-3000 A, so any missing UV spectrum
degrades the reconstruction whether or not it reaches the line.  C IV coverage
is reported as an extra column because it separates "this measurement could
sharpen" from "this object was excluded for a reason that is not true".

Sources compared, none of them re-queried here:
  work_queue.csv        instruments_master -- derived from the stems that exist
                        in RebinnedSpec_master, i.e. what we actually have
  v23_v22_crossref.csv  v23_gratings       -- what the v23 MAST query found
                        civ_covering       -- which of those gratings reach C IV
  master_catalog_v22    all_instruments    -- what the ORIGINAL v22 query found,
                        n_obs                 which settles whether a missing
                                              spectrum was known about and lost
                                              or is new in v23
"""

import csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pipeline_output", "missing_spectra_audit.csv")


def insts(s, sep=","):
    return set(x.split("/")[0].strip() for x in (s or "").split(sep) if x.strip())


def main():
    po = os.path.join(HERE, "pipeline_output")
    cross = {r["xlsx_row"]: r for r in csv.DictReader(open(os.path.join(po, "v23_v22_crossref.csv")))}
    v22 = {r["common_name"]: r for r in csv.DictReader(
        open("/Users/gtr/Work/projects/hstica/sandbox/pipeline_output/master_catalog_v22.csv"))}
    dec = {r["index"]: r for r in csv.DictReader(open(os.path.join(po, "final", "pass_decisions.csv")))} \
        if os.path.exists(os.path.join(po, "final", "pass_decisions.csv")) else {}

    rows = []
    for q in csv.DictReader(open(os.path.join(po, "work_queue.csv"))):
        c = cross.get(q["row_num"])
        if not c:
            continue
        want = insts(c.get("v23_gratings"))
        have = set(x for x in (q["instruments_master"] or "").split(";") if x)
        missing = want - have
        if not (want and missing):
            continue
        v = v22.get(q["name_mast_key"], {})
        v22_inst = insts(v.get("all_instruments"), sep="/")
        civ_cov = insts(c.get("civ_covering"))
        d = dec.get(q["order"], {})
        rows.append(dict(
            index=int(q["order"]), row_num=q["row_num"], common_name=q["name_mast_key"],
            have=";".join(sorted(have)), missing=";".join(sorted(missing)),
            n_missing=len(missing), have_none=(not have),
            v22_listed=";".join(sorted(v22_inst)), v22_n_obs=v.get("n_obs", ""),
            # a missing instrument the v22 query already knew about was LOST in
            # retrieval or rebinning; one v22 did not list is new in the v23 query
            known_to_v22=bool(missing & v22_inst),
            civ_covering=";".join(sorted(civ_cov)),
            civ_lost_entirely=bool(civ_cov and civ_cov.issubset(missing)),
            decided=bool(d), manmask=d.get("manmask", ""), reject_reason=d.get("reject_reason", ""),
        ))

    rows.sort(key=lambda r: r["index"])
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n = len(rows)
    print("objects missing at least one instrument : %d" % n)
    print("  with nothing rebinned at all          : %d" % sum(r["have_none"] for r in rows))
    print("  the missing instrument known to v22   : %d  (lost in retrieval/rebin)" % sum(r["known_to_v22"] for r in rows))
    print("  new in the v23 query                  : %d" % sum(not r["known_to_v22"] for r in rows))
    print("  missing instrument is the only C IV   : %d" % sum(r["civ_lost_entirely"] for r in rows))
    dn = [r for r in rows if r["decided"]]
    print("\nalready decided: %d   still ahead: %d" % (len(dn), n - len(dn)))
    print("  decided, verdict likely wrong (C IV lost) : %s"
          % ", ".join("%d %s" % (r["index"], r["common_name"]) for r in dn if r["civ_lost_entirely"]))
    print("  decided, measurement stands but could sharpen: %d objects" % sum(1 for r in dn if not r["civ_lost_entirely"]))
    print("\n-> %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
