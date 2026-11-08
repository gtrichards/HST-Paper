"""Assemble the publication archive: the materials that go with the paper.

GTR, 2026-11-07: "Should we maybe make a folder in HST-Paper for
publication-ready materials (as opposed to 'finalrun' which is more like the
final production run, but might contain things that aren't needed for the
publication)?"  Yes, and the reason to GENERATE it rather than copy it by hand is
that this project has already been bitten once by a hand-maintained second copy:
/Users/gtr/Work/projects/hstica/finalrun and HST-Paper/finalrun drifted, and on
2026-11-07 a commit swept up changes that had been sitting since 3 and 6
November.  A third copy maintained the same way would drift the same way.  So the
archive is a BUILD PRODUCT with a checksum manifest, and verify_archive.py says
whether what is on disk is what was built.

WHAT GOES IN, and the test is "would a reader reproducing a measurement or
checking the sample need this".

    measurements.csv    one row per object in the published sample
    decisions.csv       the ledger: verdict and full reasoning, per object
    sample_query.md     the sample-defining query and its documented checks
    spectra/            the rebinned spectrum each measurement was made from
    fits/              the adopted fit record, which carries the override that
                        produced it -- 49 of these are the ONLY copy of their
                        masks (45 HST, 4 IUE), so they are not reproducible from
                        the override stores alone
    figures/            the adopted fit figure per object
    provenance/         work_queue.csv, both override stores, the chosen-visit
                        and chosen-run tables -- the hand-made decisions about
                        WHICH photons entered each measurement
    code/               the measurement path only

WHAT STAYS OUT.  finalrun holds 81 scripts; about a dozen are the measurement
path and the rest are audits, experiments and one-off figures (coadd_experiment,
degrade_experiment, figure_ew_bug, trial_z, the Mg II and SDSS sweeps, the
luminosity-correction fixups).  They are real project history and live in the
repository, not in a published deposit.  Also out: 3.0 GB of staged exposures,
re-downloadable from MAST by pipeline/retrieve_spectra.py, and 755 MB of
iteration figures, regenerable from the records with iterate_fit --replot.  The
README lists what was omitted rather than leaving it silently absent.

WHICH OBJECTS.  The sample is the one the query must reproduce: manmask 2 or 3
(adopted) plus manmask 0 WITH C IV coverage (looked at, judged unmeasurable).
GTR: "The archive can include the manmask=0 rows that have CIV coverage."  A
reader checking the query needs the rejected-with-coverage spectra too, or they
cannot see why those were rejected.  The manmask-0 rows with NO coverage have
nothing to deposit and are excluded, which is the same cut the query makes.

spectra/ and figures/ are NOT versioned -- see archive/.gitignore.  They are
regenerable, they are pinned byte-for-byte by MANIFEST.csv, and committing ~80
MB on every rebuild would bloat the repository for no gain.  Everything small
and hand-made IS versioned.

    python build_archive.py                 # build, then write MANIFEST.csv
    python build_archive.py --no-bulk       # text products and manifest only
    HSTICA_ROOT=/path/to/finalrun python build_archive.py
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.environ.get("HSTICA_ROOT", "/Users/gtr/Work/projects/hstica/finalrun")

LEDGER = os.path.join(PAPER, "Data", "decisions.csv")
COVERAGE = os.path.join(ROOT, "pipeline_output", "final", "civ_coverage.csv")
QUEUE = os.path.join(ROOT, "pipeline_output", "work_queue.csv")

#: (BEST-record tree, rebinned-product tree, label)
TREES = [(os.path.join(ROOT, "pipeline_output", "chosen_visit", "fits"),
          os.path.join(ROOT, "pipeline_output", "chosen_visit", "rebin"), "HST"),
         (os.path.join(ROOT, "iue", "fits"), os.path.join(ROOT, "iue", "rebin"), "IUE")]
#: fallback tree for objects that were never staged for fitting -- the manmask-0
#: rows with coverage often have no chosen-visit product, only the v23 one
V23 = os.path.join(ROOT, "data_v23", "RebinnedSpec_v23")

#: The measurement path, and nothing else.  Paths are relative to the repository
#: root so that what the archive ships is literally what the repository holds.
CODE = [
    "rebinning/coadd.py",
    "ica/manual_fix.py",
    "finalrun/rebin_path.py",
    "finalrun/object_paths.py",
    "finalrun/build_work_queue.py",
    "finalrun/prepare_objects.py",
    "finalrun/fit_chosen_visit.py",
    "finalrun/iterate_fit.py",
    "finalrun/run_batch.py",
    "finalrun/build_paper_table.py",
    "finalrun/civ_coverage_audit.py",
    "finalrun/red_anchor_audit.py",
    "finalrun/iue/retrieve_iue.py",
    "finalrun/iue/choose_run_iue.py",
    "finalrun/iue/rebin_iue.py",
    "finalrun/iue/prepare_iue.py",
    "paper_figures_v23/make_table1.py",
]

PROVENANCE = [
    ("finalrun/pipeline_output/work_queue.csv", "work_queue.csv"),
    ("ica/manual_fix_overrides.json", "manual_fix_overrides.json"),
    ("finalrun/iue/iue_overrides.json", "iue_overrides.json"),
    ("ica/redshift_overrides.csv", "redshift_overrides.csv"),
    ("finalrun/pipeline_output/final/name_overrides.csv", "name_overrides.csv"),
    ("finalrun/pipeline_output/final/spectrum_overrides.csv", "spectrum_overrides.csv"),
    ("finalrun/pipeline_output/final/bal_status.csv", "bal_status.csv"),
    ("finalrun/iue/iue_chosen_run.csv", "iue_chosen_run.csv"),
    ("finalrun/iue/iue_decisions.csv", "iue_decisions.csv"),
]


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ledger():
    """order -> (manmask, folder) from the canonical ledger."""
    out = {}
    for r in csv.DictReader(open(LEDGER)):
        o = int(re.match(r"^(\d+)", r["folder"]).group(1))
        out[o] = (str(r["manmask"]).strip(), r["folder"])
    return out


def coverage():
    """order -> (covered, civ_px) from civ_coverage_audit.py --csv."""
    if not os.path.exists(COVERAGE):
        sys.exit("missing %s -- run finalrun/civ_coverage_audit.py --csv first" % COVERAGE)
    out = {}
    for r in csv.DictReader(open(COVERAGE)):
        out[int(r["order"])] = (str(r["covered"]).lower() == "true", int(r["civ_px"]))
    return out


def adopted_records():
    """order -> (record path, product path, source label) for every adopted fit."""
    out = {}
    for fits_dir, rebin_dir, label in TREES:
        for rec in sorted(glob.glob(os.path.join(fits_dir, "*", "records", "BEST_*.json"))):
            m = re.search(r"/(\d+)_[^/]*/records/", rec)
            if not m:
                continue
            o = int(m.group(1))
            nm = json.load(open(rec)).get("name", "")
            p = os.path.join(rebin_dir, "%s.fits" % nm)
            out[o] = (rec, p if os.path.exists(p) else None, label, nm)
    return out


def product_for(order, nm_keys):
    """Any rebinned product for an object with no adopted fit (the manmask-0 rows)."""
    for tree in [t[1] for t in TREES] + [V23]:
        for k in nm_keys:
            if not k:
                continue
            for p in sorted(glob.glob(os.path.join(glob.escape(tree),
                                                   "%s*.fits" % glob.escape(k)))):
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bulk", action="store_true",
                    help="skip spectra/ and figures/; text products and manifest only")
    a = ap.parse_args()

    led, cov, recs = ledger(), coverage(), adopted_records()
    q = {int(r["order"]): r for r in csv.DictReader(open(QUEUE))}

    sample = []
    for o in sorted(led):
        mm, folder = led[o]
        covered, civ_px = cov.get(o, (False, 0))
        if mm in ("2", "3"):
            keep = True
        elif mm == "0" and covered:
            keep = True
        else:
            keep = False
        if keep:
            sample.append((o, mm, folder, covered, civ_px))
    print("objects in the published sample: %d" % len(sample))
    print("   manmask 2 or 3 (adopted)                : %d"
          % sum(1 for s in sample if s[1] in ("2", "3")))
    print("   manmask 0 with C IV coverage (rejected) : %d"
          % sum(1 for s in sample if s[1] == "0"))

    for d in ("spectra", "fits", "figures", "provenance", "code"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)

    # ---- measurements.csv ------------------------------------------------
    rows = []
    n_spec = n_rec = n_fig = 0
    for o, mm, folder, covered, civ_px in sample:
        qr = q.get(o, {})
        rec = recs.get(o)
        blue = ew = l1450 = l1700 = ""
        source = ""
        recname = ""
        prod = None
        if rec:
            rp, prod, label, nm = rec
            source = label
            d = json.load(open(rp))
            res = d.get("result", {})
            blue = res.get("civ_blue", "")
            ew = res.get("civ_ew", "")
            mono = d.get("monochromatic", {})
            for key, dst in (("1450", "l1450"), ("1700", "l1700")):
                v = mono.get(key) or {}
                val = v.get("logL") if v.get("covered") else None
                if dst == "l1450":
                    l1450 = "" if val is None or val != val else round(float(val), 3)
                else:
                    l1700 = "" if val is None or val != val else round(float(val), 3)
            recname = os.path.basename(rp)
        if prod is None:
            prod = product_for(o, [str(qr.get("name_mast_key") or ""),
                                   str(qr.get("name_pub") or "")])
        rows.append(dict(
            index=o, v22_row=qr.get("row_num", ""),
            name_pub=str(qr.get("name_pub") or "").strip(),
            designation=qr.get("designation", ""),
            z=qr.get("z", ""), manmask=mm, source=source,
            civ_blueshift_kms=(round(float(blue), 1) if blue != "" else ""),
            civ_ew_A=(round(float(ew), 2) if ew != "" else ""),
            logL1450=l1450, logL1700=l1700,
            civ_px=civ_px,
            spectrum=(os.path.basename(prod) if prod else ""),
            fit_record=recname))
        if not a.no_bulk:
            if prod and os.path.exists(prod):
                shutil.copy2(prod, os.path.join(HERE, "spectra", os.path.basename(prod)))
                n_spec += 1
            if rec:
                shutil.copy2(rec[0], os.path.join(HERE, "fits", "%03d_%s" % (o, recname)))
                n_rec += 1
                # The adopted figure is renamed to BEST_<label>.png beside the
                # record when a fit is promoted, so that is the name to look
                # for; the un-prefixed <label>.png is the pre-promotion name and
                # is the fallback.  Stripping the prefix instead found only 74
                # of 366 -- and the seven it did find for IUE were the ones
                # promoted on 2026-11-07, where only the JSON had been renamed.
                # Those seven figures have since been renamed to match.
                d = os.path.dirname(os.path.dirname(rec[0]))
                stem = os.path.basename(rec[0])[len("BEST_"):-len(".json")]
                for cand in (os.path.join(d, "BEST_%s.png" % stem),
                             os.path.join(d, "%s.png" % stem)):
                    if os.path.exists(cand):
                        shutil.copy2(cand, os.path.join(HERE, "figures",
                                                        "%03d_%s" % (o, os.path.basename(cand))))
                        n_fig += 1
                        break
    flds = list(rows[0].keys())
    with open(os.path.join(HERE, "measurements.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print("   wrote measurements.csv (%d rows)" % len(rows))
    if not a.no_bulk:
        print("   spectra/ %d   fits/ %d   figures/ %d" % (n_spec, n_rec, n_fig))

    # ---- the ledger, provenance and code --------------------------------
    shutil.copy2(LEDGER, os.path.join(HERE, "decisions.csv"))
    miss = []
    for src, dst in PROVENANCE:
        p = os.path.join(PAPER, src)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(HERE, "provenance", dst))
        else:
            miss.append(src)
    for src in CODE:
        p = os.path.join(PAPER, src)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(HERE, "code", os.path.basename(src)))
        else:
            miss.append(src)
    if miss:
        print("   *** NOT FOUND, so not archived: %s" % ", ".join(miss))

    # ---- MANIFEST.csv ----------------------------------------------------
    man = []
    for dirpath, _, files in os.walk(HERE):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, HERE)
            if rel in ("MANIFEST.csv",) or rel.startswith(".git"):
                continue
            man.append(dict(path=rel, bytes=os.path.getsize(p), sha256=sha256(p)))
    man.sort(key=lambda r: r["path"])
    with open(os.path.join(HERE, "MANIFEST.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(man)
    tot = sum(r["bytes"] for r in man)
    print("   wrote MANIFEST.csv: %d files, %.1f MB" % (len(man), tot / 1048576.0))


if __name__ == "__main__":
    main()
