"""Rebuild every chosen-visit product IN PLACE from its own staged exposures.

For a pure code change in the co-adder -- here the inverse-variance error adopted
on 2026-11-17 -- the right rebuild is one where the SET OF EXPOSURES cannot
change.  fit_chosen_visit.py re-selects the visit from MAST each time, so it is
not that: it would fold any drift in the selection into the comparison.  This
reads each existing product, takes its name, instrument and redshift FROM THE
PRODUCT, and re-runs rebinning.coadd.rebin on the stage directory that built it,
with the same arguments production used (no SDSS splice: SPLICE_SDSS is False).

Products with no stage directory (the LEDA 50824 splice, index 90) are listed
and left alone; splice_spectra.py rebuilds that one from the rebuilt pieces.

ALWAYS snapshot first.  The pre-adoption tree is
    pipeline_output/archive_pre_sqrtn_20261117_0904/
and `rsync -a <archive>/rebin/ pipeline_output/chosen_visit/rebin/` reverts.

    python rebuild_from_stage.py                 # all products
    python rebuild_from_stage.py --name "Mrk 1502_FOS"
"""
import argparse, glob, json, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from rebinning import coadd as CO

REBIN = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
STAGE = os.path.join(HERE, "pipeline_output", "chosen_visit", "stage")
QUEUE = os.path.join(HERE, "pipeline_output", "work_queue.csv")


def index_for(stem):
    """work_queue order for a product stem, via the stems column."""
    q = pd.read_csv(QUEUE)
    for _, r in q.iterrows():
        if stem in str(r.get("stems", "")).split(";"):
            return int(r["order"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="one product stem, e.g. 'Mrk 1502_FOS'")
    a = ap.parse_args()
    prods = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(REBIN, "*.fits")))
    if a.name:
        prods = [p for p in prods if p == a.name]
    done, skipped, failed = [], [], []
    for stem in prods:
        base, inst = stem.rsplit("_", 1)
        i = index_for(stem)
        d = os.path.join(STAGE, "%03d_%s" % (i, inst)) if i is not None else None
        if inst == "SPLICE" or d is None or not os.path.isdir(d):
            skipped.append((stem, i, "no stage dir" if inst != "SPLICE" else "splice; use splice_spectra.py"))
            print("SKIP   %-44s %s" % (stem, skipped[-1][2]), flush=True)
            continue
        t = Table.read(os.path.join(REBIN, stem + ".fits"))
        z = float(np.nanmedian(np.asarray(t["Redshift"], float)))
        print("REBIN  %-44s index %3d  z=%.6f  %s" % (stem, i, z, os.path.basename(d)), flush=True)
        try:
            CO.rebin(base, z, inst, None, data_path=d, output_dir=REBIN, flat=True)
            done.append(stem)
        except Exception as ex:
            failed.append((stem, "%s: %s" % (type(ex).__name__, str(ex)[:160])))
            print("FAILED %-44s %s" % (stem, failed[-1][1]), flush=True)
    print("\nrebuilt %d   skipped %d   failed %d" % (len(done), len(skipped), len(failed)))
    for s_ in skipped: print("  skipped: %s  (%s)" % (s_[0], s_[2]))
    for f_ in failed:  print("  FAILED : %s  (%s)" % f_)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
