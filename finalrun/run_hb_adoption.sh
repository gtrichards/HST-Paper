#!/bin/sh
# The Handbook-truncation + per-exposure 1% guard adoption chain, 2026-11-17.
# Same chain as run_sqrtn_adoption.sh, label "hb".  Log in pipeline_output/final/.
#   0. snapshot rebin/fits/final/iue_rebin/iue_fits/tables + the code that made them
#   1. rebuild every chosen-visit product from its stage dir   (rebuild_from_stage.py)
#   2. rebuild the one splice, index 90                         (splice_spectra.py)
#   3. rebuild the IUE products through the same co-adder       (iue/rebin_iue.py)
#   4. replay every adopted HST fit as iterNN_hb, BEST_ untouched (replay_fits.py)
#   5. before/after ledger                                      (diff_nread.py --label hb)
# Revert: rsync -a --delete $SNAP/rebin/ pipeline_output/chosen_visit/rebin/ (same for fits),
#         $SNAP/{final,iue_rebin,iue_fits,tables}/ back over pipeline_output/, and
#         git checkout rebinning/coadd.py rebinning/read_spec_data.py; rm rebinning/fos_handbook_ranges.py.
set -u
PY=/opt/homebrew/Caskroom/miniforge/base/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=$HERE/pipeline_output/final/hb_adoption.log
SNAP=$HERE/pipeline_output/archive_pre_hb_$(date +%Y%m%d_%H%M)
REPO=/Users/gtr/Work/git/HST-Paper
cd "$HERE"
{
echo "START $(date)"
echo "== 0. snapshot -> $SNAP =="
mkdir -p "$SNAP"
rsync -a pipeline_output/chosen_visit/rebin/ "$SNAP/rebin/"
rsync -a pipeline_output/chosen_visit/fits/  "$SNAP/fits/"
rsync -a pipeline_output/final/              "$SNAP/final/"
rsync -a pipeline_output/iue_rebin/          "$SNAP/iue_rebin/" 2>/dev/null || true
rsync -a pipeline_output/iue_fits/           "$SNAP/iue_fits/"  2>/dev/null || true
rsync -a pipeline_output/tables/             "$SNAP/tables/"    2>/dev/null || true
cp $REPO/rebinning/coadd.py "$SNAP/coadd.py.pre_hb"
cp $REPO/rebinning/read_spec_data.py "$SNAP/read_spec_data.py.pre_hb"
git -C $REPO rev-parse HEAD > "$SNAP/HST-Paper.commit"; git -C $REPO status --short >> "$SNAP/HST-Paper.commit"
ls pipeline_output/chosen_visit/rebin | wc -l > "$SNAP/INDICES.txt"
echo "snapshot verify (rebin, fits): $(diff -rq pipeline_output/chosen_visit/rebin "$SNAP/rebin" | wc -l) $(diff -rq pipeline_output/chosen_visit/fits "$SNAP/fits" | wc -l) differing entries (want 0 0)"
echo "code: $(git -C $REPO rev-parse --short HEAD) $(git -C $REPO status --short rebinning/)"
echo "== 1. rebuild from stage =="
$PY rebuild_from_stage.py
echo "== 2. splice index 90 =="
$PY splice_spectra.py --index 90
echo "== 3. IUE =="
( cd iue && $PY rebin_iue.py )
echo "== 4. replay adopted fits as iterNN_hb =="
$PY replay_fits.py --snap "$SNAP" --label hb
echo "== 5. diff ledger =="
$PY diff_nread.py --snap "$SNAP" --label hb --csv pipeline_output/final/hb_diff.csv
echo "END $(date)"
echo ADOPTION_CHAIN_DONE
} > "$LOG" 2>&1
