#!/bin/sh
# Generic adoption chain:  ./run_adoption_chain.sh LABEL
# (supersedes run_sqrtn_adoption.sh / run_hb_adoption.sh, which were the same chain with the label baked in)
#   0. snapshot rebin/fits/final/iue_rebin/iue_fits/tables + the code that made them -> archive_pre_LABEL_<stamp>
#   1. rebuild every chosen-visit product from its stage dir   (rebuild_from_stage.py)
#   2. rebuild the one splice, index 90                         (splice_spectra.py)
#   3. rebuild the IUE products through the same co-adder       (iue/rebin_iue.py)
#   4. replay every adopted HST fit as iterNN_LABEL, BEST_ untouched (replay_fits.py)
#   5. before/after ledger                                      (diff_nread.py --label LABEL -> final/LABEL_diff.csv)
# Revert: rsync -a --delete $SNAP/rebin/ pipeline_output/chosen_visit/rebin/ (same for fits),
#         $SNAP/{final,iue_rebin,iue_fits,tables}/ back over pipeline_output/, and check out the code at $SNAP/HST-Paper.commit.
set -u
LABEL=${1:?usage: run_adoption_chain.sh LABEL}
PY=/opt/homebrew/Caskroom/miniforge/base/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=$HERE/pipeline_output/final/${LABEL}_adoption.log
SNAP=$HERE/pipeline_output/archive_pre_${LABEL}_$(date +%Y%m%d_%H%M)
REPO=/Users/gtr/Work/git/HST-Paper
cd "$HERE"
{
echo "START $(date)   label=$LABEL"
echo "== 0. snapshot -> $SNAP =="
mkdir -p "$SNAP"
rsync -a pipeline_output/chosen_visit/rebin/ "$SNAP/rebin/"
rsync -a pipeline_output/chosen_visit/fits/  "$SNAP/fits/"
rsync -a pipeline_output/final/              "$SNAP/final/"
rsync -a pipeline_output/iue_rebin/          "$SNAP/iue_rebin/" 2>/dev/null || true
rsync -a pipeline_output/iue_fits/           "$SNAP/iue_fits/"  2>/dev/null || true
rsync -a pipeline_output/tables/             "$SNAP/tables/"    2>/dev/null || true
cp $REPO/rebinning/coadd.py "$SNAP/coadd.py.pre_$LABEL"
cp $REPO/rebinning/read_spec_data.py "$SNAP/read_spec_data.py.pre_$LABEL"
git -C $REPO rev-parse HEAD > "$SNAP/HST-Paper.commit"; git -C $REPO status --short >> "$SNAP/HST-Paper.commit"
# the index list replay_fits.py reads: every index with a fits dir (it skips those without an adopted record)
ls "$SNAP/fits" | sed -n 's/^\([0-9][0-9][0-9]\)_.*/\1/p' | sort -u | sed 's/^0*//' > "$SNAP/INDICES.txt"
echo "snapshot verify (rebin, fits): $(diff -rq pipeline_output/chosen_visit/rebin "$SNAP/rebin" | wc -l) $(diff -rq pipeline_output/chosen_visit/fits "$SNAP/fits" | wc -l) differing entries (want 0 0); $(wc -w < "$SNAP/INDICES.txt") indices listed"
echo "code: $(git -C $REPO rev-parse --short HEAD) $(git -C $REPO status --short rebinning/)"
echo "== 1. rebuild from stage =="
$PY rebuild_from_stage.py
echo "== 2. splice index 90 =="
$PY splice_spectra.py --index 90
echo "== 3. IUE =="
( cd iue && $PY rebin_iue.py )
echo "== 4. replay adopted fits as iterNN_$LABEL =="
$PY replay_fits.py --snap "$SNAP" --label "$LABEL"
echo "== 5. diff ledger =="
$PY diff_nread.py --snap "$SNAP" --label "$LABEL" --csv "pipeline_output/final/${LABEL}_diff.csv"
echo "END $(date)"
echo ADOPTION_CHAIN_DONE
} > "$LOG" 2>&1
