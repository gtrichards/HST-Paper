#!/bin/sh
# The sqrt(N) adoption chain, 2026-11-17.  Detached; log in pipeline_output/final/.
#   1. rebuild every chosen-visit product from its stage dir   (rebuild_from_stage.py)
#   2. rebuild the one splice, index 90                         (splice_spectra.py)
#   3. rebuild the IUE products through the same co-adder       (iue/rebin_iue.py)
#   4. replay every adopted HST fit as iterNN_sqrtn, BEST_ untouched (replay_fits.py)
#   5. before/after ledger                                      (diff_nread.py --label sqrtn)
# Revert: rsync -a pipeline_output/archive_pre_sqrtn_20261117_0904/{rebin,fits}/ back over pipeline_output/chosen_visit/.
set -u
PY=/opt/homebrew/Caskroom/miniforge/base/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=$HERE/pipeline_output/final/sqrtn_adoption.log
SNAP=$HERE/pipeline_output/archive_pre_sqrtn_20261117_0904
cd "$HERE"
{
echo "START $(date)"
echo "snapshot: $SNAP"
echo "coadd.py: $(git -C /Users/gtr/Work/git/HST-Paper rev-parse --short HEAD) $(git -C /Users/gtr/Work/git/HST-Paper status --short rebinning/coadd.py)"
echo "== 1. rebuild from stage =="
$PY rebuild_from_stage.py
echo "== 2. splice index 90 =="
$PY splice_spectra.py --index 90
echo "== 3. IUE =="
( cd iue && $PY rebin_iue.py )
echo "== 4. replay adopted fits as iterNN_sqrtn =="
$PY replay_fits.py --snap "$SNAP" --label sqrtn
echo "== 5. diff ledger =="
$PY diff_nread.py --snap "$SNAP" --label sqrtn --csv pipeline_output/final/sqrtn_diff.csv
echo "END $(date)"
echo ADOPTION_CHAIN_DONE
} > "$LOG" 2>&1
