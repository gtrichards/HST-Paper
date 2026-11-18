#!/bin/sh
# Replay-only chain:  ./run_replay_chain.sh LABEL
# For a change to the FITTER (ica/) that leaves the co-added products untouched:
#   0. snapshot fits/final (+ the ica/ code) -> archive_pre_LABEL_<stamp>, with the index list
#   1. replay every adopted HST fit as iterNN_LABEL, BEST_ untouched   (replay_fits.py)
#   2. before/after ledger                                             (diff_nread.py --label LABEL)
# The products are NOT rebuilt (use run_adoption_chain.sh for a rebinning/ change).
set -u
LABEL=${1:?usage: run_replay_chain.sh LABEL}
PY=/opt/homebrew/Caskroom/miniforge/base/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=$HERE/pipeline_output/final/${LABEL}_adoption.log
SNAP=$HERE/pipeline_output/archive_pre_${LABEL}_$(date +%Y%m%d_%H%M)
REPO=/Users/gtr/Work/git/HST-Paper
cd "$HERE"
{
echo "START $(date)   label=$LABEL (replay only)"
echo "== 0. snapshot -> $SNAP =="
mkdir -p "$SNAP"
rsync -a pipeline_output/chosen_visit/fits/  "$SNAP/fits/"
rsync -a pipeline_output/final/              "$SNAP/final/"
mkdir -p "$SNAP/ica.pre_$LABEL"; cp $REPO/ica/run_ica.py $REPO/ica/manual_fix.py $REPO/ica/fit_composites.py "$SNAP/ica.pre_$LABEL/" 2>/dev/null
git -C $REPO rev-parse HEAD > "$SNAP/HST-Paper.commit"; git -C $REPO status --short >> "$SNAP/HST-Paper.commit"
ls "$SNAP/fits" | sed -n 's/^\([0-9][0-9][0-9]\)_.*/\1/p' | sort -u | sed 's/^0*//' > "$SNAP/INDICES.txt"
echo "snapshot verify (fits): $(diff -rq pipeline_output/chosen_visit/fits "$SNAP/fits" | wc -l) differing entries (want 0); $(wc -w < "$SNAP/INDICES.txt") indices listed"
echo "code: $(git -C $REPO rev-parse --short HEAD) $(git -C $REPO status --short ica/)"
echo "== 1. replay adopted fits as iterNN_$LABEL =="
$PY replay_fits.py --snap "$SNAP" --label "$LABEL"
echo "== 2. diff ledger =="
$PY diff_nread.py --snap "$SNAP" --label "$LABEL" --csv "pipeline_output/final/${LABEL}_diff.csv"
echo "END $(date)"
echo REPLAY_CHAIN_DONE
} > "$LOG" 2>&1
