#!/bin/sh
# Open the interactive fitter on the EXPERIMENTAL sqrt(N) co-add error product.
#
# WHAT THIS IS.  coadd.py writes varweighted_errs[i] = _med_err -- the weighted
# median of the contributing exposures' errors, i.e. a SINGLE-EXPOSURE error
# however many exposures went in, so adding data never improves the quoted
# signal-to-noise.  The product under pipeline_output/sqrtn_test/ was rebuilt
# with the formal arm of coadd._combined_error instead: 1/sqrt(sum w), times
# 1.2533 for a median above three effective exposures.  The chi2 inflation in
# that function is dead code and stays dead.
#
# WHY IT MATTERS, on index 70 (I Zw 1): the C IV blue wing is covered by 14 H13
# exposures and the rest of the line by 3 H19, so discarding the sqrt(N) gain
# erased the fact that the blue wing is the BEST-observed part of the line.  The
# fit moved from +128.4 to +1506.2 km/s, against a data half-flux centroid of
# +1440.5 and Trevor's +1574.9.
#
# NOT ADOPTED.  rebinning/coadd.py is unmodified; only this one object has been
# rebuilt this way.  The window title will read [spectra: sqrtn_test/rebin] so
# there is no doubt which tree is open.
#
#   ./open_fitter_sqrtn.sh                       open the fitter
#   ./open_fitter_sqrtn.sh --selftest "Mrk 1502_FOS"
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/opt/homebrew/Caskroom/miniforge/base/bin/python
PAPER=/Users/gtr/Work/git/HST-Paper
# Probe for the products rather than trusting $HERE -- finalrun/ exists twice
# and only one copy has pipeline_output (the other is gitignored and empty).
# This script was written WITHOUT the probe hours after that same defect was
# fixed in open_fitter.sh, and the repo copy duly pointed at nothing.
REB=""
for cand in "$HERE" /Users/gtr/Work/projects/hstica/finalrun; do
  if [ -n "$(ls "$cand/pipeline_output/sqrtn_test/rebin"/*.fits 2>/dev/null | head -n 1)" ]; then
    REB="$cand/pipeline_output/sqrtn_test/rebin"
    break
  fi
done
if [ -z "$REB" ]; then
  echo "open_fitter_sqrtn: no sqrt(N) products found.  Looked in:" >&2
  for cand in "$HERE" /Users/gtr/Work/projects/hstica/finalrun; do
    echo "    $cand/pipeline_output/sqrtn_test/rebin" >&2
  done
  exit 2
fi
echo "open_fitter_sqrtn: EXPERIMENTAL sqrt(N) tree -> $REB" >&2
# iterations still land beside the object's others, so they can be flipped through
PYTHONPATH="$PAPER${PYTHONPATH:+:$PYTHONPATH}" \
HST_PAPER_REBIN_DIR="$REB" \
HSTICA_REBIN="$REB" \
HSTICA_ITERDIR="$HERE/pipeline_output/chosen_visit/fits" \
exec "$PY" -m ica.manual_fix_gui --rebin-dir "$REB" "$@"
