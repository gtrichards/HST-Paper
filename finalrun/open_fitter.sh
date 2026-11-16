#!/bin/sh
# Open the interactive ICA fitter on the chosen-visit tree.
#
# Three things have to be right at once and each has bitten us: the interpreter
# must be the miniforge one (it is the only python here with PySide6), the
# HST-Paper repository root must be importable (the GUI does "from ica.manual_fix
# import ...", so running the file by its path is not enough), and the rebin
# directory must point at the chosen-visit spectra.  Note that the GUI reads
# HST_PAPER_REBIN_DIR, not the HSTICA_REBIN that the finalrun scripts use; both
# are set here so the GUI and anything it shells out to see the same tree.
# Keeping all of it in one script means none of it can be forgotten.
#
#   ./open_fitter.sh              open the fitter
#   ./open_fitter.sh --selftest "NGC-7469_STIS"    check it works, no window
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/opt/homebrew/Caskroom/miniforge/base/bin/python
PAPER=/Users/gtr/Work/git/HST-Paper

# WHERE THE DATA IS, which is not necessarily where this script is.  finalrun/
# exists twice -- in the HST-Paper repository and in the working tree -- and the
# two copies of this script are byte-identical, so a script that trusts its own
# location works from one and cannot work from the other: pipeline_output is
# gitignored, so the repository copy points at a directory that will always be
# empty.  On 2026-11-15 GTR ran the repository copy and got "No FITS files found"
# with no indication that a populated tree existed elsewhere.
#
# So probe for the products instead of assuming.  Override with HSTICA_DATA_ROOT.
DATA_ROOT="${HSTICA_DATA_ROOT:-}"
if [ -z "$DATA_ROOT" ]; then
  for cand in "$HERE" /Users/gtr/Work/projects/hstica/finalrun; do
    if [ -n "$(ls "$cand/pipeline_output/chosen_visit/rebin"/*.fits 2>/dev/null | head -n 1)" ]; then
      DATA_ROOT="$cand"
      break
    fi
  done
fi
if [ -z "$DATA_ROOT" ]; then
  echo "open_fitter: no rebinned spectra found." >&2
  echo "  looked in:" >&2
  for cand in "$HERE" /Users/gtr/Work/projects/hstica/finalrun; do
    echo "    $cand/pipeline_output/chosen_visit/rebin" >&2
  done
  echo "  set HSTICA_DATA_ROOT to the finalrun directory that holds them." >&2
  exit 2
fi
if [ "$DATA_ROOT" != "$HERE" ]; then
  echo "open_fitter: this script lives in $HERE," >&2
  echo "             but the spectra are in $DATA_ROOT -- using those." >&2
fi
PYTHONPATH="$PAPER${PYTHONPATH:+:$PYTHONPATH}" \
HST_PAPER_REBIN_DIR="$DATA_ROOT/pipeline_output/chosen_visit/rebin" \
HSTICA_REBIN="$DATA_ROOT/pipeline_output/chosen_visit/rebin" \
HSTICA_ITERDIR="$DATA_ROOT/pipeline_output/chosen_visit/fits" \
exec "$PY" -m ica.manual_fix_gui "$@"
