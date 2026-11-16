#!/bin/sh
# Open the interactive ICA fitter on the IUE tree.
#
# Same three requirements as open_fitter.sh and for the same reasons: the
# miniforge interpreter is the only python here with PySide6; the HST-Paper
# root must be importable because the GUI does "from ica.manual_fix import ...",
# so running the file by its path is not enough; and the rebin directory has to
# be pointed at the IUE products rather than the chosen-visit ones.
#
# The GUI reads HST_PAPER_REBIN_DIR while the finalrun scripts read
# HSTICA_REBIN, so both are set, and HSTICA_ITERDIR is kept separate from the
# HST iterations so IUE fits cannot land in the 474-object tree.
#
# The override and measurement stores are also kept separate: the GUI otherwise
# writes IUE results into ica/manual_fix_overrides.json alongside the HST pass.
#
#   ./open_fitter_iue.sh                      open the fitter on the IUE tree
#   ./open_fitter_iue.sh --selftest "QSO 1629+372_IUE"   check it, no window
#
# Objects are named <NAME>_IUE, e.g. "3C 120_IUE".
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
    if [ -n "$(ls "$cand/iue/rebin"/*.fits 2>/dev/null | head -n 1)" ]; then
      DATA_ROOT="$cand"
      break
    fi
  done
fi
if [ -z "$DATA_ROOT" ]; then
  echo "open_fitter: no rebinned spectra found." >&2
  echo "  looked in:" >&2
  for cand in "$HERE" /Users/gtr/Work/projects/hstica/finalrun; do
    echo "    $cand/iue/rebin" >&2
  done
  echo "  set HSTICA_DATA_ROOT to the finalrun directory that holds them." >&2
  exit 2
fi
if [ "$DATA_ROOT" != "$HERE" ]; then
  echo "open_fitter: this script lives in $HERE," >&2
  echo "             but the spectra are in $DATA_ROOT -- using those." >&2
fi
IUE="$DATA_ROOT/iue"
PYTHONPATH="$PAPER${PYTHONPATH:+:$PYTHONPATH}" \
HST_PAPER_REBIN_DIR="$IUE/rebin" \
HSTICA_REBIN="$IUE/rebin" \
HSTICA_ITERDIR="$IUE/fits" \
exec "$PY" -m ica.manual_fix_gui \
  --rebin-dir "$IUE/rebin" \
  --overrides "$IUE/iue_overrides.json" \
  --measurements "$IUE/iue_measurements.csv" "$@"
