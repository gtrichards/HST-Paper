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
IUE="$HERE/iue"
PYTHONPATH="$PAPER${PYTHONPATH:+:$PYTHONPATH}" \
HST_PAPER_REBIN_DIR="$IUE/rebin" \
HSTICA_REBIN="$IUE/rebin" \
HSTICA_ITERDIR="$IUE/fits" \
exec "$PY" -m ica.manual_fix_gui \
  --rebin-dir "$IUE/rebin" \
  --overrides "$IUE/iue_overrides.json" \
  --measurements "$IUE/iue_measurements.csv" "$@"
