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
PYTHONPATH="$PAPER${PYTHONPATH:+:$PYTHONPATH}" \
HST_PAPER_REBIN_DIR="$HERE/pipeline_output/chosen_visit/rebin" \
HSTICA_REBIN="$HERE/pipeline_output/chosen_visit/rebin" \
HSTICA_ITERDIR="$HERE/pipeline_output/chosen_visit/fits" \
exec "$PY" -m ica.manual_fix_gui "$@"
