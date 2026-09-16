"""
Stage 3: fit both arms of the splice test and render the diagnostic plots.

Runs the ICA fit with no manual overrides on each of the 15 objects, once from
the HST-only spectra and once from the spliced HST+SDSS spectra, and writes the
same diagnostic figure the manual-fix GUI's Save produces so the two can be
compared object by object.

Plots land next to the spectra so the whole experiment is self-contained.
"""

import os
import re
import sys
import tempfile
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")

import pandas as pd
from PySide6 import QtWidgets

from ica.manual_fix_gui import ManualFixWindow, GuiFixProcessor
from ica import manual_fix_store as store

BASE = "/Users/gtr/Dropbox/HST/RebinnedSpec_HSTSDSS_test15"
ARMS = ("hst_only", "spliced")

# The GUI window wants an override-store path, but this script deliberately
# fits with no overrides, so the store must be a throwaway that cannot reach
# the real one. This used to be a hardcoded session scratchpad, which stopped
# existing as soon as that session ended; a temp dir keeps it self-contained.
SCRATCH = tempfile.mkdtemp(prefix="splice15_fit_")


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rows = []

    for arm in ARMS:
        rebin_dir = os.path.join(BASE, arm)
        plots = os.path.join(BASE, "plots_" + arm)
        os.makedirs(plots, exist_ok=True)

        proc = GuiFixProcessor(rebin_path=rebin_dir, master_mode=True,
                               output_path=plots)
        names = proc.list_master_objects()
        print("=== %s: %d objects ===" % (arm, len(names)), flush=True)

        win = ManualFixWindow(proc, names, os.path.join(SCRATCH, "splice_ov.json"),
                              measurements_path=os.path.join(plots, "measurements.csv"),
                              plots_dir=plots)
        win.resize(1600, 820)
        win.show()

        for name in names:
            try:
                # no overrides: this is the like-for-like comparison
                res = proc.fit_for_gui(name)
                win.current = name
                win._draw(res)
                win._save_plot(os.path.join(
                    plots, re.sub(r"[^A-Za-z0-9.+_-]+", "_", name).strip("_") + ".png"))
                rows.append(dict(arm=arm, object=name, civ_blue=res["civ_blue"],
                                 civ_ew=res["civ_ew"], f2500=res["f2500"],
                                 z=res["z"], status="ok"))
                print("  %-28s blue=%9.1f ew=%9.2f" % (name[:28], res["civ_blue"],
                                                       res["civ_ew"]), flush=True)
            except Exception as exc:
                rows.append(dict(arm=arm, object=name, status="FAIL: %s"
                                 % type(exc).__name__))
                print("  %-28s FAIL %s" % (name[:28], exc), flush=True)
                traceback.print_exc()

    df = pd.DataFrame(rows)
    out = os.path.join(BASE, "fit_comparison.csv")
    df.to_csv(out, index=False)

    ok = df[df.status == "ok"]
    piv = ok.pivot_table(index="object", columns="arm",
                         values=["civ_blue", "civ_ew"])
    piv.to_csv(os.path.join(BASE, "fit_comparison_wide.csv"))
    print("\n%s\n" % piv.to_string(), flush=True)
    print("-> %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
