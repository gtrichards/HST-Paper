"""The factor m by which maskNAL's nominal 3-sigma threshold is raised (Section 2.6).

ica/manual_fix.py setup_object divides the co-added flux by its median m in place,
then divides the errors by the median of the already-normalised flux (1.0), so the
errors reach run_ica.maskNAL un-normalised and its test
"median61 - flux > 3*errs" is in effect "> 3*m sigma".  This prints the distribution
of m over the adopted objects (the spectra in the latest ledger).

    python nal_threshold_factor.py [--ledger pipeline_output/final/final_diff.csv]
"""
import argparse, os
import numpy as np, pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--ledger", default=os.path.join(HERE, "pipeline_output", "final", "final_diff.csv"))
a = ap.parse_args()
REBIN = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
m = pd.Series({sp: np.nanmedian(fits.getdata(os.path.join(REBIN, sp + ".fits"), 1)["Coadded Flux (Arbitrary Units)"])
               for sp in pd.read_csv(a.ledger).drop_duplicates("spectrum").spectrum})
q = m.quantile
print("n = %d   m: median %.2f, 16-84%% %.2f-%.2f, range %.2f-%.2f" % (len(m), m.median(), q(.16), q(.84), m.min(), m.max()))
print("effective threshold 3m (sigma): median %.1f, 16-84%% %.1f-%.1f, range %.1f-%.1f"
      % (3 * m.median(), 3 * q(.16), 3 * q(.84), 3 * m.min(), 3 * m.max()))
