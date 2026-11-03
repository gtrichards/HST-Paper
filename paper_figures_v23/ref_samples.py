"""
The SDSS (Rankine+20) and GNIRS-DQS comparison samples, shared by the figure
scripts, with their luminosity moved to 1450 A so it is on the same axis as HST.

Both catalogues give log(lambda L_lambda) at 1350 and 2500 A (LOGL1350_dr16,
LOGL2500_dr16).  These are Wu & Shen (2022) DR16Q spectral-property values
(LOGL1350, LOGL2500; EBV_dr16 is their EBV), joined by position in TOPCAT on
2023-01-02.  Checked 2026-11-02 against dr16q_prop_Oct23_2022.fits.gz from
http://quasar.astro.illinois.edu/paper_data/DR16Q/ (the release that first had
LOGL2500): every SDSS value (130,118 at 1350, 138,554 at 2500) and every
GNIRS-DQS value (169, 165) is identical.  For each object the value at 1450 A is interpolated along
that object's own power law between the two:

    log lamL(1450) = log lamL(1350)
                     + [log lamL(2500) - log lamL(1350)] * log(1450/1350) / log(2500/1350)

Objects with no 2500 A value (7% of SDSS, 17% of GNIRS) use the median slope of
their own sample instead.  The median shift is -0.012 dex in lambda L_lambda,
i.e. +0.019 dex in L_nu once lambda L_lambda is divided by nu.

Returned columns added:  logLnu1450  (log L_nu at 1450 A, erg/s/Hz)
"""

import os
import numpy as np
import pandas as pd
from astropy.table import Table

C_ANG = 2.998e18                                  # speed of light, Angstrom/s
FRAC = np.log10(1450. / 1350.) / np.log10(2500. / 1350.)


def _add_logLnu1450(df):
    l1350 = df["LOGL1350_dr16"].values
    l2500 = df["LOGL2500_dr16"].values
    has2500 = l2500 > 0
    step = np.where(has2500, l2500 - l1350, np.nan)
    step = np.where(has2500, step, np.nanmedian(step))   # sample median slope
    lamL1450 = l1350 + FRAC * step
    df = df.copy()
    df["logLnu1450"] = lamL1450 - np.log10(C_ANG / 1450.)   # lambda L_lambda -> L_nu
    return df


def load_sdss(data_dir):
    """Rankine+20 quasars with good C IV and a 1350 A luminosity."""
    t = Table.read(os.path.join(data_dir, "Rankine20_CIV_HeII_wDelGI_wdr16luminosity.fits"))
    sdss = t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()
    sdss = sdss[(sdss["LOGL1350_dr16"] > 0) & (sdss["good"] == True)]
    return _add_logLnu1450(sdss)


def load_gnirs(data_dir):
    """GNIRS-DQS quasars with good C IV and a 1350 A luminosity."""
    g = pd.read_csv(os.path.join(data_dir, "gnirsdqs_wRankine_wdr16Lum.csv"))
    g = g[(g["good"] == True) & (g["LOGL1350_dr16"].values > 0)]
    return _add_logLnu1450(g)
