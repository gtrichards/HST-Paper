"""How far does the FOS accumulation misclassification reach?

Found on Ton 469 (index 362).  rebinning/read_spec_data._is_cumulative_accum
decided whether an FOS exposure's groups are a RUNNING SUM -- in which case the
last group alone is the whole exposure -- or INDEPENDENT READOUTS, which must be
averaged.  The old test asked whether three consecutive group means happen to
rise.  Independent readouts satisfy that by chance about one time in six, and
when they do, every readout but the last is discarded.

On Ton 469 that cost a factor of 6 in the C IV window: the only exposure
covering C IV has 45 readouts and one was kept, so median S/N read 2.10 where
the proper average gives 13.30 and Sulentic+07 quote 12.9.  The reconstruction
was inadmissible under every component basis and the object sat at manmask 0
with a published Sulentic profile.  GTR: "It has a lot of absorption, but it
looks absolutely nothing like the spectrum shown in the interactive fitter."

The test is now mean(last group)/mean(all groups) against a threshold halfway
between 1.0 (independent) and 2n/(n+1) (a running sum).  THIS SCAN ASKS WHICH
EXPOSURES THE TWO TESTS DISAGREE ABOUT, and -- because that is what decides
whether a measurement moves -- which of those exposures COVER C IV and are in
an object's CHOSEN VISIT.

An exposure flagged here is not necessarily a changed measurement: an object
whose C IV window is covered by several exposures loses only the misclassified
ones, so the effect runs from negligible to severe.  The ranking is therefore by
whether the exposure is in the chosen visit AND covers C IV, with sqrt(groups)
as the S/N that was thrown away.

    python scan_fos_accum.py              # report
    python scan_fos_accum.py --csv        # write pipeline_output/final/fos_accum.csv
"""
import argparse
import csv
import glob
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")

MAST = os.path.join(HERE, "data_v23", "MAST_v23")
QUEUE = os.path.join(HERE, "pipeline_output", "work_queue.csv")
LEDGER = "/Users/gtr/Work/git/HST-Paper/Data/decisions.csv"
STAGE = os.path.join(HERE, "pipeline_output", "chosen_visit", "stage")
OUT = os.path.join(HERE, "pipeline_output", "final", "fos_accum.csv")

CIV = 1549.48


def old_test(f, index=-1):
    """The monotonic test as it stood before 2026-11-08."""
    n = f.shape[0]
    idx = index if index >= 0 else n + index
    if idx <= 0:
        return True
    if idx == 1:
        return np.mean(f[idx]) > np.mean(f[idx - 1])
    return (np.mean(f[idx]) > np.mean(f[idx - 1]) and
            np.mean(f[idx - 1]) > np.mean(f[idx - 2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()

    from rebinning.read_spec_data import _is_cumulative_accum, _cumulative_threshold

    q = {}
    for r in csv.DictReader(open(QUEUE)):
        q[str(r["name_mast_key"]).strip()] = r
    mm = {}
    for r in csv.DictReader(open(LEDGER)):
        o = int(re.match(r"^(\d+)", r["folder"]).group(1))
        mm[o] = str(r["manmask"]).strip()

    rows = []
    for d in sorted(glob.glob(os.path.join(MAST, "*", "FOS"))):
        name = os.path.basename(os.path.dirname(d))
        r = q.get(name)
        if r is None:
            continue
        order = int(r["order"])
        z = float(r["z"])
        obs = CIV * (1.0 + z)
        staged = set()
        sd = os.path.join(STAGE, "%03d_FOS" % order)
        if os.path.isdir(sd):
            staged = {os.path.basename(p).replace("_c1f.fits", "")
                      for p in glob.glob(os.path.join(glob.escape(sd), "*_c1f.fits"))}
        for p1 in sorted(glob.glob(os.path.join(glob.escape(d), "*_c1f.fits"))):
            stem = os.path.basename(p1).replace("_c1f.fits", "")
            p0 = p1.replace("_c1f", "_c0f")
            try:
                f = np.atleast_2d(np.asarray(fits.getdata(p1), float))
                p2 = p1.replace("_c1f", "_c2f")
                e = (np.atleast_2d(np.asarray(fits.getdata(p2), float))
                     if os.path.exists(p2) else None)
                hdr = fits.getheader(p1, 0)
            except Exception:
                continue
            n = f.shape[0]
            if n < 2:
                continue
            # the discriminator is the ERROR ratio (see read_spec_data);
            # err(first)/err(last) is sqrt(n) for a cumulative exposure and 1
            # for independent readouts
            ratio = np.nan
            if e is not None and e.shape[0] >= n:
                e0 = np.nanmedian(np.where(e[0] > 0, e[0], np.nan))
                el = np.nanmedian(np.where(e[n - 1] > 0, e[n - 1], np.nan))
                if np.isfinite(e0) and np.isfinite(el) and el > 0:
                    ratio = e0 / el
            o_cum = bool(old_test(f, -1))
            n_cum = bool(_is_cumulative_accum(f, -1, e, nread=hdr.get("NREAD"),
                                              grndmode=hdr.get("GRNDMODE")))
            covers = False
            lo = hi = np.nan
            if os.path.exists(p0):
                try:
                    w = np.atleast_2d(np.asarray(fits.getdata(p0), float))[-1]
                    wn = w[w > 0]
                    if wn.size:
                        lo, hi = float(np.nanmin(wn)), float(np.nanmax(wn))
                        covers = bool(lo <= obs <= hi)
                except Exception:
                    pass
            try:
                grating = fits.getheader(p1).get("FGWA_ID",
                          fits.getheader(p1).get("OPT_ELEM", ""))
            except Exception:
                grating = ""
            rows.append(dict(order=order, name=name, z=z, manmask=mm.get(order, ""),
                             exposure=stem, grating=grating, groups=n,
                             ratio=(round(ratio, 3) if np.isfinite(ratio) else np.nan),
                             threshold=round(_cumulative_threshold(n), 3),
                             nread=hdr.get("NREAD"), grndmode=str(hdr.get("GRNDMODE","")).strip(),
                             old_cumulative=o_cum, new_cumulative=n_cum,
                             changed=(o_cum != n_cum),
                             covers_civ=covers, in_chosen_visit=(stem in staged),
                             wave_lo=(round(lo, 1) if np.isfinite(lo) else np.nan),
                             wave_hi=(round(hi, 1) if np.isfinite(hi) else np.nan),
                             sn_factor=round(float(np.sqrt(n)), 2)))
    d = pd.DataFrame(rows)

    print("FOS ACCUMULATION SCAN")
    print("  objects with FOS data : %d" % d.order.nunique())
    print("  exposures examined    : %d  (2 or more groups)" % len(d))
    print()
    ch = d[d.changed]
    print("  exposures the two tests DISAGREE about : %d  (%.1f%%)"
          % (len(ch), 100.0 * len(ch) / max(len(d), 1)))
    print("     old said cumulative, now independent (readouts were DISCARDED): %d"
          % int((ch.old_cumulative & ~ch.new_cumulative).sum()))
    print("     old said independent, now cumulative (groups were AVERAGED):    %d"
          % int((~ch.old_cumulative & ch.new_cumulative).sum()))
    print()
    bad = ch[ch.old_cumulative & ~ch.new_cumulative]
    print("  of those discarding readouts:")
    print("     cover C IV                        : %d" % int(bad.covers_civ.sum()))
    print("     cover C IV AND in the chosen visit: %d" % int((bad.covers_civ & bad.in_chosen_visit).sum()))
    hot = bad[bad.covers_civ & bad.in_chosen_visit]
    print()
    print("*** EXPOSURES THAT AFFECT A MEASUREMENT: %d, over %d object(s) ***"
          % (len(hot), hot.order.nunique()))
    if len(hot):
        print("  %-5s %-26s %-5s %-11s %-5s %6s %7s %7s %s"
              % ("order", "name", "mm", "exposure", "grat", "groups", "ratio", "thresh", "S/N lost"))
        for _, x in hot.sort_values(["groups"], ascending=False).iterrows():
            print("  %-5d %-26s %-5s %-11s %-5s %6d %7.3f %7.3f   x%.1f"
                  % (x.order, str(x["name"])[:26], x.manmask, x.exposure, x.grating,
                     x.groups, x.ratio, x.threshold, x.sn_factor))
        print()
        print("  by manmask: %s" % dict(hot.groupby("manmask").order.nunique()))
    if a.csv:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        d.to_csv(OUT, index=False)
        print()
        print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
