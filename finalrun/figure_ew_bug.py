"""
Draw the C IV equivalent-width continuum bug and the one-line fix.

In ica/manual_fix.py, get_CIV_parameters accumulates

    CIV_EW += max(((flux_r[EW][i] - continuum[i]) / continuum[i]) * dlambda, 0)

where EW is the boolean mask selecting 1500-1600 A. `flux_r[EW][i]` therefore
steps across the C IV window, but `continuum[i]` steps from the *start of the
reconstruction*. On the component grid the C IV window begins at index 757, so
the continuum is read at 1260-1344 A and subtracted at 1500-1600 A -- about
240 A adrift. Since the continuum falls to the red, the value used is too high,
too much is subtracted, and the equivalent width comes out low.

The blueshift rides on the same arithmetic: it is the half-flux point of the
cumulative sum, so mis-subtracting the continuum moves it too.

The fix is to index the continuum with the same mask as the flux:
`continuum[EW][i]`.

Four panels: where the continuum is read versus where it is applied; what that
does to the EW integrand; how the cumulative sum and its half-flux point move;
and the size of the effect across every fit adopted so far.
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "pipeline_output")
OUT = os.path.join(OUTDIR, "ew_continuum_bug.png")

# Measured on every adopted fit (see the survey in the session log). NGC 3227 is
# recomputed below, since its adopted fit changed after the survey was run.
SURVEY = [("NGC 5548", 13.1), ("NGC 3516", 17.1), ("NGC 3227", None),
          ("NGC 4151", 20.6), ("NGC 4593", 20.9), ("NGC 6814", 23.3),
          ("NGC 3783", 24.3), ("NGC 4395", 24.7), ("Mrk 1044", 25.6),
          ("NGC 4253", 27.1), ("NGC 7469", 36.4)]

CONT_WINDOWS = ((1445.0, 1465.0), (1700.0, 1705.0))
EW_LO, EW_HI = 1500.0, 1600.0


def ew_series(wave, flux, cont, ew):
    """Cumulative EW exactly as get_CIV_parameters builds it."""
    w, f = wave[ew], flux[ew]
    tot, series = 0.0, [0.0]
    for i in range(len(w)):
        dl = w[i + 1] - w[i] if i + 1 < len(w) else w[i] - w[i - 1]
        tot += max(((f[i] - cont[i]) / cont[i]) * dl, 0)
        series.append(tot)
    j = int(np.abs((tot / 2) - np.array(series)).argmin())
    return tot, np.array(series[1:]), w[j]


def main():
    from ica.manual_fix import ICAManualFixProcessor

    rec = json.load(open(os.path.join(
        OUTDIR, "fit_iterations", "005_NGC_3227_COMBINED", "records",
        "BEST_iter12_troughtrue.json")))
    ov = rec["override"]
    res = ICAManualFixProcessor(
        rebin_path=os.path.join(OUTDIR, "coadd_work"), master_mode=True
    ).fit_with_overrides("NGC 3227_COMBINED", mask_ranges=ov["mask_ranges"],
                         mask_pixels=ov.get("mask_pixels", []),
                         unmask_ranges=ov.get("unmask_ranges", []),
                         unmask_pixels=ov.get("unmask_pixels", []))

    w = np.asarray(res["wave_ica"])
    f = np.asarray(res["flux_ica"])
    c1 = (w >= CONT_WINDOWS[0][0]) & (w <= CONT_WINDOWS[0][1])
    c2 = (w >= CONT_WINDOWS[1][0]) & (w <= CONT_WINDOWS[1][1])
    m, b = np.polyfit(np.concatenate((w[c1], w[c2])),
                      np.concatenate((f[c1], f[c2])), 1)
    cont = w * m + b
    ew = (w >= EW_LO) & (w <= EW_HI)
    n = int(ew.sum())
    i0 = int(np.flatnonzero(ew)[0])

    cont_used = cont[:n]          # what the code reads: indices 0..n-1
    cont_right = cont[ew]         # what it should read

    ew_bug, cum_bug, lam_bug = ew_series(w, f, cont_used, ew)
    ew_fix, cum_fix, lam_fix = ew_series(w, f, cont_right, ew)
    blue = lambda lam: (1549.48 - lam) / 1549.48 * 3e5   # as get_CIV_parameters does

    survey = [(nm, (100 * (ew_fix - ew_bug) / ew_fix) if v is None else v)
              for nm, v in SURVEY]

    fig, ax = plt.subplots(2, 2, figsize=(15.5, 10), constrained_layout=True)

    # (a) where the continuum is read vs where it is applied
    a = ax[0, 0]
    sel = (w >= 1250) & (w <= 1760)
    a.plot(w[sel], f[sel], "-k", lw=1.0, label="ICA reconstruction")
    a.plot(w[sel], cont[sel], "--", color="0.45", lw=1.2, label="fitted continuum")
    for lo, hi in CONT_WINDOWS:
        a.axvspan(lo, hi, color="tab:green", alpha=0.30, lw=0)
    a.axvspan(w[0], w[n - 1], color="tab:red", alpha=0.16, lw=0)
    a.axvspan(EW_LO, EW_HI, color="tab:blue", alpha=0.16, lw=0)
    a.plot(w[:n], cont_used, "-", color="tab:red", lw=2.5)
    a.plot(w[ew], cont_right, "-", color="tab:blue", lw=2.5)
    ytop = np.nanmax(f[sel]) * 0.92
    a.add_patch(FancyArrowPatch((0.5 * (w[0] + w[n - 1]), ytop),
                                (0.5 * (EW_LO + EW_HI), ytop),
                                arrowstyle="-|>", mutation_scale=18,
                                color="tab:red", lw=2))
    a.text(0.5 * (w[0] + w[n - 1]), ytop * 1.04, "continuum read here\n(indices 0-%d, %.0f-%.0f A)"
           % (n - 1, w[0], w[n - 1]), ha="center", va="bottom", fontsize=9, color="tab:red")
    a.text(0.5 * (EW_LO + EW_HI), ytop * 1.04, "but subtracted here\n(C IV, %.0f-%.0f A)"
           % (EW_LO, EW_HI), ha="center", va="bottom", fontsize=9, color="tab:blue")
    a.annotate("continuum fit windows", xy=(1455, cont[c1][0]), xytext=(1380, 8.5),
               fontsize=8, color="tab:green",
               arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.1))
    a.annotate("", xy=(1702, cont[c2][0]), xytext=(1420, 8.2),
               arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.1))
    a.set_xlim(1250, 1760); a.set_ylim(0, np.nanmax(f[sel]) * 1.22)
    a.set_xlabel("Rest wavelength (A)"); a.set_ylabel("Flux (arb.)")
    a.set_title("(a) the index slip: continuum[i] walks from index 0, not from the C IV window")
    a.legend(loc="center left", fontsize=8)

    # (b) what that does to the integrand
    b_ = ax[0, 1]
    integ_used = (f[ew] - cont_used) / cont_used
    integ_right = (f[ew] - cont_right) / cont_right
    b_.plot(w[ew], integ_right, "-", color="tab:blue", lw=1.8,
            label="integrand with aligned continuum")
    b_.plot(w[ew], integ_used, "-", color="tab:red", lw=1.8,
            label="integrand as computed now")
    b_.fill_between(w[ew], integ_used, integ_right, color="tab:orange", alpha=0.55,
                    label="EW lost to the wrong continuum")
    b_.axhline(0, color="0.5", lw=0.8)
    b_.set_xlim(EW_LO, EW_HI)
    b_.set_xlabel("Rest wavelength (A)")
    b_.set_ylabel(r"integrand  $(f-c)/c$")
    b_.set_title("(b) the quantity actually summed, pixel by pixel\n"
                 "the shaded area is the %.1f A of EW that goes missing" % (ew_fix - ew_bug))
    b_.legend(loc="upper left", fontsize=8)

    # (c) the cumulative sum and the half-flux point
    c = ax[1, 0]
    c.plot(w[ew], cum_bug, "-", color="tab:red", lw=2, label="cumulative EW, as computed")
    c.plot(w[ew], cum_fix, "-", color="tab:blue", lw=2, label="cumulative EW, aligned")
    for cum, lam, col, lab in ((cum_bug, lam_bug, "tab:red", "now"),
                               (cum_fix, lam_fix, "tab:blue", "aligned")):
        c.axhline(cum[-1] / 2, color=col, ls=":", lw=1.1)
        c.axvline(lam, color=col, ls="--", lw=1.6)
        c.plot([lam], [cum[-1] / 2], "o", color=col, ms=7)
        c.text(EW_LO + 2, cum[-1] / 2, "%s: %+.1f km/s" % (lab, blue(lam)),
               color=col, fontsize=9, va="bottom")
    c.set_xlim(EW_LO, EW_HI)
    lo_z, hi_z = min(lam_bug, lam_fix) - 1.5, max(lam_bug, lam_fix) + 1.5
    ins = c.inset_axes([0.55, 0.12, 0.42, 0.42])
    zs = (w[ew] >= lo_z) & (w[ew] <= hi_z)
    ins.plot(w[ew][zs], cum_bug[zs], "-", color="tab:red", lw=2)
    ins.plot(w[ew][zs], cum_fix[zs], "-", color="tab:blue", lw=2)
    for cum, lam, col in ((cum_bug, lam_bug, "tab:red"), (cum_fix, lam_fix, "tab:blue")):
        ins.axhline(cum[-1] / 2, color=col, ls=":", lw=1.0)
        ins.axvline(lam, color=col, ls="--", lw=1.4)
        ins.plot([lam], [cum[-1] / 2], "o", color=col, ms=6)
    ins.set_xlim(lo_z, hi_z)
    ins.set_title("half-flux points, %.2f A apart (1 grid pixel)"
                  % abs(lam_fix - lam_bug), fontsize=8)
    ins.tick_params(labelsize=7)
    c.indicate_inset_zoom(ins, edgecolor="0.4")
    c.set_xlabel("Rest wavelength (A)")
    c.set_ylabel("cumulative EW (A)")
    c.set_title("(c) the blueshift is the half-flux point of this curve, so it moves too")
    c.legend(loc="upper left", fontsize=8)

    # (d) size of the effect across the adopted fits
    d = ax[1, 1]
    names = [s[0] for s in survey]; vals = [s[1] for s in survey]
    cols = ["tab:orange" if nm == "NGC 3227" else "tab:blue" for nm in names]
    d.barh(range(len(names)), vals, color=cols)
    d.set_yticks(range(len(names))); d.set_yticklabels(names, fontsize=9)
    d.invert_yaxis()
    med = float(np.median(vals))
    d.axvline(med, color="k", ls="--", lw=1.4)
    d.text(med, len(names) - 0.4, " median %.1f%%" % med, fontsize=9, va="top")
    for i, v in enumerate(vals):
        d.text(v + 0.4, i, "%.1f%%" % v, va="center", fontsize=8)
    d.set_xlim(0, max(vals) * 1.18)
    d.set_xlabel("EW understated by")
    d.set_title("(d) every adopted fit is affected; the size tracks the continuum slope")

    fig.suptitle("C IV equivalent width: continuum indexed from the wrong place "
                 "(ica/manual_fix.py, get_CIV_parameters)\n"
                 "fix: continuum[i] -> continuum[EW][i]   "
                 "   NGC 3227 example: EW %.1f -> %.1f A,   blueshift %+.1f -> %+.1f km/s"
                 % (ew_bug, ew_fix, blue(lam_bug), blue(lam_fix)), fontsize=12)
    fig.savefig(OUT, dpi=130)
    print("EW  %.2f -> %.2f  (%.1f%% low)" % (ew_bug, ew_fix, 100 * (ew_fix - ew_bug) / ew_fix))
    print("blue %+.1f -> %+.1f km/s" % (blue(lam_bug), blue(lam_fix)))
    print("-> %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
