"""
The case for a redshift correction on Z 291-51 (index 32), in one figure.

Top: the two anchor lines, C III] and Mg II, with the data and the ICA model at
the catalogue redshift and at the trial value that zeroes Mg II. A redshift
error shows as the model sitting to one side of the data at BOTH anchors and
moving onto them together; a model-shape problem does not.

Bottom: the anchor cross-correlation offsets against the trial shift (from
trial_z.py), with the C IV blueshift alongside, and the range of redshifts NED
lists for the object.
"""
import csv, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from ica.manual_fix import ICAManualFixProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pipeline_output", "fit_iterations", "032_Z_291-51_STIS")
WORK = os.path.join(HERE, "pipeline_output", "coadd_work")
REBIN = "/Users/gtr/Dropbox/HST/Pratsos/RebinnedSpec_master"
STEM = "Z 291-51_STIS"
C = 299792.458
Z_CAT = 0.027870
DV_TRIAL = 160.0                      # where Mg II zeroes
NED = (0.027838, 0.028243)            # range GTR saw in NED
REC = os.path.join(OUT, "records", "BEST_iter04_gtr_absorbers.json")

ov = json.load(open(REC))["override"]

def fit_at(dv):
    if dv == 0:
        proc = ICAManualFixProcessor(rebin_path=REBIN, master_mode=True)
        return proc.fit_with_overrides(STEM, mask_ranges=ov["mask_ranges"],
                                       mask_pixels=ov.get("mask_pixels", []), comps_use=ov.get("comps_use"))
    zt = (1 + Z_CAT) * (1 + dv / C) - 1
    fac = (1 + Z_CAT) / (1 + zt)
    h = fits.open(os.path.join(REBIN, STEM + ".fits"))
    h[1].data["Rest-frame Wavelength"] = h[1].data["Rest-frame Wavelength"] * fac
    tmp = STEM + "__zfig"
    h.writeto(os.path.join(WORK, tmp + ".fits"), overwrite=True)
    try:
        proc = ICAManualFixProcessor(rebin_path=WORK, master_mode=True)
        r = proc.fit_with_overrides(tmp, mask_ranges=[[a * fac, b * fac] for a, b in ov["mask_ranges"]],
                                    mask_pixels=[p * fac for p in ov.get("mask_pixels", [])], comps_use=ov.get("comps_use"))
    finally:
        os.remove(os.path.join(WORK, tmp + ".fits"))
    return r, zt

r0 = fit_at(0)
r1, zt = fit_at(DV_TRIAL)

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.22)
panels = [(fig.add_subplot(gs[0, 0]), 1908.73, (1880, 1940), "C III] 1909"),
          (fig.add_subplot(gs[0, 1]), 2798.75, (2760, 2840), "Mg II 2799")]
for ax, lam, (lo, hi), title in panels:
    for r, col, lab in ((r0, "tab:red", "model at catalogue z = %.6f" % Z_CAT),
                        (r1, "tab:blue", "model at z = %.6f  (+%d km/s)" % (zt, DV_TRIAL))):
        w = np.asarray(r["wave_arb"]); f = np.asarray(r["flux_arb"])
        if r is r0:
            s = (w >= lo) & (w <= hi); ax.plot(w[s], f[s], color="0.35", lw=1.0, label="data (rest frame at catalogue z)")
        wi = np.asarray(r["wave_ica"]); fi = np.asarray(r["flux_ica"]); s = (wi >= lo) & (wi <= hi)
        if r is r1:
            # put the trial model back on the catalogue-z wavelength scale so the shift is visible against the same data
            wi = wi / ((1 + Z_CAT) / (1 + zt))
            s = (wi >= lo) & (wi <= hi)
        ax.plot(wi[s], fi[s], color=col, lw=1.6, label=lab)
    ax.axvline(lam, color="k", ls="--", lw=0.8)
    ax.set_title(title); ax.set_xlabel("rest wavelength (Å), catalogue z"); ax.set_ylabel("flux (arb.)")
    ax.legend(fontsize=8, loc="upper right")
    note = ("data redward of the red model;\nthe +160 km/s model moves about halfway onto it\n(C III] is a blend -- part of the offset is model shape)"
            if lam < 2000 else "data redward of the red model;\nthe +160 km/s model lands on it")
    ax.text(0.02, 0.95, note, transform=ax.transAxes, va="top", fontsize=8.5)

ax = fig.add_subplot(gs[1, :])
rows = list(csv.DictReader(open(os.path.join(OUT, "records", "trial_z.csv"))))
def col(name):
    k = [c for c in rows[0] if name.lower() in c.lower()][0]
    return np.array([float(r[k]) for r in rows])
dv = col("dv"); ciii = col("CIII"); mgii = col("MgII"); blue = col("blue")
ax.axhline(0, color="k", lw=0.8)
ax.plot(dv, ciii, "o-", color="tab:green", label="C III] cross-correlation offset (model − data)")
ax.plot(dv, mgii, "s-", color="tab:purple", label="Mg II cross-correlation offset")
ned_dv = [C * (z - Z_CAT) / (1 + Z_CAT) for z in NED]
ax.axvspan(ned_dv[0], ned_dv[1], color="gold", alpha=0.3, label="range of NED redshifts (%.6f–%.6f)" % NED)
ax.axvline(DV_TRIAL, color="tab:blue", ls=":", lw=1.2)
for q in (69, 138, 207, 276): ax.axvline(q, color="0.8", lw=0.6, zorder=0)
ax.set_xlabel("trial redshift shift Δv relative to catalogue z (km/s); grey lines = 69 km/s grid quanta")
ax.set_ylabel("anchor offset (km/s)")
ax2 = ax.twinx(); ax2.plot(dv, blue, "d--", color="tab:red", label="C IV blueshift (right axis)")
ax2.set_ylabel("C IV blueshift (km/s)", color="tab:red")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
ax.set_title("both anchors lean the same way: Mg II zeroes at Δv ≈ +160 km/s; C III] closes only ~0.7 km/s per km/s and would need ~+350")
fig.suptitle("Z 291-51 (index 32, row 197): is the catalogue redshift low?", fontsize=13)
png = os.path.join(OUT, "figure_redshift_case.png")
fig.savefig(png, dpi=130, bbox_inches="tight"); print("->", png)
print("model at +%d: blue=%.1f ew=%.2f ; at catalogue: blue=%.1f ew=%.2f" % (DV_TRIAL, r1["civ_blue"], r1["civ_ew"], r0["civ_blue"], r0["civ_ew"]))
