"""Talk figures: what goes wrong in a C IV co-add, shown on the real inputs.

GTR, 2026-11-17: "something that I can show in a talk that says 'this is why it
was so hard for a student to do -- even for Claude to do without many
iterations -- but this is why I am confident that we have it right' ... with
the individual spectra shown."  Modelled on the top panel of Figure 3
(examples_fullprocess_J09270.pdf): every exposure in its own colour with its
error band.  Here the range is narrowed to C IV and the error arrays get a
panel of their own, because the error array is where most of the trouble was.

Figure A, one column per object, three rows:
    exposures      native F_lambda, coloured by grating and epoch, error band
    errors         err/flux per exposure, log scale -- the panel that explains it
    co-add         the product before (grey) and after (black) the sqrt(N)
                   error, with the measured blueshift / EW old -> new
Figure B, two single-exposure failure modes:
    the groups of one FOS exposure with NREAD=19 (flux flat, error falling)
    the error array of index 233 with its near-zero pixels

Writes NEW files under pipeline_output/final/talk/.  Nothing is overwritten.
"""
import glob, json, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, "/Users/gtr/Work/git/richardsplot")
import richardsplot  # noqa: house style -- serif, ticks in, minor ticks
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from rebinning import lower_res_rebin as LR

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "pipeline_output", "chosen_visit", "stage")
REBIN = os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
ARCH = glob.glob(os.path.join(HERE, "pipeline_output", "archive_pre_sqrtn_*"))[-1]
OUT = os.path.join(HERE, "pipeline_output", "final", "talk")
Q = pd.read_csv(os.path.join(HERE, "pipeline_output", "work_queue.csv")).set_index("order")
D = pd.read_csv(os.path.join(HERE, "pipeline_output", "final", "sqrtn_diff_graded.csv")).set_index("order")
CIV = 1549.48
#: FOS FGWA_ID codes -> the official grating names.  GTR: "Please use the official
#: grating names (G190H, for example) not your shorthand (H19)."
GRATING = {"H13": "G130H", "H19": "G190H", "H27": "G270H", "H40": "G400H", "H57": "G570H",
           "H78": "G780H", "L15": "G160L", "L65": "G650L", "PRI": "PRISM"}
def grating_name(code):
    code = str(code).strip()
    return GRATING.get(code, code)


def tex(s):
    return s.replace("_", r"\_").replace("+", r"$+$").replace("-", r"$-$")


def native_labels(index, inst):
    """(label, wmin, wmax) for every native exposure / segment, from the headers."""
    d = os.path.join(STAGE, "%03d_%s" % (index, inst)); out = []
    for p in sorted(glob.glob(d + "/*_c1f.fits")):
        h = fits.getheader(p, 0); W = np.atleast_2d(fits.getdata(p.replace("_c1f", "_c0f")))[-1]; W = W[W > 0]
        out.append(("%s  %s" % (grating_name(h.get("FGWA_ID", "?")), str(h.get("DATE-OBS", "?")).strip()), W.min(), W.max()))
    for p in sorted(glob.glob(d + "/*_x1d.fits")):
        h = fits.getheader(p, 0); h1 = fits.getheader(p, 1); dd = fits.getdata(p, 1)
        for row in dd:
            w = np.asarray(row["WAVELENGTH"], float); e = np.asarray(row["ERROR"], float); w = w[(w > 0) & (e > 0)]
            if w.size < 20: continue
            out.append(("%s/%s  %s" % (str(h.get("OPT_ELEM", "?")).strip(), str(row["SEGMENT"]).strip(), str(h1.get("DATE-OBS", "?"))[:10]), w.min(), w.max()))
    return out


def native_segments(index, inst):
    """(label, w_obs, flux, err) for every native exposure / segment, raw from the archive."""
    d = os.path.join(STAGE, "%03d_%s" % (index, inst)); out = []
    for p in sorted(glob.glob(d + "/*_c1f.fits")):
        h = fits.getheader(p, 0)
        W = np.atleast_2d(fits.getdata(p.replace("_c1f", "_c0f")))[-1]
        Fx = np.atleast_2d(fits.getdata(p))[-1]; Ex = np.atleast_2d(fits.getdata(p.replace("_c1f", "_c2f")))[-1]
        out.append(("%s  %s" % (grating_name(h.get("FGWA_ID", "?")), str(h.get("DATE-OBS", "?")).strip()), W, Fx, Ex))
    for p in sorted(glob.glob(d + "/*_x1d.fits")):
        h = fits.getheader(p, 0); h1 = fits.getheader(p, 1); dd = fits.getdata(p, 1)
        for row in dd:
            out.append(("%s/%s  %s" % (str(h.get("OPT_ELEM", "?")).strip(), str(row["SEGMENT"]).strip(), str(h1.get("DATE-OBS", "?"))[:10]),
                        np.asarray(row["WAVELENGTH"], float), np.asarray(row["FLUX"], float), np.asarray(row["ERROR"], float)))
    return out


def exposures_on_lattice(index, inst, base, z):
    """Every exposure row AS THE PIPELINE READER DELIVERS IT, on the lattice, with
    its grating/segment label recovered by matching coverage to the native files.

    This replaced a version that re-read the raw files itself.  For index 220
    that version drew G140L and both G230L exposures across C IV, when the
    reader delivers exactly one G230L stripe there and nothing else -- the
    G140L red end is dead and the second G230L exposure's central wavelength
    does not reach C IV.  Draw what the co-add saw, not what the archive holds.
    Fluxes are continuum-normalised by the reader's own step (as Figure 3,
    middle panel); physical units are not recoverable after it, so the top
    row is in those units.
    """
    sys.path.insert(0, HERE)
    from figure_coadd_nanmedian import place
    lam, F, E, M = place(base, os.path.join(STAGE, "%03d_%s" % (index, inst)), z, inst=inst)
    nat = native_labels(index, inst); used = set(); out = []
    for i in range(F.shape[0]):
        ok = np.isfinite(F[i]) & (E[i] > 0)
        if ok.sum() < 20: continue
        lo, hi = lam[ok].min(), lam[ok].max()
        # the native segment with the LARGEST OVERLAP with this row (nearest-endpoint
        # matching mislabelled a G230L/NUVA row as NUVC and a G160L row as G190H)
        def overlap(t): return max(0.0, min(t[2], hi) - max(t[1], lo)) - (1e6 if id(t) in used else 0)
        best = max(nat, key=overlap) if nat else ("row %d" % i, lo, hi)
        used.add(id(best))
        out.append((best[0], lam[ok] / (1 + z), F[i][ok], E[i][ok]))
    return out


def on_lattice(w, f, e, name, z):
    """One native exposure rebinned onto the project's 69 km/s lattice, in
    physical units -- the same rebinner coadd.rebin uses, so what is drawn is
    what the co-add actually sees per exposure (COS at native resolution has
    err/flux ~ 100 per pixel; on the lattice it is comparable to FOS)."""
    # the reader's definition of a usable pixel: finite, positive error, and flux
    # NOT exactly zero (a dead pixel, not a measurement of zero -- see
    # read_spec_data); and ASCENDING wavelength, which FOS files are not natively
    # and which the rebinner silently requires
    g = np.isfinite(f) & np.isfinite(e) & (e > 0) & (w > 0) & (f != 0)
    if g.sum() < 20:
        return None
    o = np.argsort(w[g]); w, f, e = w[g][o], f[g][o], e[g][o]
    try:
        nw, nf, ne, nm = LR.HSTLowResRebin(w, f, e, np.zeros(len(w)), name, z)
    except Exception:
        return None
    ok = np.isfinite(nf) & np.isfinite(ne) & (ne > 0)
    return nw[ok] / (1 + z), nf[ok], ne[ok]


def ylim_pct(ax, ys, lo=1.0, hi=99.5, log=False):
    v = np.concatenate([np.asarray(y, float) for y in ys if len(y)]) if ys else np.array([])
    v = v[np.isfinite(v)]
    if log: v = v[v > 0]
    if v.size < 10: return
    a, b = np.percentile(v, lo), np.percentile(v, hi)
    pad = (b / a) ** 0.15 if log else 0.08 * (b - a)
    ax.set_ylim((a / pad, b * pad) if log else (a - pad, b + pad))


def product(path):
    t = Table.read(path)
    w = np.asarray(t["Rest-frame Wavelength"], float); f = np.asarray(t["Coadded Flux (Arbitrary Units)"], float)
    e = np.asarray(t["Coadded Flux Errors"], float); m = np.asarray(t["Bad Pixel Mask"], float)
    g = (m == 0) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    return w, f, e, g


def norm_window(w, f, lo=1620.0, hi=1700.0):
    k = (w >= lo) & (w <= hi) & np.isfinite(f)
    m = np.nanmedian(f[k]) if k.sum() > 5 else np.nanmedian(f[np.isfinite(f)])
    return f / m if m and np.isfinite(m) and m != 0 else f


def column(axs, index, inst, stem, lo=1420.0, hi=1720.0):
    z = float(np.nanmedian(np.asarray(Table.read(os.path.join(REBIN, stem + ".fits"))["Redshift"], float)))
    base = stem.rsplit("_", 1)[0]
    ex = exposures_on_lattice(index, inst, base, z)
    labels = sorted(set(l for l, *_ in ex))
    cmap = plt.get_cmap("tab10"); col = {l: cmap(i % 10) for i, l in enumerate(labels)}
    counts = {l: sum(1 for ll, *_ in ex if ll == l) for l in labels}
    ax0, ax1, ax2, ax3, ax4 = axs
    fl, rr = [], []
    # what the ARCHIVE holds that the reader did not deliver: every native segment,
    # rebinned onto the lattice, drawn in grey wherever no delivered row covers it
    # (the reader trims each end inward while >20 of 65 pixels have S/N <= 0, and
    # drops flux == 0 pixels -- cut_edge_pix.py / read_spec_data.py)
    delivered = {}
    for lab, rr_, f_, e_ in ex:
        delivered.setdefault(lab, []).append(rr_)
    n_grey = 0; grey_labels = []
    for lab, w_, f_, e_ in native_segments(index, inst):
        got = on_lattice(w_, f_, e_, "native", z)
        if got is None: continue
        rw, rf, re_ = got
        cov = np.zeros(len(rw), bool)
        for rr_ in delivered.get(lab, []):
            cov |= (rw >= rr_.min() - 0.5) & (rw <= rr_.max() + 0.5)
        k = (~cov) & (rw >= lo) & (rw <= hi) & (re_ > 0)
        if k.sum() < 20: continue
        ax1.plot(rw[k], np.abs(rf[k]) / re_[k], color="0.55", lw=0.5, alpha=0.8, zorder=1); rr.append(np.abs(rf[k]) / re_[k])
        n_grey += 1
        grey_labels.append("%s %.0f--%.0f (S/N %.2f)" % (tex(lab.split("  ")[0]), rw[k].min(), rw[k].max(), np.nanmedian(np.abs(rf[k]) / re_[k])))
    if n_grey:
        from matplotlib.lines import Line2D
        ax1.legend([Line2D([], [], color="0.55", lw=1)],
                   ["in the archive, NOT delivered to the co-add (reader edge-trim):\n" + "\n".join(grey_labels)],
                   fontsize=6.3, loc="lower left", frameon=True, framealpha=0.9, edgecolor="none")
    for lab, r, f, e in ex:
        k = (r >= lo) & (r <= hi)
        if k.sum() < 5: continue
        ax0.fill_between(r[k], f[k] - e[k], f[k] + e[k], color=col[lab], alpha=0.20, lw=0)
        ax0.plot(r[k], f[k], color=col[lab], lw=0.7, alpha=0.95)
        ax1.plot(r[k], np.abs(f[k]) / e[k], color=col[lab], lw=0.7, alpha=0.95)
        fl.append(f[k]); rr.append(np.abs(f[k]) / e[k])
    for l in labels:
        ax0.plot([], [], color=col[l], lw=2.5, label=r"%s  ($\times$%d)" % (tex(l), counts[l]))
    ax0.legend(fontsize=7, frameon=False, loc="best", handlelength=1.2)
    ylim_pct(ax0, fl); ax1.set_yscale("log"); ylim_pct(ax1, rr, log=True)
    ax1.axhline(1.0, color="0.5", lw=0.6, ls=":")
    # the co-add: flux (identical before/after) with the numbers, then the error before/after
    wa, fa, ea, ga = product(os.path.join(ARCH, "rebin", stem + ".fits"))
    wb, fb, eb, gb = product(os.path.join(REBIN, stem + ".fits"))
    ka = ga & (wa >= lo) & (wa <= hi); kb = gb & (wb >= lo) & (wb <= hi)
    ax2.plot(wb[kb], fb[kb], "k-", lw=0.7); ylim_pct(ax2, [fb[kb]])
    r = D.loc[index]
    ax2.text(0.98, 0.95, r"blueshift %+.0f $\rightarrow$ %+.0f km\,s$^{-1}$" "\n" r"EW %.1f $\rightarrow$ %.1f \AA" "\n" r"$D_{\rm CIV}$ %.3f $\rightarrow$ %.3f"
             % (r.blue_before, r.blue_after, r.ew_before, r.ew_after, r.D_old, r.D_new),
             transform=ax2.transAxes, va="top", ha="right", fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    ax3.plot(wa[ka], np.abs(fa[ka]) / ea[ka], color="0.55", lw=0.9, label="before: single-exposure error")
    ax3.plot(wb[kb], np.abs(fb[kb]) / eb[kb], color="k", lw=0.9, label=r"after: $1/\sqrt{\sum w}$")
    ax3.set_yscale("log"); ylim_pct(ax3, [np.abs(fa[ka]) / ea[ka], np.abs(fb[kb]) / eb[kb]], log=True)
    ax3.legend(fontsize=7, frameon=False, loc="best")
    # the best single exposure (highest median S/N in C IV) against the co-add
    best = None
    for lab, rr_, f_, e_ in ex:
        kk = (rr_ >= 1500) & (rr_ <= 1600)
        if kk.sum() < 20: continue
        sn = float(np.nanmedian(np.abs(f_[kk]) / e_[kk]))
        if best is None or sn > best[0]: best = (sn, lab, rr_, f_, e_)
    if best is not None:
        sn, lab, rb, fbx, ebx = best
        kb2 = (rb >= lo) & (rb <= hi)
        ax4.plot(rb[kb2], norm_window(rb, fbx)[kb2], color="C3", lw=0.8, alpha=0.9,
                 label="best single exposure: %s (C IV S/N %.1f)" % (tex(lab), sn))
    kc = gb & (wb >= lo) & (wb <= hi)
    snc = float(np.nanmedian(np.abs(fb[gb & (wb >= 1500) & (wb <= 1600)]) / eb[gb & (wb >= 1500) & (wb <= 1600)]))
    ax4.plot(wb[kc], norm_window(wb, np.where(gb, fb, np.nan))[kc], color="k", lw=0.9,
             label="the co-add (C IV S/N %.1f)" % snc)
    ylim_pct(ax4, [norm_window(wb, np.where(gb, fb, np.nan))[kc]])
    _y0, _y1 = ax4.get_ylim(); ax4.set_ylim(_y0, _y1 + 0.35 * (_y1 - _y0))   # headroom for the legend
    ax4.legend(fontsize=7, frameon=False, loc="upper left")
    for ax in axs:
        ax.axvspan(1500, 1600, color="#e8eef7", zorder=0); ax.axvline(CIV, color="0.6", lw=0.5, ls=":")
        ax.set_xlim(lo, hi)
    ax0.set_title(r"%s  (index %d, %s)" % (tex(str(Q.loc[index, "name_pub"])), index, inst), fontsize=10)


def figure_a():
    cases = [(70, "FOS", "Mrk 1502_FOS"), (372, "FOS", "NGC2841UB3_FOS"), (220, "COS", "2MASS J10215235+4645158_COS")]
    fig, axs = plt.subplots(5, len(cases), figsize=(5.2 * len(cases), 15.2), sharex="col")
    for j, (i, inst, stem) in enumerate(cases):
        column(axs[:, j], i, inst, stem)
    axs[0, 0].set_ylabel(r"exposures (normalised)")
    axs[1, 0].set_ylabel(r"S/N per pixel (each exposure)")
    axs[2, 0].set_ylabel(r"co-added $F_\lambda$ (arbitrary)")
    axs[3, 0].set_ylabel(r"co-added S/N per pixel")
    axs[4, 0].set_ylabel(r"normalised to 1620--1700 \AA")
    for ax in axs[4, :]: ax.set_xlabel(r"Rest-frame Wavelength (\AA)")
    fig.suptitle(r"Rows: the exposures the co-add saw; their S/N per pixel; the co-add; the co-add's S/N before and after $\sqrt{N}$; the best single exposure against the co-add", fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p = os.path.join(OUT, "talk_A_exposures_errors_coadd.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


def figure_b():
    """Two defects that live inside a SINGLE FOS file, before any co-adding.

    Left + middle: an FOS file taken with NREAD = 19 holds 19 readouts, and each
    one is the running sum of everything before it (GRNDMODE = SPECTROSCOPY).  The
    last readout IS the exposure.  Treating the 19 as independent spectra and
    averaging them keeps the right flux (left) but quotes an error ~ sqrt(19) too
    small (middle).  Right: the first six diodes of a red-Digicon G190H exposure,
    below the Handbook's 1590 A start of that mode, carry errors 1e-7 of the rest.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15.6, 4.6))
    p = os.path.join(STAGE, "070_FOS", "y26c0103t_c1f.fits")
    F = np.atleast_2d(fits.getdata(p)); E = np.atleast_2d(fits.getdata(p.replace("_c1f", "_c2f")))
    W = np.atleast_2d(fits.getdata(p.replace("_c1f", "_c0f")))[-1]; z = 0.059756; r = W / (1 + z)
    n = F.shape[0]; k = (r > 1525) & (r < 1720)
    # LEFT: first readout vs last readout -- same spectrum, different noise
    axs[0].plot(r[k], F[0][k] * 1e14, color="C0", lw=0.6, alpha=0.9, label="readout 1 of 19 (first group)")
    axs[0].plot(r[k], F[-1][k] * 1e14, color="k", lw=0.7, label="readout 19 of 19 (last group = the exposure)")
    axs[0].plot(r[k], F.mean(axis=0)[k] * 1e14 + 1.2, color="C3", lw=0.6, alpha=0.9, label="mean of the 19 (offset +1.2): same spectrum")
    ylim_pct(axs[0], [F[0][k] * 1e14, F[-1][k] * 1e14 + 1.2])
    axs[0].legend(fontsize=7, frameon=False, loc="upper right")
    axs[0].set_title("Mrk 1502, FOS G190H y26c0103t: NREAD = 19 cumulative readouts -- FLUX", fontsize=9)
    axs[0].set_ylabel(r"$F_\lambda$ ($10^{-14}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)")
    # MIDDLE: the quoted error of readout k, relative to the last, against 1/sqrt(k/19)
    kk = np.arange(1, n + 1)
    ratio = np.array([np.nanmedian(E[g][k] / E[-1][k]) for g in range(n)])
    axs[1].plot(kk, ratio, "ko", ms=5, label="median error of readout $k$ / error of readout 19")
    axs[1].plot(kk, np.sqrt(n / kk), "-", color="C3", lw=1.0, label=r"$\sqrt{19/k}$: what accumulating counts predicts")
    axs[1].axhline(np.nanmedian(E.mean(axis=0)[k] / E[-1][k]), color="C0", ls="--", lw=1.0,
                   label="mean of the 19 error arrays / readout 19 = %.2f" % np.nanmedian(E.mean(axis=0)[k] / E[-1][k]))
    axs[1].axhline(1.0, color="0.6", lw=0.5, ls=":")
    axs[1].set_xlabel("readout $k$"); axs[1].set_ylabel("error relative to the last readout")
    axs[1].set_xlim(0, n + 1); axs[1].set_ylim(0.8, 4.8)
    axs[1].legend(fontsize=7, frameon=False, loc="upper right")
    axs[1].set_title("the same file -- ERROR: each readout is the running sum, so only the last is the exposure", fontsize=9)
    axs[1].text(0.97, 0.56, "rule adopted:\nGRNDMODE = SPECTROSCOPY, NREAD = n: keep the last group\nNREAD = 1 with several groups: independent readouts, average",
                transform=axs[1].transAxes, fontsize=7, va="top", ha="right")
    # RIGHT: index 233's NATIVE exposure: six pixels quoted at 1e-7 of the real error
    q = os.path.join(STAGE, "233_FOS", "y29c0302t_c2f.fits")
    E2 = np.atleast_2d(fits.getdata(q))[-1]; W2 = np.atleast_2d(fits.getdata(q.replace("_c2f", "_c0f")))[-1]
    z2 = float(np.nanmedian(np.asarray(Table.read(glob.glob(os.path.join(REBIN, "*J11243917*"))[0])["Redshift"], float)))
    r2 = W2 / (1 + z2); k2 = (E2 > 0) & (W2 > 0)
    axs[2].plot(r2[k2], E2[k2], "k-", lw=0.6)
    med = np.nanmedian(E2[k2]); axs[2].axhline(0.01 * med, color="firebrick", ls="--", lw=0.9, label=r"1\% of the median error (per-exposure guard)")
    bad = k2 & (E2 < 0.01 * med); axs[2].plot(r2[bad], E2[bad], "o", color="firebrick", ms=5, mfc="none", label="the first six diodes: error $10^{-7}$ of the rest")
    axs[2].axvline(1590.0 / (1 + z2), color="C0", lw=1.0, ls="-.", label=r"1590 \AA\ observed: Handbook start of G190H on the red Digicon")
    axs[2].axvspan(r2[k2].min() - 5, 1590.0 / (1 + z2), color="#dbe7f5", zorder=0)
    axs[2].set_yscale("log"); axs[2].set_ylabel("error, one exposure")
    axs[2].set_title("PG 1121+422 (index 233), G190H y29c0302t, red Digicon", fontsize=9)
    axs[2].legend(fontsize=7, frameon=False, loc="center", bbox_to_anchor=(0.55, 0.42))
    axs[2].set_xlim(r2[k2].min() - 5, r2[k2].max() + 5)
    for ax in (axs[0], axs[2]):
        ax.set_xlabel(r"Rest-frame Wavelength (\AA)"); ax.axvspan(1500, 1600, color="#e8eef7", zorder=0)
    fig.tight_layout()
    p = os.path.join(OUT, "talk_B_single_exposure_modes.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


def figure_c(lo=1490.0, hi=1540.0):
    """Mrk 1502's own exposures co-added four ways, on the project's lattice.

    Emulated with figure_coadd_nanmedian.place / weights_of / coadd, which mirror
    coadd.rebin up to the morph (per-exposure continuum normalisation, 1/sigma^2
    weights, weighted median of flux and error).  The REAL adopted product is
    overlaid, rescaled by one constant over 1490-1497 A where only G130H
    contributes, as the check that the emulation of (3) is faithful.

    What each treatment shows, all on the same exposures:
      (1) plain mean               the G190H ramp junk and its zero-error pixels both get in
      (2) 1/sigma^2 median, no guard   the first six G190H pixels carry errors 1e-7 of
                                   their median (weight ~1e14): six pixels outvote
                                   fourteen good G130H exposures at 1498-1500 A
      (3) this pipeline            the error floor masks those pixels AFTER the co-add:
                                   the hijack is gone, but so are 14 exposures' worth of
                                   data there -- a six-pixel hole.  The 1514.6 A spike
                                   (first G190H-only pixel, on the ramp) remains.
      (4) proposed                 the floor applied PER EXPOSURE before co-adding, and
                                   the first 25 A of each G190H excluded: no hole, no
                                   spike, an honest gap where only ramp data exist.
    """
    sys.path.insert(0, HERE)
    from figure_coadd_nanmedian import place, weights_of, coadd
    z = 0.059756
    lam, F, E, M = place("Mrk 1502", os.path.join(STAGE, "070_FOS"), z, inst="FOS")
    r = lam / (1 + z)
    ends = np.array([r[np.isfinite(F[i])].max() if np.isfinite(F[i]).any() else np.nan for i in range(F.shape[0])])
    starts = np.array([r[np.isfinite(F[i])].min() if np.isfinite(F[i]).any() else np.nan for i in range(F.shape[0])])
    is130 = ends < 1530; is190 = (starts > 1470) & (starts < 1510)
    k = (r >= lo) & (r <= hi)

    def wmed(Fm, Em):
        W = weights_of(Em, M, lam, z); return coadd(Fm, Em, W, True)
    def floor_after(f, e):
        """coadd.py's guard: mask co-added pixels whose error is below 1 per cent of the median."""
        f = f.copy(); e = e.copy(); med = np.nanmedian(e[np.isfinite(e) & (e > 0)])
        bad = np.isfinite(e) & (e < 0.01 * med); f[bad] = np.nan; e[bad] = np.nan; return f, e
    def floor_per_exposure(Em):
        """the same guard applied to each exposure BEFORE co-adding: weight 0 there."""
        E2 = Em.copy()
        for i in range(E2.shape[0]):
            g = np.isfinite(E2[i]) & (E2[i] > 0)
            if g.sum() > 20:
                med = np.nanmedian(E2[i][g]); E2[i][g & (E2[i] < 0.01 * med)] = 0.0
        return E2

    f1 = np.nanmean(F, axis=0)                          # (1)
    f2, e2 = wmed(F, E)                                 # (2) no guard
    f3, e3 = floor_after(*wmed(F, E))                   # (3) this pipeline's guard
    F4 = F.copy(); E4 = floor_per_exposure(E)           # (4) guard per exposure + Handbook range
    HB = 1590.0 / (1 + z)                                # FOS IHB v6 Table 1-3: G190H on the red Digicon begins at 1590 A (diode 503)
    for i in np.where(is190)[0]:
        F4[i, r < HB] = np.nan; E4[i, r < HB] = 0.0
    f4, e4 = wmed(F4, E4)
    # the real product, rescaled where only G130H contributes
    t = Table.read(os.path.join(REBIN, "Mrk 1502_FOS.fits"))
    pw = np.asarray(t["Rest-frame Wavelength"], float); pf = np.asarray(t["Coadded Flux (Arbitrary Units)"], float)
    pm = np.asarray(t["Bad Pixel Mask"], float); pe = np.asarray(t["Coadded Flux Errors"], float)
    ok = (pm == 0) & np.isfinite(pf) & (pe > 0)
    ref = (pw >= 1490) & (pw <= 1497)
    scale = np.nanmedian(f3[(r >= 1490) & (r <= 1497)]) / np.nanmedian(pf[ok & ref])
    pk = (pw >= lo) & (pw <= hi)

    fig, axs = plt.subplots(5, 1, figsize=(9.5, 15.5), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.3, 1, 1, 1, 0.9]))
    ax0, axA, axB, axC, ax2 = axs
    for i in range(F.shape[0]):
        c = "C0" if is130[i] else ("C1" if is190[i] else "0.6"); g = np.isfinite(F[i]) & k
        ax0.plot(r[g], F[i][g], color=c, lw=0.6, alpha=0.8)
    ax0.plot([], [], color="C0", lw=2.5, label=r"G130H, 13/02/94 ($\times$14): ends at 1515.2 \AA")
    ax0.plot([], [], color="C1", lw=2.5, label=r"G190H, 14/09/94 ($\times$3), red Digicon: array begins at 1498.6 \AA; Handbook range begins at 1500.3 \AA\ (1590 obs)")
    ax0.set_ylim(0.3, 2.6); ax0.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax0.set_ylabel("exposures (norm.)"); ax0.set_title("Mrk 1502 (I Zw 1), index 70: the same exposures, three rules for combining them", fontsize=11)

    YL = (0.5, 2.6)
    def rule(ax, text):
        ax.text(0.012, 0.95, text, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(fc="white", ec="0.7", alpha=0.95, pad=3))
    # A: the naive combinations, two lines
    axA.plot(r[k], f1[k], color="C3", lw=0.9, label="plain mean of the exposures")
    axA.plot(r[k], f2[k], color="C1", lw=0.9, label=r"$1/\sigma^2$-weighted median, no guard")
    axA.set_ylim(*YL); axA.legend(fontsize=7.5, frameon=False, loc="upper right"); axA.set_ylabel("co-add (norm.)")
    rule(axA, "RULE: combine every pixel as delivered.\nRESULT: six pixels with quoted error $10^{-7}\\times$ median carry weight $10^{14}$\nand outvote fourteen good exposures at 1498--1500 \\AA; the ramp junk enters at 1514.6.")
    # B: this pipeline
    axB.plot(pw[pk & ok], pf[pk & ok] * scale, color="0.75", lw=4, alpha=0.9, label="the adopted product (rescaled)")
    axB.plot(r[k], f3[k], color="k", lw=1.1, label="emulation of the pipeline's rule")
    axB.set_ylim(*YL); axB.legend(fontsize=7.5, frameon=False, loc="lower right"); axB.set_ylabel("co-add (norm.)")
    hole = np.isnan(f3) & (r > 1497) & (r < 1501)
    if hole.any(): axB.axvspan(r[hole].min() - 0.17, r[hole].max() + 0.17, color="firebrick", alpha=0.18, lw=0)
    axB.annotate("hole: 6 px masked AFTER the co-add,\nunder 14 good G130H exposures", xy=(1499.0, 0.95), xytext=(1503.5, 0.68), fontsize=7.5, color="firebrick", arrowprops=dict(arrowstyle="->", color="firebrick", lw=0.8))
    axB.annotate("spike: first G190H-only pixel,\non the ramp, nobody to outvote it", xy=(1514.6, 1.70), xytext=(1519.5, 2.25), fontsize=7.5, color="firebrick", arrowprops=dict(arrowstyle="->", color="firebrick", lw=0.8))
    rule(axB, "RULE (adopted): $1/\\sigma^2$-weighted median; then mask any CO-ADDED pixel whose error\nis below 1\\% of the co-add's median; last group of cumulative exposures; $1/\\sqrt{\\sum w}$ error.\nRESULT: the hijack is caught, but the guard acts on the co-added pixel, so the\nfourteen good exposures at 1498--1500 are discarded with the six bad ones.")
    # C: proposed
    axC.plot(r[k], f4[k], color="k", lw=1.1, label="emulation of the recommended rule")

    axC.set_ylim(*YL); axC.legend(fontsize=7.5, frameon=False, loc="lower right"); axC.set_ylabel("co-add (norm.)")
    axC.annotate("1498--1500 now filled by G130H", xy=(1499.0, 1.08), xytext=(1503.5, 0.68), fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    _sp = (r > 1514) & (r < 1515.3)
    axC.annotate("1514.6 \\AA\\ remains: inside the Handbook range (1605 \\AA\\ obs),\nfirst G190H-only pixel; S/N 7.5, carried with its error", xy=(1514.6, float(np.nanmax(f4[_sp])) if np.isfinite(f4[_sp]).any() else 1.7), xytext=(1521.0, 0.62), fontsize=7.5, color="firebrick", arrowprops=dict(arrowstyle="->", color="firebrick", lw=0.8))
    rule(axC, "RULE (recommended): truncate each exposure to the range the FOS Instrument Handbook (v6, Table 1-3)\nquotes for its grating and detector -- G190H on the red Digicon begins at 1590 \\AA\\ -- and apply the\n1\\% guard per EXPOSURE.  RESULT: the six zero-error diodes never enter and G130H fills 1498--1500;\nthe 1514.6 \\AA\\ pixel stays.  Blueshift and EW unchanged (measured).")
    # errors
    ax2.plot(r[k], e2[k], color="C1", lw=0.9, label=r"no guard: error $10^{-8}$, i.e. weight $10^{16}$ -- S/N would hide this (flux there is 0)")
    ax2.plot(r[k], e3[k], color="k", lw=1.1, label="adopted")
    ax2.plot(r[k], e4[k], color="k", lw=1.1, ls="--", dashes=(4, 2), label="recommended")
    ax2.set_yscale("log"); ax2.set_ylim(3e-9, 3); ax2.legend(fontsize=7.5, frameon=False, loc="lower right"); ax2.set_ylabel("co-added error (norm.)")
    for ax in axs:
        ax.axvspan(1500, 1600, color="#e8eef7", zorder=0)
        ax.axvspan(1498.6, 1590.0 / (1 + z), color="#f4b183", zorder=0, alpha=0.8)          # beyond the Handbook range
        ax.axvspan(1590.0 / (1 + z), 1650.0 / (1 + z), color="#fde8c8", zorder=0, alpha=0.35)  # footnote g: strong faceplate absorption
        ax.axvline(1515.2, color="C0", lw=0.6, ls=":")
    ax2.set_xlabel(r"Rest-frame Wavelength (\AA)"); ax2.set_xlim(lo, hi)
    fig.text(0.5, 0.004, r"dark orange: G190H diodes beyond the Handbook range ($<1590$ \AA\ obs).  light orange: Handbook footnote g ($<1650$ \AA\ obs).  blue dotted: last G130H pixel.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    p = os.path.join(OUT, "talk_C_Mrk1502_four_coadds.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    figure_a(); figure_b(); figure_c()
