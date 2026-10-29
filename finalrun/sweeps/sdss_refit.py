"""Refit the ICA with SDSS added to the HST spectrum, and see what moves.

GTR's prescription, in GTR's words: "take the current best fit ICA model, use
that to normalize the SDSS spectrum with respect to HST, then see how much
things change if the ICA fit is instead done with respect to HST+SDSS."

WHY A BRIDGE IS NEEDED.  The objects SDSS can help are exactly the ones where
it cannot be cross-normalised directly.  Of 212 adopted objects with both
spectra, 26 have real HST/SDSS overlap -- and those are low-redshift, with HST
already past 3000 A, so SDSS adds NOTHING inside the 1260-3000 A ICA range.
The 156 that do gain are high redshift, HST stopping near 1600 A while SDSS
begins near 1800 A, with no overlap at all.  So the scale cannot be measured
where the two spectra meet, because they never meet.  The HST-fitted ICA model
spans the whole ICA range and is the only thing that reaches across the gap.

SCALE.  s = median(model / SDSS flux) over every SDSS pixel inside the ICA
range.  The model is in the host's arbitrary flux units, so s * SDSS is too.
SDSS flux is in 1e-17 erg/s/cm2/A, so the Continuum Normalisation for the
donated pixels is the constant 1e-17 / s, which keeps flux * continuum in
physical units and leaves the monochromatic luminosities measurable -- and
makes L2500 measurable from real data for the first time on these objects.

The scale is also measured in five disjoint windows.  Their scatter about the
global value is the honest quality flag on the bridge: it is the model's SHAPE
being tested over a range it never saw.  Measured on three objects it runs
from 4 per cent (index 474) to 20 per cent with a blue-to-red tilt (index 464),
the tilt meaning the model's continuum slope is slightly wrong out there.

SPLICE, NOT CO-ADD, following splice_spectra.py and for its reasons: SDSS
supplies ONLY pixels the HST spectrum does not cover, so no pixel is ever
averaged across instruments of different resolution.  Here that rule costs
nothing, because the donated range (roughly 1800-3000 A) is nowhere near C IV.
SDSS log-lambda spacing is 1e-4 dex, the same step as the rebin lattice, so the
donor lands on the host's lattice by rounding, as a donor spectrum does.

The mask and component set are held at the adopted values, so the only thing
that changes is the data the fit sees.
"""
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
FINALRUN = os.path.dirname(HERE)
sys.path.insert(0, FINALRUN)
import object_diagnostics as diag

LITE = os.path.join(FINALRUN, "data_v23", "SDSS_spec", "lite")
REBIN = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "rebin")
FITS_DIR = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "fits")
FINAL = os.path.join(FINALRUN, "pipeline_output", "final")
WORK = os.path.join(FINALRUN, "pipeline_output", "chosen_visit", "sdss_refit_work")
STEP = 1e-4 * np.log(10.0)
ICA_LO, ICA_HI = 1260.0, 3000.0
WINDOWS = [(1750, 1850), (1950, 2200), (2200, 2650), (2650, 2700), (2900, 3000)]

C_WAVE = "Rest-frame Wavelength"
C_FLUX = "Coadded Flux (Arbitrary Units)"
C_ERRS = "Coadded Flux Errors"
C_MASK = "Bad Pixel Mask"
C_ZED  = "Redshift"
C_CONT = "Continuum Normalisation"


def build_combined(stem, z, model_w, model_f, fn_sdss):
    """Host + scaled SDSS on one lattice.  Returns (path, info) or (None, why)."""
    hp = os.path.join(REBIN, stem + ".fits")
    sp = os.path.join(LITE, fn_sdss)
    if not os.path.exists(hp) or not os.path.exists(sp):
        return None, "missing input spectrum"
    h = fits.open(hp)[1].data
    hw = np.asarray(h[C_WAVE], float)
    hf = np.asarray(h[C_FLUX], float)
    he = np.asarray(h[C_ERRS], float)
    hm = np.asarray(h[C_MASK], float)
    hc = np.asarray(h[C_CONT], float) if C_CONT in h.columns.names else np.full(hw.size, np.nan)
    hg = np.isfinite(hf) & (hm == 0)
    if hg.sum() < 50:
        return None, "host has no usable pixels"

    with fits.open(sp) as g:
        d = g[1].data
        lam = 10.0 ** np.asarray(d["loglam"], float)
        sf = np.asarray(d["flux"], float)
        iv = np.asarray(d["ivar"], float)
    sw = lam / (1.0 + z)
    sg = (iv > 0) & np.isfinite(sf) & (sw >= ICA_LO) & (sw <= ICA_HI)
    if sg.sum() < 50:
        return None, "SDSS has no usable pixels in the ICA range"

    mod = np.interp(sw[sg], model_w, model_f)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mod / sf[sg]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size < 50:
        return None, "no usable model/SDSS ratio"
    s = float(np.median(ratio))
    mad = float(np.median(np.abs(ratio - s)) / s)

    wsc = []
    for lo, hi in WINDOWS:
        k = sg & (sw >= lo) & (sw <= hi)
        if k.sum() > 20:
            r = mod[(sw[sg] >= lo) & (sw[sg] <= hi)] / sf[k]
            r = r[np.isfinite(r) & (r > 0)]
            if r.size > 20:
                wsc.append(float(np.median(r)) / s)
    tilt = (max(wsc) / min(wsc)) if len(wsc) >= 2 else np.nan

    ref = np.log(hw[0])
    kh = np.rint((np.log(hw) - ref) / STEP).astype(int)
    ks = np.rint((np.log(sw[sg]) - ref) / STEP).astype(int)
    have = set(kh[hg].tolist())
    take = np.array([k not in have for k in ks])
    if take.sum() < 50:
        return None, "SDSS adds fewer than 50 pixels the host lacks"

    se = np.full(sg.sum(), np.nan)
    ivv = iv[sg]
    se[ivv > 0] = 1.0 / np.sqrt(ivv[ivv > 0])

    k_all = np.concatenate([ks[take], kh[hg]])
    f_all = np.concatenate([sf[sg][take] * s, hf[hg]])
    e_all = np.concatenate([se[take] * s, he[hg]])
    c_all = np.concatenate([np.full(int(take.sum()), 1e-17 / s), hc[hg]])
    o = np.argsort(k_all)
    k_all, f_all, e_all, c_all = k_all[o], f_all[o], e_all[o], c_all[o]
    uniq, first = np.unique(k_all, return_index=True)
    k_all, f_all, e_all, c_all = uniq, f_all[first], e_all[first], c_all[first]

    full = np.arange(k_all.min(), k_all.max() + 1)
    wave = np.exp(ref + full * STEP)
    at = np.searchsorted(full, k_all)

    def lift(a):
        out = np.full(full.size, np.nan)
        out[at] = a
        return out

    flux, errs, cont = lift(f_all), lift(e_all), lift(c_all)
    mask = np.where(np.isfinite(flux), 0.0, 1.0)
    cols = fits.ColDefs([
        fits.Column(name=C_WAVE, format="D", array=wave),
        fits.Column(name=C_FLUX, format="D", array=flux),
        fits.Column(name=C_ERRS, format="D", array=errs),
        fits.Column(name=C_MASK, format="D", array=mask),
        fits.Column(name=C_ZED,  format="D", array=np.full(wave.size, z)),
        fits.Column(name=C_CONT, format="D", array=cont)])
    os.makedirs(WORK, exist_ok=True)
    out = os.path.join(WORK, stem + ".fits")
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(out, overwrite=True)
    return out, dict(scale=s, scale_mad=mad, scale_window_tilt=tilt,
                     sdss_px_added=int(take.sum()), host_px=int(hg.sum()),
                     sdss_lo=float(sw[sg][take].min()), sdss_hi=float(sw[sg][take].max()))


def main():
    dec = pd.read_csv("/Users/gtr/Work/git/HST-Paper/Data/decisions.csv")
    dec["index"] = dec["folder"].map(lambda f: int(str(f).split("_")[0]))
    ad = dec[dec["manmask"].isin([2, 3])]
    sd = pd.read_csv(os.path.join(FINAL, "sdss_spectra.csv"))
    sd = sd[sd["fn_sdss"].notna()]
    sm = {int(r["index"]): r for _, r in sd.iterrows()}
    only = set(sys.argv[1:])
    rows = []
    for _, a in ad.iterrows():
        i = int(a["index"])
        if only and str(i) not in only:
            continue
        s = sm.get(i)
        if s is None:
            continue
        recs = (glob.glob(os.path.join(FITS_DIR, "%d_*" % i, "records", "BEST_*.json")) or
                glob.glob(os.path.join(FITS_DIR, "%03d_*" % i, "records", "BEST_*.json")))
        if not recs:
            continue
        rec = json.load(open(recs[0]))
        stem, z, ov = rec["name"], rec["result"]["z"], rec["override"]
        comps = ov.get("comps_use")
        try:
            base = diag.fit(stem, ov, comps, REBIN)
        except Exception as exc:
            print("  %4d  base refit failed: %s" % (i, type(exc).__name__), flush=True)
            continue
        path, info = build_combined(stem, z, np.asarray(base["wave_ica"], float),
                                    np.asarray(base["flux_ica"], float), str(s["fn_sdss"]))
        if path is None:
            print("  %4d %-34s SKIP: %s" % (i, stem[:34], info), flush=True)
            continue
        try:
            new = diag.fit(stem, ov, comps, WORK)
        except Exception as exc:
            print("  %4d %-34s combined refit failed: %s" % (i, stem[:34], type(exc).__name__), flush=True)
            continue
        b0, e0 = base["civ_blue"], base["civ_ew"]
        b1, e1 = new["civ_blue"], new["civ_ew"]

        # Fit quality at C IV, on the SAME pixels both ways.  This is the number
        # that decides whether adding SDSS helps: the host data under C IV are
        # untouched, so any change is the fit choosing different components.
        def civ_resid(r):
            wa = np.asarray(r["wave_arb"], float); fa = np.asarray(r["flux_arb"], float)
            ea = np.asarray(r["errs_arb"], float); ma = np.asarray(r["mask_arb"], int)
            wi = np.asarray(r["wave_ica"], float); fi = np.asarray(r["flux_ica"], float)
            g = np.isfinite(fa) & (ea > 0) & (ma == 0) & (wa >= 1500) & (wa <= 1600)
            if g.sum() < 10:
                return np.nan, np.nan
            d = (fa[g] - np.interp(wa[g], wi, fi)) / ea[g]
            return float(np.median(d)), float(np.sqrt(np.mean(d ** 2)))
        m0, r0 = civ_resid(base)
        m1, r1 = civ_resid(new)
        row = dict(index=i, folder=a["folder"], manmask=a["manmask"], z=z,
                   comps=comps or "auto",
                   blue_hst=b0, blue_hst_sdss=b1, dblue=b1 - b0,
                   ew_hst=e0, ew_hst_sdss=e1,
                   dlogew=(np.log10(e1) - np.log10(e0)) if e0 > 0 and e1 > 0 else np.nan,
                   scale_hst=diag.scale(base), scale_hst_sdss=diag.scale(new),
                   civ_resid_med_hst=m0, civ_resid_rms_hst=r0,
                   civ_resid_med_hst_sdss=m1, civ_resid_rms_hst_sdss=r1)
        row.update(info)
        rows.append(row)
        print("  %4d %-32s +%4d px (%.0f-%.0f A)  blue %8.1f -> %8.1f (%+8.1f, %4.1f q)  "
              "logEW %+.3f  CIVrms %.2f->%.2f"
              % (i, stem[:32], info["sdss_px_added"], info["sdss_lo"], info["sdss_hi"],
                 b0, b1, b1 - b0, abs(b1 - b0) / 69.03, row["dlogew"], r0, r1),
              flush=True)
    t = pd.DataFrame(rows)
    outp = os.path.join(FINAL, "sdss_refit.csv")
    t.to_csv(outp, index=False)
    print("\n%d objects -> %s" % (len(t), outp))


if __name__ == "__main__":
    main()
