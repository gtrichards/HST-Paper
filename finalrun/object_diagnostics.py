"""Standard per-object diagnostics on an adopted override.

The four checks run on every object of the final pass, kept here rather than in
a scratch file because they are what the decision notes in
pipeline_output/final/decisions.csv quote, and a reader who wants to reproduce a
row needs them.  It was a scratch file for most of the pass and was twice
deleted by a system cleanup of /private/tmp mid-session, which is the immediate
reason it now lives in the repository.

    fit(stem, override, comps, rebin_path)   refit under an override
    scale(res)        median(model)/median(data) over 1445-1465 A; healthy
                      0.88-1.00.  Says where the morphed continuum sits.
    hand(res, red=)   two-point continuum from a blue and a red anchor, the
                      half-flux point of the continuum-subtracted DATA, and a
                      400-draw bootstrap.  Independent of the reconstruction.
                      Returns None when an anchor is missing -- which happens
                      from index 427 on, where the data stop short of 1700 A and
                      a substitute anchor has to be named explicitly.
    veto_tests(res)   what a sub-continuum veto is actually made of: how many
                      pixels, over what span, and -- the part that decides
                      whether it matters -- what FRACTION OF THE CLAMPED LINE
                      FLUX falls in that span.  A veto carrying 0.0 per cent is
                      cosmetic.  data_over_cont_in_span above 1 means the data
                      sit above the continuum where the model dips below it, so
                      masking would make the fit worse.
"""
import os, sys, json, numpy as np
sys.path.insert(0, "/Users/gtr/Work/projects/hstica/finalrun")
sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
os.environ.setdefault("HSTICA_REBIN", "/Users/gtr/Work/projects/hstica/finalrun/pipeline_output/chosen_visit/rebin")
C = 299792.458
LAM0 = 1549.48

def fit(stem, ov, comps=None, path=None):
    from ica.manual_fix import ICAManualFixProcessor
    proc = ICAManualFixProcessor(rebin_path=path, master_mode=True)
    return proc.fit_with_overrides(stem, mask_ranges=[list(r) for r in ov.get("mask_ranges", [])],
                                   mask_pixels=[float(x) for x in ov.get("mask_pixels", [])],
                                   unmask_ranges=[list(r) for r in ov.get("unmask_ranges", [])],
                                   unmask_pixels=[float(x) for x in ov.get("unmask_pixels", [])],
                                   comps_use=comps or ov.get("forced_components") or None)

def scale(res, lo=1445.0, hi=1465.0):
    wa, fa = np.asarray(res["wave_arb"], float), np.asarray(res["flux_arb"], float)
    wi, fi = np.asarray(res["wave_ica"], float), np.asarray(res["flux_ica"], float)
    s = (wa >= lo) & (wa <= hi) & np.isfinite(fa)
    t = (wi >= lo) & (wi <= hi) & np.isfinite(fi)
    if s.sum() < 3 or t.sum() < 3: return float("nan")
    return float(np.median(fi[t]) / np.median(fa[s]))

def hand(res, blue=(1445.0, 1465.0), red=(1700.0, 1705.0), nboot=400, seed=1):
    wa = np.asarray(res["wave_arb"], float); fa = np.asarray(res["flux_arb"], float)
    ea = np.asarray(res["errs_arb"], float); ma = np.asarray(res["mask_arb"])
    good = np.isfinite(fa) & np.isfinite(ea) & (ea > 0) & (np.asarray(ma, int) == 0)
    def anch(w0, w1):
        s = good & (wa >= w0) & (wa <= w1)
        return (float(np.mean(wa[s])), float(np.median(fa[s])), int(s.sum())) if s.sum() >= 3 else (np.nan,)*2+(0,)
    xb, yb, nb = anch(*blue); xr, yr, nr = anch(*red)
    if nb < 3 or nr < 3: return None
    line = (wa - xb) * (yr - yb) / (xr - xb) + yb
    sel = good & (wa >= 1500.0) & (wa <= 1600.0)
    w, d, e = wa[sel], (fa - line)[sel], ea[sel]
    def half(dd):
        cum = np.cumsum(np.clip(dd, 0, None))
        if cum[-1] <= 0: return np.nan
        return float(np.interp(0.5 * cum[-1], cum, w))
    v0 = (LAM0 - half(d)) / LAM0 * C
    rng = np.random.default_rng(seed)
    vs = [(LAM0 - half(d + rng.normal(0, e))) / LAM0 * C for _ in range(nboot)]
    cont = np.interp(LAM0, wa[good], line[good])
    ew = float(np.trapezoid(np.clip(d, 0, None) / cont, w))
    return dict(v=float(v0), v_err=float(np.nanstd(vs)), ew=ew, n_px=int(sel.sum()),
                nb=nb, nr=nr, anchors=(round(xb,1), round(xr,1)))

def veto_tests(res):
    """Cumulative-flux and under-fitting tests on the vetoed stretch."""
    import iterate_fit as IF
    wi = np.asarray(res["wave_ica"], float); fi = np.asarray(res["flux_ica"], float)
    ew, cont = IF.civ_continuum(wi, fi)
    if ew is None: return None
    below = ew & (fi < cont)
    clamped = np.clip(fi - cont, 0, None)
    tot = clamped[ew].sum()
    cum = np.cumsum(clamped[ew]); half = np.interp(0.5*cum[-1], cum, wi[ew]) if cum[-1] > 0 else np.nan
    lo, hi = (wi[below].min(), wi[below].max()) if below.any() else (np.nan, np.nan)
    inspan = ew & (wi >= lo) & (wi <= hi)
    frac = float(clamped[inspan].sum()/tot) if tot > 0 else float("nan")
    wa = np.asarray(res["wave_arb"], float); fa = np.asarray(res["flux_arb"], float)
    ma = np.asarray(res["mask_arb"], int)
    c_at = np.interp(wa, wi, cont)
    s = np.isfinite(fa) & (ma == 0) & (wa >= lo) & (wa <= hi)
    ratio = float(np.median(fa[s]/c_at[s])) if s.sum() else float("nan")
    return dict(n_below=int(below.sum()), span=(round(float(lo),1), round(float(hi),1)),
                cumflux_frac_in_span=round(frac,4), data_over_cont_in_span=round(ratio,3),
                n_data_px_in_span=int(s.sum()), half_flux_wave=round(float(half),2),
                worst_frac=round(float(np.min((fi[below]-cont[below])/cont[below])),4) if below.any() else 0.0)
