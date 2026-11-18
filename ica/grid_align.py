"""Putting a spectrum on the MFICA component grid.

The components (Rankine et al. 2020) are tabulated on a fixed rest-frame grid,
log10(lambda) = k x 1e-4 (69.03 km/s pixels).  Our co-added spectra are on the
SDSS observed-frame grid divided by (1+z), so they have the same spacing but a
phase relative to the component grid that is set by each object's redshift.
Until 2026-11-17 the fit paired spectrum and components BY INDEX after cutting
the components to the pixels strictly redward of the first data pixel, so every
data pixel was fitted against a component pixel 0-69 km/s redder (median +32
km/s over the 433 chosen-visit products).  The reconstruction lives on the
component grid and C IV is measured there, so the blueshift was biased low by
that redshift-dependent amount.

The fix: resample the spectrum onto the component grid by linear interpolation
in log10(lambda), the ordinary step in any MFICA application to spectra on a
different grid.  Errors are interpolated the same way (an over-estimate of the
interpolated pixel's variance, so slightly conservative); a component pixel is
masked if any data pixel contributing to it is masked or unusable.
"""
import numpy as np


def to_component_grid(wave, flux, errs, mask, sel, wave_comp):
    """Return (comp_sel, flux_i, errs_i, mask_i).

    comp_sel : boolean mask on wave_comp -- the component pixels covered by the
               selected data pixels wave[sel];
    flux_i, errs_i, mask_i : the spectrum interpolated onto wave_comp[comp_sel],
               mask_i == 1 where the pixel must not be fitted.
    """
    wave = np.asarray(wave, float); flux = np.asarray(flux, float)
    errs = np.asarray(errs, float); mask = np.asarray(mask, float)
    sel = np.asarray(sel, bool)
    w, f, e, m = wave[sel], flux[sel], errs[sel], (mask[sel] > 0)
    comp_sel = (wave_comp >= w.min()) & (wave_comp <= w.max())
    x = np.log10(wave_comp[comp_sel])
    good = np.isfinite(f) & np.isfinite(e) & (e > 0)
    if good.sum() < 2:
        n = int(comp_sel.sum())
        return comp_sel, np.zeros(n), np.ones(n), np.ones(n)
    xs = np.log10(w)
    flux_i = np.interp(x, xs[good], f[good])
    errs_i = np.interp(x, xs[good], e[good])
    # a component pixel inherits a mask from either neighbouring data pixel
    bad = np.interp(x, xs, (m | ~good).astype(float)) > 0
    return comp_sel, flux_i, errs_i, bad.astype(float)
