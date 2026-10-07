"""
rebinning/read_spec_data.py
Migrated from: Trevor Code/Read_spec_data_2026AP.py
Migration date: 2026-04-30

Import/path changes:
  - `import Cut_Edge_Pix_TVM_NoDQ as Cut_Edge_Pix_TVM`
      → `from rebinning import cut_edge_pix as Cut_Edge_Pix_TVM`
  - `import SpecCuts_HSLA`
      → `from rebinning import spec_cuts_hsla as SpecCuts_HSLA`
  - `spec = fits.open(path+"original\\%s"%Identifier)` in read_hsla()
      → `fits.open(os.path.join(path, "original", Identifier))`
      (backslash replaced with os.path.join for cross-platform portability)
No algorithmic changes.

Original file header (Read_spec_data_2026AP.py):
  AP: 2026-03-08 — 2026AP iteration
  This is Read_spec_data_2026E.py with one change:
    Imports Cut_Edge_Pix_TVM_NoDQ instead of Cut_Edge_Pix_TVM_DQOnly.
    The DQ==0 quality cut is removed entirely.
  All other logic is identical to 2026E:
    - FOS multi-row error formula: sqrt(sum(e**2)) / (N-1)  [hypothesized Aug 2022]
    - No SNR edge cut
    - SpecCuts wavelength edge cuts still applied
"""

import os
import pandas as pd
import numpy as np
from astropy.io import fits
import glob

from rebinning import cut_edge_pix as Cut_Edge_Pix_TVM
from rebinning import spec_cuts_hsla as SpecCuts_HSLA


# AP 2026-06-17: TEMPORARY exposure cap for COS reverberation-mapping monitoring
# targets. These objects have far more exposures than any other (Mrk 817 ~1600,
# NGC 5548 ~800), which makes the pure-Python coadd in coadd.py appear to hang.
# STOPGAP: keep only the N highest-S/N exposures. See Migration_Log.md
# "TEMP: COS exposure cap" -- REVISIT this decision.
_COS_EXPOSURE_CAP = {"Mrk 817": 100, "NGC 5548": 100}


#: Per-file record of what the most recent flat read kept and dropped, and why.
#: The rebinned product says nothing about the exposures behind it, so there was
#: no way to show which archive files actually reached a fit -- needed both for
#: the review figures and for the paper's "spectra available / spectra used"
#: columns.  Purely additive: nothing here changes what the readers return.
PROVENANCE = []


def _prov_reset():
    del PROVENANCE[:]


def _prov(fn, kept, reason=""):
    PROVENANCE.append(dict(file=os.path.basename(fn), path=fn,
                           kept=bool(kept), reason=reason))


def _wrong_instrument(fn, want):
    """True if this file's INSTRUME does not match the directory it was filed in.

    27 STIS files across nine objects were retrieved into COS directories, and
    read_cos_flat globs *_x1d.fits without asking what wrote them, so STIS G430M
    and G230L exposures were being read as COS and co-added with the FUV data.
    G430M lies outside the ICA range and merely inflated the lattice; G230L
    overlaps it at 1600-3200 A, so on Mrk 231 and Mrk 817 optical-arm STIS data
    was mixing into the COS co-add.  Cheap to check, so check.
    """
    try:
        got = str(fits.getheader(fn).get('INSTRUME', '')).strip().upper()
    except Exception:
        return False
    return bool(got) and got != want.upper()


def _cos_median_snr(fn):
    """Median per-pixel S/N of a COS x1d exposure, for ranking. Flattens all
    rows (segments/stripes) so it works for FUV (2 seg) and NUV (3 stripe) data."""
    data = fits.open(fn)[1].data
    flux = np.concatenate([np.asarray(r) for r in data['FLUX']])
    err  = np.concatenate([np.asarray(r) for r in data['ERROR']])
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    return float(np.nanmedian(flux[good] / err[good])) if good.any() else -np.inf


#: Below this the wavelength solution is not physical for any HST UV mode.
_MIN_PHYSICAL_WAVE = 500.0


#: STIS echelle modes write one row per spectral order.  An order is narrow --
#: about 18 A in the rest frame near C IV at z~0.09 -- so it must never be
#: treated as a separate exposure: see _rows_of.
_ECHELLE = ("E140M", "E230M", "E140H", "E230H")


def _rows_of(data, stitch=False):
    """Each row of an extracted spectrum, cleaned, as its own arrays.

    With `stitch`, all rows are concatenated into a single spectrum instead.
    That is required for STIS echelle modes and wrong for everything else.
    coadd.rebin divides every row it is given by that row's own fitted
    continuum before co-adding, and an echelle order is narrower than a broad
    emission line: on Mrk 1383 (E140M, 44 orders) the order holding the C IV
    core spans 1575-1593 A in the rest frame, has no line-free pixels at all,
    and was normalised by its own median -- which is the line.  The line came
    back flat.  Measured against the 1445-1465 A continuum, the 1530-1570 A
    band read 2.08 in the students' master co-add and 0.95 here; stitching the
    orders first restores it to 3.09 on a single exposure.  The pre-flat
    reader, read_stis, special-cased E140M and concatenated its orders for
    exactly this reason; the generalisation below to per-row output was made
    to recover COS segments and NUV stripes, which are hundreds of Angstroms
    wide and genuinely want separate treatment, and it silently took echelle
    with it.

    COS writes one row per detector segment -- two for the FUV (FUVA, FUVB) and
    three for the NUV stripes -- and STIS echelle modes one row per order.  The
    readers used to take row 0 alone for every grating except E140M, discarding
    about half of each FUV spectrum and two thirds of each NUV spectrum.  For
    [VV2006] J104839.4+442820 at z=0.999 that lost stripe NUVB, rest
    1391-1639 A, which holds C IV with 129 good pixels, and the object was
    excluded for having no C IV while the line sat in the file.

    Pixels whose wavelength is not physical are dropped.  COS G140L writes
    segment FUVB with a solution running from -28 A: the segment is largely
    unilluminated there and the solution meaningless, but the DQ flags do not
    say so.  NGC 985 picked up 12209 such pixels, which stretched the
    co-addition lattice from about 3000 points to 20923 and broke the morph
    against its 11177-point reference continuum.  HST ultraviolet coverage
    begins near 900 A, so nothing real is lost.
    """
    out = []
    for t in range(data.size):
        w = np.asarray(data['WAVELENGTH'][t], float)
        f = np.asarray(data['FLUX'][t], float)
        e = np.asarray(data['ERROR'][t], float)
        q = np.asarray(data['DQ'][t], float)
        ok = np.isfinite(w) & (w > _MIN_PHYSICAL_WAVE)
        if ok.sum() < 2:
            continue
        w, f, e, q = w[ok], f[ok], e[ok], q[ok]
        order = np.argsort(w, kind='stable')
        out.append((w[order], f[order], e[order], q[order]))
    if stitch and len(out) > 1:
        w = np.concatenate([r[0] for r in out])
        f = np.concatenate([r[1] for r in out])
        e = np.concatenate([r[2] for r in out])
        q = np.concatenate([r[3] for r in out])
        order = np.argsort(w, kind='stable')
        return [(w[order], f[order], e[order], q[order])]
    return out


def _carries_flux(fn):
    """Does this extracted spectrum contain any non-zero flux at all?

    Short COS and STIS acquisition exposures are archived with
    productType SCIENCE and a full-length, correctly-shaped table whose FLUX
    column is zero in every pixel.  They survive the empty-BinTable check above,
    then fail the edge-pixel quality cut with "No good pixels found", and that
    exception aborts the whole object -- five objects in the first repair batch,
    among them NGC 3516, NGC 3783 and Ton 469, each lost to a single 12-second
    exposure.

    Dropping them carries no information loss: a spectrum of zeros contributes
    nothing to an inverse-variance-weighted co-add, and the alternative is not
    a different co-add but no co-add at all.  This is the same treatment the
    empty-BinTable files already get, for the same reason.
    """
    try:
        data = fits.open(fn)[1].data
        if data is None or len(data) == 0:
            return False
        for row in data:
            f = np.asarray(row['FLUX'], float)
            if np.any(np.isfinite(f) & (f != 0.0)):
                return True
    except Exception:
        return True          # unreadable for another reason: let the reader say so
    return False


def _is_cumulative_accum(flux_arr, index=-1):
    """Are an FOS exposure's groups cumulative readouts rather than independent?

    FOS ACCUM data can be written either as independent sub-integrations or as
    running sums, in which case each group is larger than the one before and the
    last group alone is the whole exposure.  The original test compared three
    consecutive group means, `mean(g[-1]) > mean(g[-2]) > mean(g[-3])`, which
    raises IndexError on an exposure with fewer than three groups -- and one such
    exposure aborts the whole object.  Of 92 FOS exposures in the first repair
    batch, 13 have exactly two groups, and they broke 7 of the 9 FOS objects.

    The comparison is generalised to however many groups exist, so the
    three-or-more case is bit-for-bit what it always was:

      >= 3 groups   mean(g[-1]) > mean(g[-2]) > mean(g[-3])   (unchanged)
         2 groups   mean(g[-1]) > mean(g[-2])
         1 group    True -- with a single group both branches return that group

    This is a fix to an implementation defect, not a change to the method: no
    exposure that the old code could read is read differently.
    """
    n = flux_arr.shape[0]
    idx = index if index >= 0 else n + index
    if idx <= 0:
        return True
    if idx == 1:
        return np.mean(flux_arr[idx]) > np.mean(flux_arr[idx - 1])
    return (np.mean(flux_arr[idx]) > np.mean(flux_arr[idx - 1]) and
            np.mean(flux_arr[idx - 1]) > np.mean(flux_arr[idx - 2]))


def read_data(Identifier, path, data_origin, z):
    if data_origin == "FOS":
        return read_fos(Identifier, path, z)
    elif data_origin == "STIS":
        return read_stis(Identifier, path, z)
    elif data_origin == "COS":
        return read_cos(Identifier, path, z)
    elif data_origin == "HSLA":
        return read_hsla(Identifier, path, z)
    elif data_origin == "SDSS-RM":
        return read_sdssrm(Identifier, path, z)
    print("data_origin not recognized")


def read_data_flat(name, path, data_origin, z):
    """
    Read spectra from a flat directory layout produced by pipeline/download_spectra.py.

    Unlike read_data(), which expects NecessaryParams.csv and per-obs subdirectories,
    this function globs all relevant files directly from `path` and reads the grating
    from each file's FITS primary header (OPT_ELEM keyword).

    Parameters
    ----------
    name : str
        Object common name (used for logging in Cut_Edge_Pix calls).
    path : str
        Flat directory containing the FITS files (raw_data/{name}/{inst}/).
    data_origin : str
        Instrument: 'COS', 'STIS', or 'FOS'.
    z : float
        Redshift.
    """
    if data_origin == "COS":
        return read_cos_flat(name, path, z)
    elif data_origin == "STIS":
        return read_stis_flat(name, path, z)
    elif data_origin == "FOS":
        return read_fos_flat(name, path, z)
    print("data_origin not recognized for flat reader")


def read_sdssrm(Identifier, path, z):
    fn_list = glob.glob(os.path.join(glob.escape(path), glob.escape(Identifier), '*.fits'))
    array_lens = []
    for i in range(len(fn_list)):
        wavelength = 10.**fits.open(fn_list[i])[1].data["LOGLAM"]
        array_lens.append(len(wavelength))
    array_len = max(array_lens)

    waves  = np.zeros((len(fn_list), array_len))
    fluxes = np.zeros((len(fn_list), array_len))
    errs   = np.zeros((len(fn_list), array_len))
    masks  = np.zeros((len(fn_list), array_len))

    for i in range(len(fn_list)):
        hdu = fits.open(fn_list[i])
        loglam = hdu[1].data["LOGLAM"]
        wave   = 10.**loglam
        flux   = hdu[1].data["FLUX"]
        err    = 1. / np.sqrt(hdu[1].data["IVAR"])
        mask   = hdu[1].data["AND_MASK"]
        waves[i,:len(wave)]  = wave
        fluxes[i,:len(wave)] = flux
        errs[i,:len(wave)]   = err
        masks[i,:len(wave)]  = mask

    return waves, fluxes, errs, masks


def read_hsla(Identifier, path, z):
    spec = fits.open(os.path.join(path, "original", Identifier))
    coadd_wave = spec[1].data["WAVE"]
    coadd_flux = spec[1].data["FLUX"]
    coadd_errs = spec[1].data["ERROR"]
    coadd_mask = np.zeros(len(coadd_wave))

    #Manually cut red end - big problem for HSLA co-adds
    obj_name = "_".join(Identifier.split("_")[:-1])
    if obj_name in SpecCuts_HSLA.RedEdges:
        lambdaend = SpecCuts_HSLA.RedEdges[obj_name]
        indstop  = np.argmin( np.abs((coadd_wave/(1+z))-lambdaend) )
        coadd_wave = coadd_wave[:indstop]
        coadd_flux = coadd_flux[:indstop]
        coadd_errs = coadd_errs[:indstop]
        coadd_mask = coadd_mask[:indstop]

    coadd_wave = Cut_Edge_Pix_TVM.Cut_Edge_Pix(np.zeros(len(coadd_wave)), coadd_wave, coadd_flux, coadd_errs, coadd_wave.copy(), len(coadd_wave), "wavelength", False, z, "%s"%(Identifier), "HSLA")
    coadd_flux = Cut_Edge_Pix_TVM.Cut_Edge_Pix(np.zeros(len(coadd_wave)), coadd_wave, coadd_flux, coadd_errs, coadd_flux.copy(), len(coadd_flux), "flux", False, z, "%s"%(Identifier), "HSLA")
    #HSLA saves empty values as zero; change to nan
    coadd_flux[coadd_flux==0] = np.nan
    coadd_errs = Cut_Edge_Pix_TVM.Cut_Edge_Pix(np.zeros(len(coadd_wave)), coadd_wave, coadd_flux, coadd_errs, coadd_errs.copy(), len(coadd_errs), "flux error", False, z, "%s"%(Identifier), "HSLA")
    coadd_mask = Cut_Edge_Pix_TVM.Cut_Edge_Pix(np.zeros(len(coadd_wave)), coadd_wave, coadd_flux, coadd_errs, coadd_mask.copy(), len(coadd_mask), "masks", False, z, "%s"%(Identifier), "HSLA")

    return np.array([coadd_wave]), np.array([coadd_flux]), np.array([coadd_errs]), np.array([coadd_mask])


def read_cos(Identifier, path, z):
    #get observation details
    try:
        obs_details = pd.read_csv(path+"%s/all_exposures.txt" %(Identifier), sep="\s+")
        gratings    = obs_details["Grating"].values
        spec_names  = obs_details["Rootname"].values
    except FileNotFoundError:
        obs_details = pd.read_csv(path+"%s/NecessaryParams.csv" %(Identifier))
        gratings    = obs_details["filters"].values
        spec_names  = obs_details["obs_id"].values

    #take max wave size for initializing below
    array_sizes = []
    for i in range(len(spec_names)):
        try:
            data = fits.open(path+'%s/%s/%s_x1d.fits' %(Identifier,spec_names[i],spec_names[i]))
        except FileNotFoundError:
            try:
                data = fits.open(path+'%s/%s.fits' %(Identifier,spec_names[i]))
            except FileNotFoundError:
                data = fits.open(path+'%s/Data/%s_x1d.fits' %(Identifier,spec_names[i]))
        data = data[1].data
        if gratings[i]=='E140M':
            array_sizes.append(data.size*1024)
        else:
            array_sizes.append(len(data['Wavelength'][0]))
    array_len = max(array_sizes)

    waves     = np.zeros((len(spec_names), array_len))
    fluxes    = np.zeros((len(spec_names), array_len))
    flux_errs = np.zeros((len(spec_names), array_len))
    masks     = np.zeros((len(spec_names), array_len))

    for i in range(len(spec_names)):
        Bad_list = []
        try:
            data = fits.open(path+'%s/%s/%s_x1d.fits' %(Identifier,spec_names[i],spec_names[i]))[1].data
        except FileNotFoundError:
            try:
                data = fits.open(path+'%s/%s.fits' %(Identifier,spec_names[i]))[1].data
            except FileNotFoundError:
                data = fits.open(path+'%s/Data/%s_x1d.fits' %(Identifier,spec_names[i]))[1].data

        if gratings[i]=="E140M":
            wavelength = []
            flux       = []
            fluxerr    = []
            DQ         = []
            for t in np.arange(0,data.size,1):
                wavelength = np.append(wavelength, data['WAVELENGTH'][t])
                flux       = np.append(flux, data['FLUX'][t])
                fluxerr    = np.append(fluxerr, data['ERROR'][t])
                DQ         = np.append(DQ, data['DQ'][t])
        else:
            wavelength = data['WAVELENGTH'][0]
            flux       = data['FLUX'][0]
            fluxerr    = data['ERROR'][0]
            DQ         = data['DQ'][0]

        flux_wmask = flux.copy()
        err_wmask  = fluxerr.copy()
        sel        = ((wavelength >= 1215.0) & (wavelength <= 1216.0))
        masks[i,:][(err_wmask == 0.0)] = 1
        # A pixel whose flux is exactly 0.0 is not a measurement of zero flux: it is a
        # dead or unilluminated pixel that the archive flags in DQ, and 43% of
        # NGC 3783's COS pixels in the C IV window are of this kind, every one of them
        # carrying a non-zero error.  Cut_Edge_Pix already defines a good pixel as
        # flux != 0 and fluxerr != 0, but uses that only to trim the ends of the array,
        # so interior dead pixels reached the co-add with full 1/err^2 weight and pulled
        # the weighted median toward zero.  This applies the same definition everywhere.
        masks[i,:][(flux_wmask == 0.0)] = 1
        waves[i,:]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                wavelength, array_len, "wavelength", False, z, "%s - %s"%(Identifier,spec_names[i]), "COS")
        fluxes[i,:]    = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                flux_wmask, array_len, "flux", False, z, "%s - %s"%(Identifier,spec_names[i]), "COS")
        flux_errs[i,:] = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                err_wmask, array_len, "flux error", False, z, "%s - %s"%(Identifier,spec_names[i]), "COS")
        masks[i,:]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                masks[i,:], array_len, "masks", False, z, "%s - %s"%(Identifier,spec_names[i]), "COS")

    return waves, fluxes, flux_errs, masks


def read_stis(Identifier, path, z):
    obs_details = pd.read_csv(path+"%s/NecessaryParams.csv" %(Identifier))
    gratings    = obs_details["filters"].values
    spec_names  = obs_details["obs_id"].values

    array_sizes = []
    for i in range(len(spec_names)):
        try:
            data = fits.open(path+'%s/%s/%s_x1d.fits' %(Identifier,spec_names[i],spec_names[i]))
        except FileNotFoundError:
            data = fits.open(path+'%s/%s/%s_sx1.fits' %(Identifier,spec_names[i],spec_names[i]))
        data = data[1].data
        if gratings[i]=='E140M':
            array_sizes.append(data.size*1024)
        else:
            array_sizes.append(len(data['Wavelength'][0]))
    array_len = max(array_sizes)

    waves     = np.zeros((len(spec_names), array_len))
    fluxes    = np.zeros((len(spec_names), array_len))
    flux_errs = np.zeros((len(spec_names), array_len))
    masks     = np.zeros((len(spec_names), array_len))

    for i in range(len(spec_names)):
        try:
            data = fits.open(path+'%s/%s/%s_x1d.fits' %(Identifier,spec_names[i],spec_names[i]))[1].data
        except FileNotFoundError:
            data = fits.open(path+'%s/%s/%s_sx1.fits' %(Identifier,spec_names[i],spec_names[i]))[1].data
        if gratings[i]=="E140M":
            wavelength = []
            flux       = []
            fluxerr    = []
            DQ         = []
            for t in np.arange(0,data.size,1):
                wavelength = np.append(wavelength, data['WAVELENGTH'][t])
                flux       = np.append(flux, data['FLUX'][t])
                fluxerr    = np.append(fluxerr, data['ERROR'][t])
                DQ         = np.append(DQ, data['DQ'][t])
        else:
            wavelength = data['WAVELENGTH'][0]
            flux       = data['FLUX'][0]
            fluxerr    = data['ERROR'][0]
            DQ         = data['DQ'][0]

        flux_wmask = flux.copy()
        err_wmask  = fluxerr.copy()
        sel        = ((wavelength >= 1215.0) & (wavelength <= 1216.0))
        masks[i,:len(err_wmask)][err_wmask==0.] = 1
        # A pixel whose flux is exactly 0.0 is not a measurement of zero flux: it is a
        # dead or unilluminated pixel that the archive flags in DQ, and 43% of
        # NGC 3783's COS pixels in the C IV window are of this kind, every one of them
        # carrying a non-zero error.  Cut_Edge_Pix already defines a good pixel as
        # flux != 0 and fluxerr != 0, but uses that only to trim the ends of the array,
        # so interior dead pixels reached the co-add with full 1/err^2 weight and pulled
        # the weighted median toward zero.  This applies the same definition everywhere.
        masks[i,:len(err_wmask)][(flux_wmask == 0.0)] = 1
        waves[i,:len(err_wmask)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                wavelength, array_len, "wavelength", False, z, "%s - %s"%(Identifier,spec_names[i]), "STIS")
        fluxes[i,:len(err_wmask)]    = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                flux_wmask, array_len, "flux", False, z, "%s - %s"%(Identifier,spec_names[i]), "STIS")
        flux_errs[i,:len(err_wmask)] = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                err_wmask, array_len, "flux error", False, z, "%s - %s"%(Identifier,spec_names[i]), "STIS")
        masks[i,:len(err_wmask)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                masks[i,:len(err_wmask)], array_len, "masks", False, z, "%s - %s"%(Identifier,spec_names[i]), "STIS")

    return waves, fluxes, flux_errs, masks


def read_fos(Identifier, path, z):
    # AP: HST original read_fos — no try-except, no bad_indices removal.

    obs_details = pd.read_csv(path+"%s/NecessaryParams.csv"%Identifier)
    gratings    = obs_details["filters"].values
    spec_names  = obs_details["obs_id"].values

    array_lens = np.array([], dtype=int)
    for spectrum in spec_names:
        wave = fits.open(path+"%s/%s/%s_c0f.fits" % (Identifier, spectrum, spectrum))[0].data
        array_lens = np.append(array_lens, wave.shape[-1])
    array_len = max(array_lens)

    waves     = np.zeros((len(spec_names), array_len))
    fluxes    = np.zeros((len(spec_names), array_len))
    flux_errs = np.zeros((len(spec_names), array_len))
    masks     = np.zeros((len(spec_names), array_len))

    def accum_flag(index):
        return _is_cumulative_accum(flux, index)

    for i in range(len(spec_names)):
        wave = fits.open(path+"%s/%s/%s_c0f.fits" % (Identifier, spec_names[i], spec_names[i]))[0].data

        ind = -1

        if len(wave.shape) > 1:
            nzero = (wave[ind]>0.)
            if (wave[ind][nzero][0] > wave[ind][nzero][-1]):
                wavelength = wave[ind][::-1]
            else:
                wavelength = wave[ind][:]
        else:
            nzero = (wave>0.)
            if (wave[nzero][0] > wave[nzero][-1]):
                wavelength = wave[::-1]
            else:
                wavelength = wave[:]

        flux    = np.zeros(len(wavelength))
        fluxerr = np.zeros(len(wavelength))
        DQ      = np.zeros(len(wavelength))

        obs_flux    = fits.open(path+'%s/%s/%s_c1f.fits' %(Identifier,spec_names[i],spec_names[i]))[0].data
        obs_fluxerr = fits.open(path+'%s/%s/%s_c2f.fits' %(Identifier,spec_names[i],spec_names[i]))[0].data
        obs_DQ      = fits.open(path+'%s/%s/%s_cqf.fits' %(Identifier,spec_names[i],spec_names[i]))[0].data

        if len(wave.shape) > 1:
            if accum_flag(ind):
                if (wave[ind][nzero][0] > wave[ind][nzero][-1]):
                   flux    = obs_flux[ind][::-1]
                   fluxerr = obs_fluxerr[ind][::-1]
                   DQ      = obs_DQ[ind][::-1]
                else:
                   flux    = obs_flux[ind]
                   fluxerr = obs_fluxerr[ind]
                   DQ      = obs_DQ[ind]
            else:
                for l in range(len(wavelength)):
                    if (wave[ind][nzero][0] > wave[ind][nzero][-1]):
                        flux[l]    = np.mean(obs_flux[:,-l-1])
                        fluxerr[l] = np.sqrt(sum(obs_fluxerr[:,-l-1]**2)) / (obs_fluxerr.shape[0]-1)  # AP: restored /N-1 (hypothesized Aug 2022)
                        DQ[l]      = sum(obs_DQ[:,-l-1])
                    else:
                        flux[l]    = np.mean(obs_flux[:,l])
                        fluxerr[l] = np.sqrt(sum(obs_fluxerr[:,l]**2)) / (obs_fluxerr.shape[0]-1)  # AP: restored /N-1 (hypothesized Aug 2022)
                        DQ[l]      = sum(obs_DQ[:,l])
        else:
            if (wave[nzero][0] > wave[nzero][-1]):
                flux    = obs_flux[::-1]
                fluxerr = obs_fluxerr[::-1]
                DQ      = obs_DQ[::-1]
            else:
                flux    = obs_flux
                fluxerr = obs_fluxerr
                DQ      = obs_DQ

        sel                       = (wavelength >= 1205) & (wavelength <= 1225)
        mask                      = DQ>0
        flux_wmask                = flux.copy()
        err_wmask                 = fluxerr.copy()
        masks[i,:len(err_wmask)][err_wmask==0.] = 1
        # A pixel whose flux is exactly 0.0 is not a measurement of zero flux: it is a
        # dead or unilluminated pixel that the archive flags in DQ, and 43% of
        # NGC 3783's COS pixels in the C IV window are of this kind, every one of them
        # carrying a non-zero error.  Cut_Edge_Pix already defines a good pixel as
        # flux != 0 and fluxerr != 0, but uses that only to trim the ends of the array,
        # so interior dead pixels reached the co-add with full 1/err^2 weight and pulled
        # the weighted median toward zero.  This applies the same definition everywhere.
        masks[i,:len(err_wmask)][(flux_wmask == 0.0)] = 1

        waves[i,:len(flux)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                wavelength, array_len, "wavelength", False, z, "%s - %s"%(Identifier,spec_names[i]),"FOS")
        fluxes[i,:len(flux)]    = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                flux_wmask, array_len, "flux", False, z, "%s - %s"%(Identifier,spec_names[i]),"FOS")
        flux_errs[i,:len(flux)] = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                err_wmask, array_len, "flux error", False, z, "%s - %s"%(Identifier,spec_names[i]),"FOS")
        masks[i,:len(flux)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                masks[i,:len(err_wmask)], array_len, "masks", False, z, "%s - %s"%(Identifier,spec_names[i]),"FOS")

    return waves, fluxes, flux_errs, masks


# ── Flat-layout readers (master catalog / MAST download) ─────────────────────
# Files land directly in the instrument folder with no object subdirectory
# and no NecessaryParams.csv.  Grating is read from the FITS primary header.

def read_cos_flat(name, path, z):
    """Read COS x1d files from a flat directory (no NecessaryParams.csv)."""
    # Only use per-exposure _x1d.fits products (matches Trevor's legacy convention).
    # _x1dsum.fits files are CalCOS per-visit coadds of the same exposures, so
    # including them would double-count the same photons in coadd.py.
    _prov_reset()
    fn_list = sorted(glob.glob(os.path.join(glob.escape(path), '*_x1d.fits')))
    for fn in sorted(glob.glob(os.path.join(glob.escape(path), '*_x1dsum.fits'))):
        _prov(fn, False, "x1dsum: CalCOS per-visit coadd of exposures already read individually")
    if not fn_list:
        raise FileNotFoundError("No COS x1d files found in %s" % path)

    # Drop files where CalCOS extraction failed and produced an empty BinTable
    # (zero rows). These are still archived by MAST and would crash the readers
    # downstream. Failed visits are typically re-observed under a new ASN_ID.
    _right = [fn for fn in fn_list if not _wrong_instrument(fn, 'COS')]
    if len(_right) < len(fn_list):
        for fn in fn_list:
            if fn not in _right:
                _prov(fn, False, "not a COS file (INSTRUME says otherwise)")
        print("  %s COS: skipping %d file(s) filed here but written by another "
              "instrument: %s" % (name, len(fn_list) - len(_right),
                                  ", ".join(os.path.basename(f) for f in fn_list
                                            if f not in _right)))
    fn_list = _right
    if not fn_list:
        raise FileNotFoundError("No COS x1d files found in %s" % path)

    _nonempty = [fn for fn in fn_list if fits.open(fn)[1].data is not None
                 and len(fits.open(fn)[1].data) > 0]
    for fn in fn_list:
        if fn not in _nonempty:
            _prov(fn, False, "empty extraction (zero-row BinTable)")
    fn_list = _nonempty
    if not fn_list:
        raise FileNotFoundError("No non-empty COS x1d files found in %s" % path)

    kept = [fn for fn in fn_list if _carries_flux(fn)]
    for fn in fn_list:
        if fn not in kept:
            _prov(fn, False, "no non-zero flux")
    if len(kept) < len(fn_list):
        print("  %s COS: skipping %d exposure(s) with no non-zero flux: %s"
              % (name, len(fn_list) - len(kept),
                 ", ".join(os.path.basename(f) for f in fn_list if f not in kept)))
    fn_list = kept
    if not fn_list:
        raise FileNotFoundError("No COS x1d files carrying flux in %s" % path)

    # AP 2026-06-17: TEMP cap for monitoring targets -- see _COS_EXPOSURE_CAP above.
    if name in _COS_EXPOSURE_CAP and len(fn_list) > _COS_EXPOSURE_CAP[name]:
        n_keep = _COS_EXPOSURE_CAP[name]
        ranked = sorted(fn_list, key=_cos_median_snr, reverse=True)
        for fn in ranked[n_keep:]:
            _prov(fn, False, "capped: not among the %d highest-S/N exposures (TEMP)" % n_keep)
        fn_list = sorted(ranked[:n_keep])  # re-sort by name for deterministic order
        print("CAP COS %s: kept %d highest-S/N of %d exposures (TEMP -- see "
              "Migration_Log.md)" % (name, n_keep, len(ranked)), flush=True)

    gratings = []
    for fn in fn_list:
        hdr = fits.open(fn)[0].header
        gratings.append(hdr.get('OPT_ELEM', hdr.get('FILTER', 'UNKNOWN')))
        _prov(fn, True, "")

    array_sizes = []
    n_rows = 0
    for i, fn in enumerate(fn_list):
        data = fits.open(fn)[1].data
        for _w, _f, _e, _q in _rows_of(data):
            array_sizes.append(len(_w))
            n_rows += 1
    array_len = max(array_sizes)

    waves     = np.zeros((n_rows, array_len))
    fluxes    = np.zeros((n_rows, array_len))
    flux_errs = np.zeros((n_rows, array_len))
    masks     = np.zeros((n_rows, array_len))

    # One entry per detector segment, not per file.  COS segments overlap
    # and each carries roughly 2000 dead pixels at its edges whose flux is
    # zero but whose error is not, so the err==0 mask rule does not catch
    # them.  Concatenated into a single row those zeros land inside the
    # other segment's range, where edge trimming cannot reach, and the
    # co-addition averages them into the line: Mrk 290 came out with flux
    # pinned to zero straight across C IV.  Kept as separate rows they are
    # trimmed and inverse-variance weighted like any other spectrum.
    i = -1
    for fn in fn_list:
        base = os.path.basename(fn).replace('_x1d.fits', '').replace('_x1dsum.fits', '')
        data = fits.open(fn)[1].data
        rows = _rows_of(data)
        for t, (wavelength, flux, fluxerr, DQ) in enumerate(rows):
            i += 1
            spec_id = base if len(rows) == 1 else '%s.%d' % (base, t)
            flux_wmask = flux.copy()
            err_wmask  = fluxerr.copy()
            # Slice to len(err_wmask) so that COS NUV stripes (1274 px) don't crash
            # when this object also has COS FUV segments (16384 px) and array_len is
            # the max. Matches the FOS/STIS reader pattern; padding stays as zeros
            # and coadd.py filters with waves[waves!=0].
            masks[i,:len(err_wmask)][(err_wmask == 0.0)] = 1
            # A pixel whose flux is exactly 0.0 is not a measurement of zero flux: it is a
            # dead or unilluminated pixel that the archive flags in DQ, and 43% of
            # NGC 3783's COS pixels in the C IV window are of this kind, every one of them
            # carrying a non-zero error.  Cut_Edge_Pix already defines a good pixel as
            # flux != 0 and fluxerr != 0, but uses that only to trim the ends of the array,
            # so interior dead pixels reached the co-add with full 1/err^2 weight and pulled
            # the weighted median toward zero.  This applies the same definition everywhere.
            masks[i,:len(err_wmask)][(flux_wmask == 0.0)] = 1
            waves[i,:len(err_wmask)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask,
                                                    wavelength, array_len, "wavelength", False, z,
                                                    "%s - %s" % (name, spec_id), "COS")
            fluxes[i,:len(err_wmask)]    = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask,
                                                    flux_wmask, array_len, "flux", False, z,
                                                    "%s - %s" % (name, spec_id), "COS")
            flux_errs[i,:len(err_wmask)] = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask,
                                                    err_wmask, array_len, "flux error", False, z,
                                                    "%s - %s" % (name, spec_id), "COS")
            masks[i,:len(err_wmask)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask,
                                                    masks[i,:len(err_wmask)], array_len, "masks", False, z,
                                                    "%s - %s" % (name, spec_id), "COS")

    return waves, fluxes, flux_errs, masks


def read_stis_flat(name, path, z):
    """Read STIS x1d/sx1 files from a flat directory (no NecessaryParams.csv)."""
    # _x1d.fits and _sx1.fits are detector-specific products, not duplicates:
    # MAMA modes (G140L/M, G230L/M, E140M, E230M, PRISM) → _x1d.fits;
    # CCD modes with CR-SPLIT (G430L/M, G750L/M, G230LB, G230MB) → _sx1.fits.
    # They are mutually exclusive per rootname, so include both.
    _prov_reset()
    fn_list = sorted(
        glob.glob(os.path.join(glob.escape(path), '*_x1d.fits')) +
        glob.glob(os.path.join(glob.escape(path), '*_sx1.fits'))
    )
    if not fn_list:
        raise FileNotFoundError("No STIS x1d/sx1 files found in %s" % path)

    # Defensive: drop files where pipeline extraction produced an empty BinTable.
    _right = [fn for fn in fn_list if not _wrong_instrument(fn, 'STIS')]
    if len(_right) < len(fn_list):
        for fn in fn_list:
            if fn not in _right:
                _prov(fn, False, "not a STIS file (INSTRUME says otherwise)")
        print("  %s STIS: skipping %d file(s) filed here but written by another "
              "instrument" % (name, len(fn_list) - len(_right)))
    fn_list = _right
    if not fn_list:
        raise FileNotFoundError("No STIS x1d/sx1 files found in %s" % path)

    # Not currently observed in STIS data, but matches the COS reader hardening.
    _nonempty = [fn for fn in fn_list if fits.open(fn)[1].data is not None
                 and len(fits.open(fn)[1].data) > 0]
    for fn in fn_list:
        if fn not in _nonempty:
            _prov(fn, False, "empty extraction (zero-row BinTable)")
    fn_list = _nonempty
    if not fn_list:
        raise FileNotFoundError("No non-empty STIS x1d/sx1 files found in %s" % path)

    kept = [fn for fn in fn_list if _carries_flux(fn)]
    for fn in fn_list:
        if fn not in kept:
            _prov(fn, False, "no non-zero flux")
    if len(kept) < len(fn_list):
        print("  %s STIS: skipping %d exposure(s) with no non-zero flux: %s"
              % (name, len(fn_list) - len(kept),
                 ", ".join(os.path.basename(f) for f in fn_list if f not in kept)))
    fn_list = kept
    if not fn_list:
        raise FileNotFoundError("No STIS x1d/sx1 files carrying flux in %s" % path)

    gratings = []
    for fn in fn_list:
        hdr = fits.open(fn)[0].header
        gratings.append(hdr.get('OPT_ELEM', hdr.get('FILTER', 'UNKNOWN')))
        _prov(fn, True, "")

    # Echelle orders are stitched into one spectrum per exposure; every other
    # STIS mode writes a single row anyway.  See _rows_of for why.
    stitch_of = {fn: (str(g).upper() in _ECHELLE) for fn, g in zip(fn_list, gratings)}

    array_sizes = []
    n_rows = 0
    for i, fn in enumerate(fn_list):
        data = fits.open(fn)[1].data
        for _w, _f, _e, _q in _rows_of(data, stitch=stitch_of[fn]):
            array_sizes.append(len(_w))
            n_rows += 1
    array_len = max(array_sizes)

    waves     = np.zeros((n_rows, array_len))
    fluxes    = np.zeros((n_rows, array_len))
    flux_errs = np.zeros((n_rows, array_len))
    masks     = np.zeros((n_rows, array_len))

    # One entry per detector segment, not per file.  COS segments overlap
    # and each carries roughly 2000 dead pixels at its edges whose flux is
    # zero but whose error is not, so the err==0 mask rule does not catch
    # them.  Concatenated into a single row those zeros land inside the
    # other segment's range, where edge trimming cannot reach, and the
    # co-addition averages them into the line: Mrk 290 came out with flux
    # pinned to zero straight across C IV.  Kept as separate rows they are
    # trimmed and inverse-variance weighted like any other spectrum.
    i = -1
    for fn in fn_list:
        base = os.path.basename(fn).replace('_x1d.fits', '').replace('_sx1.fits', '')
        data = fits.open(fn)[1].data
        rows = _rows_of(data, stitch=stitch_of[fn])
        for t, (wavelength, flux, fluxerr, DQ) in enumerate(rows):
            i += 1
            spec_id = base if len(rows) == 1 else '%s.%d' % (base, t)
            flux_wmask = flux.copy()
            err_wmask  = fluxerr.copy()
            masks[i,:len(err_wmask)][err_wmask == 0.] = 1
            # A pixel whose flux is exactly 0.0 is not a measurement of zero flux: it is a
            # dead or unilluminated pixel that the archive flags in DQ, and 43% of
            # NGC 3783's COS pixels in the C IV window are of this kind, every one of them
            # carrying a non-zero error.  Cut_Edge_Pix already defines a good pixel as
            # flux != 0 and fluxerr != 0, but uses that only to trim the ends of the array,
            # so interior dead pixels reached the co-add with full 1/err^2 weight and pulled
            # the weighted median toward zero.  This applies the same definition everywhere.
            masks[i,:len(err_wmask)][(flux_wmask == 0.0)] = 1
            waves[i,:len(err_wmask)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask,
                                                    wavelength, array_len, "wavelength", False, z,
                                                    "%s - %s" % (name, spec_id), "STIS")
            fluxes[i,:len(err_wmask)]    = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask,
                                                    flux_wmask, array_len, "flux", False, z,
                                                    "%s - %s" % (name, spec_id), "STIS")
            flux_errs[i,:len(err_wmask)] = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask,
                                                    err_wmask, array_len, "flux error", False, z,
                                                    "%s - %s" % (name, spec_id), "STIS")
            masks[i,:len(err_wmask)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask,
                                                    masks[i,:len(err_wmask)], array_len, "masks", False, z,
                                                    "%s - %s" % (name, spec_id), "STIS")

    return waves, fluxes, flux_errs, masks


def read_fos_flat(name, path, z):
    """
    Read FOS c0f/c1f/c2f/cqf files from a flat directory (no NecessaryParams.csv).
    Discovers exposures by globbing *_c0f.fits; algorithm is identical to read_fos().
    """
    _prov_reset()
    wave_files = sorted(glob.glob(os.path.join(glob.escape(path), '*_c0f.fits')))
    if not wave_files:
        raise FileNotFoundError("No FOS c0f files found in %s" % path)

    spec_names = [os.path.basename(f).replace('_c0f.fits', '') for f in wave_files]
    for _fn in wave_files:
        _prov(_fn, True, "")

    array_lens = np.array([], dtype=int)
    for sn in spec_names:
        wave = fits.open(os.path.join(path, '%s_c0f.fits' % sn))[0].data
        array_lens = np.append(array_lens, wave.shape[-1])
    array_len = max(array_lens)

    def accum_flag(index, flux_arr):
        return _is_cumulative_accum(flux_arr, index)

    def _grid_groups(wave):
        """Indices of the groups, split by which wavelength solution they use.

        An FOS exposure's groups are normally repeated readouts on ONE grid, and
        the reader averages them by pixel index.  A SPECTROPOLARIMETRY exposure
        is different: its groups alternate between the two polarisation
        channels, whose wavelength solutions differ -- on 3C 273's H19 exposures
        by 6.11 A, about 16 pixels and 833 km/s -- so averaging them by index
        smears every line across 6 A and drags its centroid several hundred
        km/s.  That is how 3C 273's C III] came to sit 1285 km/s blueward of
        systemic in the co-add (index 111), and Mrk 486's only C IV-covering
        spectrum is built entirely from such files (index 51).

        Splitting by grid and emitting one row per channel costs nothing
        elsewhere -- an exposure whose groups share a grid yields a single
        group, exactly as before -- and lets the rebin step align the channels
        by wavelength, which is where alignment belongs.  Interpolating one
        channel onto the other's grid would work too and is deliberately not
        done: the co-addition is built on "no interpolation", and COS detector
        segments already take this same separate-rows treatment.
        """
        nz = wave > 0.
        groups, reps = [], []
        for g in range(wave.shape[0]):
            ok = nz[g]
            placed = False
            for j, r in enumerate(reps):
                both = ok & nz[r]
                if both.any() and np.allclose(wave[g][both], wave[r][both],
                                              rtol=0, atol=1e-3):
                    groups[j].append(g)
                    placed = True
                    break
            if not placed:
                reps.append(g)
                groups.append([g])
        return groups

    #: One entry per row that will be handed to the co-adder: (name, wavelength,
    #: flux, fluxerr, DQ).  Usually one per file; more when a file's groups use
    #: more than one wavelength solution (see _grid_groups).
    entries = []
    for sn in spec_names:
        wave        = fits.open(os.path.join(path, '%s_c0f.fits' % sn))[0].data
        obs_flux    = fits.open(os.path.join(path, '%s_c1f.fits' % sn))[0].data
        obs_fluxerr = fits.open(os.path.join(path, '%s_c2f.fits' % sn))[0].data
        cqf_path    = os.path.join(path, '%s_cqf.fits' % sn)
        if os.path.exists(cqf_path):
            obs_DQ = fits.open(cqf_path)[0].data
        else:
            obs_DQ = np.zeros_like(obs_flux)

        ind = -1

        if len(np.shape(wave)) > 1:
            _gg = _grid_groups(np.asarray(wave, float))
            if len(_gg) > 1:
                print("FOS %s: %d groups on %d wavelength solutions "
                      "(spectropolarimetry); kept as %d separate spectra rather "
                      "than averaged by pixel index"
                      % (sn, wave.shape[0], len(_gg), len(_gg)), flush=True)
                for k, gidx in enumerate(_gg):
                    sub_w = np.asarray(wave, float)[gidx]
                    sub_f = np.asarray(obs_flux, float)[gidx]
                    sub_e = np.asarray(obs_fluxerr, float)[gidx]
                    sub_q = np.asarray(obs_DQ, float)[gidx]
                    entries.append(("%s.%d" % (sn, k), sub_w, sub_f, sub_e, sub_q))
                continue
        entries.append((sn, wave, obs_flux, obs_fluxerr, obs_DQ))

    waves     = np.zeros((len(entries), array_len))
    fluxes    = np.zeros((len(entries), array_len))
    flux_errs = np.zeros((len(entries), array_len))
    masks     = np.zeros((len(entries), array_len))

    for i, (sn, wave, obs_flux, obs_fluxerr, obs_DQ) in enumerate(entries):
        ind = -1

        if len(wave.shape) > 1:
            nzero = (wave[ind] > 0.)
            wavelength = wave[ind][::-1] if wave[ind][nzero][0] > wave[ind][nzero][-1] else wave[ind][:]
        else:
            nzero = (wave > 0.)
            wavelength = wave[::-1] if wave[nzero][0] > wave[nzero][-1] else wave[:]

        flux    = np.zeros(len(wavelength))
        fluxerr = np.zeros(len(wavelength))
        DQ      = np.zeros(len(wavelength))

        if len(wave.shape) > 1:
            if accum_flag(ind, obs_flux):
                if wave[ind][nzero][0] > wave[ind][nzero][-1]:
                    flux    = obs_flux[ind][::-1]
                    fluxerr = obs_fluxerr[ind][::-1]
                    DQ      = obs_DQ[ind][::-1]
                else:
                    flux    = obs_flux[ind]
                    fluxerr = obs_fluxerr[ind]
                    DQ      = obs_DQ[ind]
            else:
                for l in range(len(wavelength)):
                    if wave[ind][nzero][0] > wave[ind][nzero][-1]:
                        flux[l]    = np.mean(obs_flux[:, -l-1])
                        fluxerr[l] = np.sqrt(sum(obs_fluxerr[:, -l-1]**2)) / (obs_fluxerr.shape[0]-1)
                        DQ[l]      = sum(obs_DQ[:, -l-1])
                    else:
                        flux[l]    = np.mean(obs_flux[:, l])
                        fluxerr[l] = np.sqrt(sum(obs_fluxerr[:, l]**2)) / (obs_fluxerr.shape[0]-1)
                        DQ[l]      = sum(obs_DQ[:, l])
        else:
            if wave[nzero][0] > wave[nzero][-1]:
                flux    = obs_flux[::-1]
                fluxerr = obs_fluxerr[::-1]
                DQ      = obs_DQ[::-1]
            else:
                flux    = obs_flux
                fluxerr = obs_fluxerr
                DQ      = obs_DQ

        flux_wmask = flux.copy()
        err_wmask  = fluxerr.copy()
        masks[i,:len(err_wmask)][err_wmask == 0.] = 1
        # A pixel whose flux is exactly 0.0 is not a measurement of zero flux: it is a
        # dead or unilluminated pixel that the archive flags in DQ, and 43% of
        # NGC 3783's COS pixels in the C IV window are of this kind, every one of them
        # carrying a non-zero error.  Cut_Edge_Pix already defines a good pixel as
        # flux != 0 and fluxerr != 0, but uses that only to trim the ends of the array,
        # so interior dead pixels reached the co-add with full 1/err^2 weight and pulled
        # the weighted median toward zero.  This applies the same definition everywhere.
        masks[i,:len(err_wmask)][(flux_wmask == 0.0)] = 1

        waves[i,:len(flux)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask,
                                                wavelength, array_len, "wavelength", False, z,
                                                "%s - %s" % (name, sn), "FOS")
        fluxes[i,:len(flux)]    = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask,
                                                flux_wmask, array_len, "flux", False, z,
                                                "%s - %s" % (name, sn), "FOS")
        flux_errs[i,:len(flux)] = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask,
                                                err_wmask, array_len, "flux error", False, z,
                                                "%s - %s" % (name, sn), "FOS")
        masks[i,:len(flux)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask,
                                                masks[i,:len(err_wmask)], array_len, "masks", False, z,
                                                "%s - %s" % (name, sn), "FOS")

    return waves, fluxes, flux_errs, masks
