"""Synthetic spectra with known answers, one complication at a time.

GTR, 2026-11-13: "Let's also add some more 'cartoon' examples.  Things that I can
show in a talk to demonstrate the empirical code testing that was done.  Maybe
that will help decide the _combined_error issue too?"

These are stronger than the real-spectrum regression fixtures, because the input
is known analytically.  A real fixture can only say "nothing changed"; a cartoon
can say "the code recovers the right answer", and when it does not it says by how
much.  Each cartoon isolates ONE complication that cost this project time:

    single_clean            the control: one exposure, flat noise
    independent_readouts    20 groups, NREAD=1 -- must be AVERAGED (index 362)
    cumulative_groups       20 groups, NREAD=20, errors falling as 1/sqrt(k+1)
                            -- the last group alone is the exposure (index 306)
    nine_agreeing           9 exposures of the same truth: the co-added error
                            must fall as 1/sqrt(9), i.e. to sigma/3
    nine_disagreeing        9 exposures whose levels differ by 20 per cent, far
                            beyond their errors: the combined error must be
                            INFLATED to reflect that, not reduced
    zero_error_pixel        one pixel quoted at 1e-7 of the real error (index 233)
    two_epochs_offset       two gratings whose overlap differs by 25 per cent
                            (index 70, I Zw 1, 18 per cent in reality)

THE TRUTH is a power-law continuum plus one Gaussian C IV line at a chosen
blueshift and equivalent width, written in the observed frame for a chosen
redshift, in FOS c0f/c1f/c2f form so that the group logic in read_spec_data is
exercised.  Everything is seeded, so the files are reproducible byte-for-byte.

    python make_cartoons.py          # write tests/spectra/cartoons/<name>/
"""
import json
import os

import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cartoons")

CIV = 1549.48
Z = 0.1
#: the answer every cartoon should return
TRUE_BLUESHIFT = 1000.0          # km/s, positive = blueshifted
TRUE_EW = 40.0                   # Angstrom, rest frame
TRUE_FWHM = 4000.0               # km/s
CONT_1450 = 1.0e-14              # erg/s/cm2/A at 1450 A rest
ALPHA = -1.5                     # f_lambda power law
SIGMA_FRAC = 0.05                # per-pixel noise as a fraction of the continuum


def truth(wave_rest, ew=None):
    """Continuum plus the C IV line, in rest-frame wavelength.

    `ew` overrides TRUE_EW for the exposures that must differ in LINE STRENGTH
    rather than in level -- see cartoon_nine_varying_line.
    """
    cont = CONT_1450 * (wave_rest / 1450.0) ** ALPHA
    lam0 = CIV * (1.0 - TRUE_BLUESHIFT / 299792.458)
    sig = lam0 * (TRUE_FWHM / 299792.458) / 2.3548
    # EW is defined against the continuum, so scale the Gaussian to give it
    peak = (TRUE_EW if ew is None else ew) / (sig * np.sqrt(2.0 * np.pi))
    line = peak * np.exp(-0.5 * ((wave_rest - lam0) / sig) ** 2)
    return cont * (1.0 + line)


def grid(lo_obs, hi_obs, n):
    return np.linspace(lo_obs, hi_obs, n)


def write_fos(path_stem, wave_obs, groups_flux, groups_err, nread, grndmode,
              grating, exptime, date):
    """FOS-style c0f/c1f/c2f with one row per group."""
    for ext, data in (("c0f", np.tile(wave_obs, (len(groups_flux), 1))),
                      ("c1f", np.asarray(groups_flux)),
                      ("c2f", np.asarray(groups_err))):
        h = fits.Header()
        h["FGWA_ID"] = grating
        h["NREAD"] = nread
        h["GRNDMODE"] = grndmode
        h["EXPTIME"] = exptime
        h["DATE-OBS"] = date
        h["ROOTNAME"] = os.path.basename(path_stem)
        fits.PrimaryHDU(np.asarray(data, dtype=">f4"), header=h).writeto(
            "%s_%s.fits" % (path_stem, ext), overwrite=True)


def one_exposure(rng, wave_obs, n_groups, cumulative, level=1.0,
                 sigma_frac=SIGMA_FRAC, ew=None):
    """Return (fluxes, errors) for an exposure's groups.

    cumulative: group k holds k+1 readouts' worth of time, so its noise falls as
    1/sqrt(k+1) while the CALIBRATED flux level stays put -- which is exactly why
    a test on the flux means cannot tell the two cases apart.
    """
    w_rest = wave_obs / (1.0 + Z)
    base = truth(w_rest, ew=ew) * level
    cont = CONT_1450 * (w_rest / 1450.0) ** ALPHA * level
    fl, er = [], []
    for k in range(n_groups):
        scale = np.sqrt(k + 1.0) if cumulative else 1.0
        sd = cont * sigma_frac / scale
        fl.append(base + rng.normal(0.0, sd))
        er.append(sd)
    return np.array(fl), np.array(er)


def main():
    os.makedirs(OUT, exist_ok=True)
    wave_obs = grid(1300.0 * (1 + Z), 1900.0 * (1 + Z), 2064)
    made = []

    def emit(name, note, builder, extra=None):
        d = os.path.join(OUT, name, "exposures")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        builder(d)
        meta = dict(name=name, inst="FOS", z=Z, obj="CARTOON %s" % name,
                    synthetic=True, tests=note,
                    truth=dict(blueshift_kms=TRUE_BLUESHIFT, ew_A=TRUE_EW,
                               fwhm_kms=TRUE_FWHM, cont_1450=CONT_1450,
                               alpha=ALPHA, sigma_fraction=SIGMA_FRAC))
        if extra:
            meta["truth"].update(extra)
        json.dump(meta, open(os.path.join(OUT, name, "fixture.json"), "w"), indent=1)
        made.append(name)
        print("  %-26s %s" % (name, note[:68]))

    def b_single(d):
        rng = np.random.default_rng(1)
        f, e = one_exposure(rng, wave_obs, 1, False)
        write_fos(os.path.join(d, "cart0001"), wave_obs, f, e, 1,
                  "RAPID-READOUT", "H19", 1000.0, "01/01/95")
    emit("cartoon_single_clean",
         "the control: one exposure, one group, flat noise", b_single)

    def b_indep(d):
        rng = np.random.default_rng(2)
        f, e = one_exposure(rng, wave_obs, 20, False)
        write_fos(os.path.join(d, "cart0002"), wave_obs, f, e, 1,
                  "RAPID-READOUT", "H19", 1000.0, "01/01/95")
    emit("cartoon_independent_readouts",
         "20 independent readouts (NREAD=1) that must be averaged: noise should "
         "fall as 1/sqrt(20)", b_indep, dict(n_groups=20, expect_noise_gain=np.sqrt(20)))

    def b_cumul(d):
        rng = np.random.default_rng(3)
        f, e = one_exposure(rng, wave_obs, 20, True)
        write_fos(os.path.join(d, "cart0003"), wave_obs, f, e, 20,
                  "SPECTROSCOPY", "H19", 1000.0, "01/01/95")
    emit("cartoon_cumulative_groups",
         "20 CUMULATIVE groups (NREAD=20): flux level flat, errors fall as "
         "1/sqrt(k+1); the last group alone is the exposure", b_cumul,
         dict(n_groups=20, cumulative=True))

    def b_nine_agree(d):
        rng = np.random.default_rng(4)
        for i in range(9):
            f, e = one_exposure(rng, wave_obs, 1, False)
            write_fos(os.path.join(d, "cart01%02d" % i), wave_obs, f, e, 1,
                      "RAPID-READOUT", "H19", 1000.0, "01/01/95")
    emit("cartoon_nine_agreeing",
         "9 exposures of the SAME truth: the co-added error must fall to "
         "sigma/3, so scatter/error should be 1", b_nine_agree,
         dict(n_exposures=9, expect_combined_error_gain=3.0))

    def b_nine_disagree(d):
        rng = np.random.default_rng(5)
        for i in range(9):
            lv = 1.0 + 0.20 * ((i % 3) - 1)      # -20%, 0, +20%
            f, e = one_exposure(rng, wave_obs, 1, False, level=lv)
            write_fos(os.path.join(d, "cart02%02d" % i), wave_obs, f, e, 1,
                      "RAPID-READOUT", "H19", 1000.0, "01/01/95")
    # WHAT THIS ACTUALLY TESTS, corrected 2026-11-14.  It was built to make the
    # exposures disagree so that the chi2 inflation in _combined_error would
    # fire, and it does not: coadd.py divides each exposure by its OWN continuum
    # before co-adding (coadd.py, "new_binned_fluxes /= continuum"), which
    # removes a pure multiplicative level difference exactly.  Measured, the
    # 20 per cent spread becomes 0.7 per cent and chi2/dof is 1.01 -- the same
    # as cartoon_nine_agreeing at 0.96.  The pipeline is right and the cartoon's
    # claim was wrong.  It is kept, because "a level offset must NOT inflate the
    # error" is worth pinning down, and cartoon_nine_varying_line below is the
    # one that actually exercises the inflation.
    emit("cartoon_nine_disagreeing",
         "9 exposures whose levels differ by 20 per cent: the per-exposure "
         "continuum division removes this exactly, so the error must NOT be "
         "inflated -- chi2/dof stays at 1",
         b_nine_disagree, dict(n_exposures=9, level_spread=0.20,
                               expect_chi2_dof=1.0, expect_inflation=False))

    def b_nine_vary_line(d):
        rng = np.random.default_rng(9)
        for i in range(9):
            # EW 30 -> 50 A across the nine.  Dividing by the continuum cannot
            # remove this: the continuum is the same and the LINE differs.
            f, e = one_exposure(rng, wave_obs, 1, False,
                                ew=30.0 + 20.0 * i / 8.0)
            write_fos(os.path.join(d, "cart05%02d" % i), wave_obs, f, e, 1,
                      "RAPID-READOUT", "H19", 1000.0, "01/01/95")
    emit("cartoon_nine_varying_line",
         "9 exposures whose LINE STRENGTH runs EW 30-50 A, a difference the "
         "continuum division cannot remove: the combined error must be "
         "INFLATED inside C IV and not outside it",
         b_nine_vary_line, dict(n_exposures=9, ew_lo=30.0, ew_hi=50.0,
                                ew_A=40.0, expect_inflation=True))

    def b_zero_err(d):
        rng = np.random.default_rng(6)
        f, e = one_exposure(rng, wave_obs, 1, False)
        k = len(wave_obs) // 3
        f[0][k:k + 4] = 1e-21
        e[0][k:k + 4] = 1e-21
        write_fos(os.path.join(d, "cart0300"), wave_obs, f, e, 1,
                  "RAPID-READOUT", "H19", 1000.0, "01/01/95")
    emit("cartoon_zero_error_pixel",
         "four pixels quoted at 1e-21: no information, but weight ~1e42 unless "
         "they are masked", b_zero_err, dict(n_bad_pixels=4))

    def b_two_epoch(d):
        rng = np.random.default_rng(7)
        blue = grid(1300.0 * (1 + Z), 1620.0 * (1 + Z), 1032)
        red = grid(1560.0 * (1 + Z), 1900.0 * (1 + Z), 1032)
        fb, eb = one_exposure(rng, blue, 1, False, level=0.80)
        write_fos(os.path.join(d, "cart0400"), blue, fb, eb, 1,
                  "RAPID-READOUT", "H13", 1000.0, "01/02/94")
        fr, er = one_exposure(rng, red, 1, False, level=1.00)
        write_fos(os.path.join(d, "cart0401"), red, fr, er, 1,
                  "RAPID-READOUT", "H19", 1000.0, "01/09/94")
    emit("cartoon_two_epochs_offset",
         "two gratings seven months apart whose overlap differs by 25 per cent: "
         "a step in the continuum the ICA keys on", b_two_epoch,
         dict(level_blue=0.80, level_red=1.00, overlap_ratio=1.25))

    def b_nine_tilted(d):
        rng = np.random.default_rng(8)
        for i in range(9):
            f, e = one_exposure(rng, wave_obs, 1, False)
            if i % 3 == 0:
                # a SHAPE difference: a 15 per cent tilt across the band, which
                # survives the continuum normalisation that divides out a level
                # offset.  This is what "the exposures differ by more than their
                # errors" has to mean for the co-add to see it, and it is the
                # case the 2026-09-23 revert was worried about.
                wr = wave_obs / (1.0 + Z)
                tilt = 1.0 + 0.15 * (wr - wr.mean()) / (wr.max() - wr.min())
                f = f * tilt
            write_fos(os.path.join(d, "cart03%02d" % i), wave_obs, f, e, 1,
                      "RAPID-READOUT", "H19", 1000.0, "01/01/95")
    emit("cartoon_nine_tilted",
         "9 exposures, three of them tilted 15 per cent across the band: a SHAPE "
         "disagreement that survives continuum normalisation, so the combined "
         "error must be inflated", b_nine_tilted,
         dict(n_exposures=9, n_tilted=3, tilt_fraction=0.15))

    print()
    print("%d cartoons written to %s" % (len(made), OUT))
    print("truth: blueshift %+.0f km/s, EW %.0f A, FWHM %.0f km/s, z = %.2f"
          % (TRUE_BLUESHIFT, TRUE_EW, TRUE_FWHM, Z))


if __name__ == "__main__":
    main()
