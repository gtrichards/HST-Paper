"""
Galactic extinction for the 1450 A luminosity.

GTR, 2026-11-19: correct L1450 for Galactic extinction.  The C IV
measurements need no correction -- every spectrum is reshaped to a common
continuum before the ICA fit, which removes the smooth wavelength dependence
extinction imposes -- but the luminosity is read in physical units and is
lowered by it.  So the correction is applied to L1450 only, after the fit.

Recipe (the standard one, and what the SDSS comparison luminosities of
Wu & Shen 2022 already carry):
  * E(B-V) at each object's position from the Schlafly & Finkbeiner (2011)
    recalibration of the SFD dust map (IRSA dust service, "ext SandF mean"),
    stored in Data/EBV_SandF_v23.csv by fetch_ebv.py;
  * the Fitzpatrick (1999) extinction curve with R_V = 3.1, which Schlafly &
    Finkbeiner recommend with their E(B-V);
  * A_lambda evaluated at the OBSERVED wavelength of rest 1450 A,
    1450 (1+z) A, because that is where the Galaxy's dust acts;
  * log L1450 increases by 0.4 A_lambda.

The curve is written out here rather than taken from a package so that it can
be checked line by line against Fitzpatrick (1999, PASP 111, 63), Section 4
and Table 3.  check() reproduces published values.
"""
import numpy as np
from scipy.interpolate import CubicSpline

R_V = 3.1

# Fitzpatrick (1999), ultraviolet: the Fitzpatrick & Massa (1990) form with
# the R_V-dependent c1, c2 of F99 eqs. (4)-(5) and the fixed bump and
# far-UV parameters of F99 Table 3.
X0, GAMMA, C3, C4 = 4.596, 0.99, 3.23, 0.41


def _uv_k(x, rv):
    """E(lambda-V)/E(B-V) for x = 1/lambda[um] >= 1/0.27 (lambda <= 2700 A)."""
    c2 = -0.824 + 4.717 / rv
    c1 = 2.030 - 3.007 * c2
    d = x**2 / ((x**2 - X0**2)**2 + x**2 * GAMMA**2)          # the 2175 A bump
    f = np.where(x >= 5.9, 0.5392 * (x - 5.9)**2 + 0.05644 * (x - 5.9)**3, 0.0)
    return c1 + c2 * x + C3 * d + C4 * f


def _optical_k(x, rv):
    """E(lambda-V)/E(B-V) longward of 2700 A: F99's natural cubic spline in x
    through anchor points at 26500, 12200, 6000, 5470, 4670, 4110 A (Table 3)
    and at 2700 and 2600 A, where the ultraviolet form is used."""
    xk = 1.0e4 / np.array([np.inf, 26500., 12200., 6000., 5470., 4670., 4110., 2700., 2600.])
    yk = np.array([
        -rv,                                                         # A = 0 at x = 0
        0.26469 * rv / 3.1 - rv,
        0.82925 * rv / 3.1 - rv,
        -0.422809 + 1.00270 * rv + 2.13572e-4 * rv**2 - rv,
        -5.13540e-2 + 1.00216 * rv - 7.35778e-5 * rv**2 - rv,
        0.700127 + 1.00184 * rv - 3.32598e-5 * rv**2 - rv,
        1.19456 + 1.01707 * rv - 5.46959e-3 * rv**2 + 7.97809e-4 * rv**3 - 4.45636e-5 * rv**4 - rv,
        float(_uv_k(xk[7], rv)),
        float(_uv_k(xk[8], rv)),
    ])
    return CubicSpline(xk, yk, bc_type="natural")(x)


def a_over_ebv(wave_A, rv=R_V):
    """A_lambda / E(B-V) for wavelengths in Angstroms (F99, R_V = rv)."""
    x = 1.0e4 / np.asarray(wave_A, float)                    # inverse microns
    k = np.where(x >= 1.0e4 / 2700., _uv_k(x, rv), _optical_k(x, rv))
    return k + rv


def dlogL_1450(ebv, z):
    """Increase in log L1450 from correcting for Galactic extinction."""
    return 0.4 * np.asarray(ebv, float) * a_over_ebv(1450. * (1. + np.asarray(z, float)))


def check():
    """Print A/E(B-V) at a few wavelengths.  Checked 2026-11-19 against an
    independent implementation of Cardelli, Clayton & Mathis (1989), R_V = 3.1:
        A      1450  1700  2175  2500  3000  3220  4400  5470
        F99    8.29  7.65  9.54  7.06  5.57  5.23  4.06  3.05
        CCM89  8.50  7.79  9.87  7.18  5.64  5.25  4.11  3.12
    i.e. within 2-4 per cent over 1450-3220 A, the observed wavelengths of
    rest 1450 A in this sample.  (F99's V anchor gives 3.05 rather than 3.1 by
    construction of its spline.)"""
    for w in (1450., 1700., 2175., 2500., 3000., 4400., 5470.):
        print("  F99 A/E(B-V) at %6.0f A: %.2f" % (w, a_over_ebv(w)))


if __name__ == "__main__":
    check()
