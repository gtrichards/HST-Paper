# Publication archive

The materials that go with the paper: every C IV blueshift and equivalent width
we publish, the spectrum each was measured from, the fit that produced it, and
the hand-made decisions that determined which photons went in.

Built by `build_archive.py`, checked by `verify_archive.py`. It is a build
product, not a hand-maintained copy — see *How this is maintained* below.

## The sample

| | objects |
|---|---|
| manmask 2 or 3 — adopted measurements | 366 |
| manmask 0 with C IV coverage — examined, judged unmeasurable | 68 |
| **total in this archive** | **434** |

A further 40 catalogue objects have no C IV coverage at all and are excluded;
there is nothing to deposit for them. `sample_query.md` states the cut and **is
explicit that the query reproducing it has not yet been written or tested.**

manmask: **3** final, **2** accepted with a stated reservation, **0**
unmeasurable. The reservation behind every 2 is written out in `decisions.csv`;
it is never a bare flag.

## Contents

| | |
|---|---|
| `measurements.csv` | one row per object: index, name, designation, z, manmask, source, blueshift, EW, log L1450, log L1700, usable C IV pixels, and the spectrum and fit record it came from |
| `decisions.csv` | the ledger — the verdict for every one of the 474 catalogue objects and the full reasoning for it |
| `sample_query.md` | the sample definition, and what is still missing from it |
| `spectra/` | 434 rebinned spectra, one per object |
| `fits/` | 366 adopted fit records, index-prefixed. Each carries the override that produced it |
| `figures/` | 366 adopted fit figures, index-prefixed |
| `provenance/` | the hand-made inputs: work queue, both override stores, redshift and name overrides, chosen-visit and chosen-run tables, BAL status |
| `code/` | the measurement path |
| `MANIFEST.csv` | every file with its size and SHA-256 |

**`fits/` is not a convenience copy.** Forty-nine of the 366 adopted fits have
masks that exist *only* in their record — 45 HST and 4 IUE — because they were
made with `iterate_fit --mask` on the command line rather than through the GUI,
so nothing wrote them to an override store. Without these records those 45
measurements cannot be reproduced.

## Units and conventions

- **Blueshift**, km/s: `((1549.48 − λ_half) / 1549.48) × c`, where λ_half is the
  half-flux point of the clamped cumulative line flux over 1500–1600 Å rest.
  **Positive is blueshifted.**
- The wavelength lattice is **69.03 km/s** per pixel, so blueshifts are
  quantised at that interval. This is deliberate and is not a rounding error.
- **Continuum**: a straight line through the reconstruction at 1445–1465 and
  1700–1705 Å. A C IV centroid is set by the continuum's **tilt** and is
  insensitive to its **level**; the equivalent width is the other way round.
  `decisions.csv` records which objects have an extrapolated red anchor.
- **Equivalent width**, Å, rest frame, emission positive.
- **log L1450, log L1700**: log₁₀(νLν) in erg/s. Blank where the window is not
  covered.
- **Cosmology**: flat ΛCDM, H₀ = 70 km/s/Mpc, Ω_m = 0.3.
- Fitting follows the MFICA prescription of Rankine et al. (2020). Where this
  project found bugs in its own code it fixed them; the prescription itself was
  not altered.

## What is deliberately not here

- **Staged exposures**, 3.0 GB, re-downloadable from MAST by the retrieval
  scripts in the repository.
- **Iteration figures**, 755 MB — every trial fit, not just the adopted one.
  Regenerable from `fits/` with `iterate_fit --replot`.
- **Sixty-nine of the project's 81 scripts.** `code/` holds the measurement
  path. The rest are audits, experiments and one-off figures and live in the
  repository's `finalrun/` with their history.

None of it is needed to reproduce a published number, and all of it is in the
repository. It is listed here so that nothing looks hidden.

## How this is maintained

`spectra/` and `figures/` are **not** versioned; they are regenerable and
`MANIFEST.csv` pins them byte-for-byte. Everything small and hand-made is
versioned.

Rebuild and check:

    python build_archive.py
    python verify_archive.py

`verify_archive.py` exits non-zero if the archive does not match its manifest,
so it can gate a deposit. To see what moved between versions, keep the old
manifest and run `verify_archive.py --against OLD_MANIFEST.csv`.

**Why a build script rather than a copied folder.** This project already has two
copies of its production tree — the working tree and the one in this repository
— and they drifted: a commit on 7 November 2026 swept up changes that had been
sitting uncommitted since 3 and 6 November, and that was only caught because the
two were diffed first. A third hand-maintained copy would drift the same way. So
the archive is generated from the authoritative sources every time, and the
manifest is what makes a stale one detectable.

`build_archive.py` reads the production tree named by `HSTICA_ROOT`, defaulting
to the path on the author's machine.
