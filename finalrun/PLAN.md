# Plan of action, from 2026-09-22

Written so it survives context compaction. Update it as things move.

## Where the sample stands

The data set was regenerated from the archive on 2026-09-20/21: 455 of 474
objects returned public HST products, 647 spectra rebinned (the master had
576), and the coverage scan, identity map and work queue rebuilt on top.
`finalrun/rebin_path.py` makes the regenerated tree the default; `HSTICA_REBIN`
overrides it, and `HSTICA_ITERDIR` switches where fit iterations are written.

Four bugs were found and fixed between the archive and the fit: object names
containing square brackets (glob character classes, 18 objects invisible),
FOS exposures with fewer than three readout groups, zero-flux acquisition
exposures aborting an object, and -- the large one -- only the first detector
segment of each COS/STIS file ever being read, which discarded about half of
every FUV spectrum and two thirds of every NUV one.

Consequences: objects with no usable C IV fell from 71 to 44; fittable objects
rose from 346 to 366; and 12 of the 26 objects excluded at manmask 0 turned out
to have C IV coverage all along.

## Ordering (unchanged, set by GTR)

By category first, then by redshift within each category:

1. **Group 1, no SDSS spectrum** -- 159 objects. 120 have a decision row,
   **39 remain**, indices 111-159, all fittable.
2. **Group 2, HST + SDSS** -- 315 objects, 54 with a decision row.
3. **IUE group** -- the 19 objects with no HST data at all, plus 3 carried at
   manmask 1 that do have HST data which never reaches C IV (indices 8, 41, 96).

The "index" used throughout is position in `pipeline_output/work_queue.csv`
(`order`), not the v22 spreadsheet row; both columns are in that file.

## The work, in order

### 1. The reopened exclusions (~16 objects) -- needs Fable
Twelve objects that gained C IV: 13, 15, 40, 50, 68, 69, 78, 80, 84, 85, 87,
111. Three whose coverage went from marginal to full: 22, 31, 97. Plus 20 and
27. These are new measurements, not re-reviews.

### 2. Re-review indices 1-110 (73 measured objects) -- needs Fable
Do **not** hand-refit from the stale masks. The comparison figures in
`pipeline_output/final/fit_compare/` show what the old masks give on the new
data, which is a floor, not what the fit can reach -- which is why GTR judged
nearly all of 1-57 improvable. An automated first pass is being run into
`pipeline_output/fit_iterations_v23/` instead; review those in blocks of ten.

### 3. Finish group 1 -- 39 objects, indices 111-159 -- needs Fable
Index 111 is 3C 273, which now has 140 usable C IV pixels. The ten provisional
verdicts for 111-120 were fitted on the superseded tree, are marked PROVISIONAL
in their notes, and get replaced.

### 4. Group 2, the SDSS objects -- 315 objects
**Blocked until SDSS is plumbed in.** `rebinning/coadd.py` already accepts a
plate-MJD-fiber name and will fit, morph and splice an SDSS continuum, but the
catalogue bridge passes `fn_sdss=None` for every object, so the whole sample is
currently HST-only. This gates two thirds of the sample and is being worked on
in parallel (Opus).

## Open items, none blocking

- Pre-existing zero-error regions: 125 of 626 spectra carry pixels with flux,
  zero error and no mask flag, up to 75 per cent of a spectrum. The master has
  it at the same rate (110 of 559), so it is not ours. It is why 89 spectra look
  as though C IV is entirely masked when the flux is there.
- Four objects marked manmask 0 must be revisited in group 2 because GTR's v22
  note asks whether SDSS helps: indices 171, 208, 210, 228.
- The controlled vocabulary needs a term for Ton 951 (76): the observation that
  would cover C IV exists but has no extracted product, and extraction is ruled
  out. `insufficient_coverage` understates it.
- Index 037 (Mrk 279) is the one object whose C IV signal-to-noise genuinely
  fell, 16.1 to 13.1.
- `ica/manual_fix_overrides.json` holds an uncommitted hand fit for PG 1402+261
  from the stale 111-120 block.

## Standing checks

- `finalrun/audit_tree.py` tests every rebinned spectrum for the ways this
  pipeline has silently produced wrong data. Run it after any change to the
  readers or the co-adder.
- Before any replay, snapshot `pipeline_output/fit_iterations` and verify the
  record checksum afterwards: replaying writes records in place.

## Models

Fable 5.1 for fitting and per-object judgement; Opus 5 for the mechanical work,
the notebook entries and the block-end commits. Prompt for the switch at block
boundaries in both directions.

## Open, found 2026-09-22 during the reopened-exclusion work

- **Ton S 210 STIS loses its C IV pixels inside the co-addition step.** They
  survive the reader (about 1000 raw-good per order), the rebin (47 and 23 per
  order) and the continuum (positive throughout), and are dead in the output.
  Not diagnosed. Index 97 has COS covering C IV so it is not blocked.
  `finalrun/trace_civ.py OBJ INST Z` reproduces the first three stages.
- Four G140L objects (2MASS J11190530+5925140, FBQS J2226-0901, Q1130+6026,
  [VV98] J102847.0+391758) lose C IV at the reader's edge trim: the line sits
  on the far red end of G140L, which GTR's notes already call noisy. Treated
  as marginal rather than as a defect.
- The zero-continuum fallback normalises 10,804 exposure-segments by a constant
  in the current tree, most of them STIS echelle orders that sit wholly inside
  an emission-line exclusion window. Levels match their neighbours to first
  order; the within-order slope is what is lost. Worth a look if echelle
  co-adds look stepped.
