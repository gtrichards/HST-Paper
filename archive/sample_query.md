# The sample-defining query

**STATUS: NOT YET WRITTEN OR TESTED.** This file is a placeholder so that the
archive's structure is honest about what is missing. It must not be read as a
verified recipe.

GTR, 2026-11-06: *"we need a final query that produces exactly the objects in
Table 1 – after we drop the rows that are manmask=0. That is the query should
recognize when there is no CIV coverage. Noting that we might want to keep the
manmask=0 rows where there is indeed CIV coverage, but we have deemed it
unusable. I will get my students to double check that the query produces exactly
the objects in the Table."*

And on 2026-11-08, on this archive: *"It is worth noting that we have not yet
tested (or redefined, if necessary) the query that we have confirmed can be used
to regenerate the exact object list that we will publish in Table 1."*

## What the query has to reproduce

Measured from the rebinned products by `code/civ_coverage_audit.py`, which is
the authority here because it tests the data rather than a catalogue column:

| | objects |
|---|---|
| manmask 2 or 3 — adopted measurements | 366 |
| manmask 0 **with** C IV coverage — examined, judged unmeasurable | 68 |
| **the published sample** | **434** |
| manmask 0 with **no** C IV coverage — excluded, nothing was there to judge | 40 |
| total catalogue | 474 |

"Covered" means at least 10 pixels in 1500–1600 Å rest that are unmasked, have
finite flux and a **positive error** — the fitter's own criterion. Each clause
was added because of a specific failure; see the header of
`code/civ_coverage_audit.py`.

## Two things known to be wrong with the obvious approaches

**Do not use `work_queue.civ_frac_best`.** It disagrees with the measured
coverage for 32 objects. Thirty are the IUE additions, which it cannot see
because it was computed from HST spectra alone; two are HST objects it simply
has wrong (orders 164 and 220, which it reads as 0 while their adopted products
carry 272 and 240 usable C IV pixels and both are measured).

**Do not use MAST's `em_min_aa`/`em_max_aa` alone.** That was the v22 selection
criterion and those columns are the *union* across every observation of a
target, so an object passes whenever its bluest and reddest exposures bracket
C IV even when nothing covers it — the union spans the gap between gratings.
All seven of the blanket-authorisation manmask-0 rows pass that test, and three
of them turned out to have IUE coverage instead. The test is necessary but not
sufficient, and it errs in the safe direction.

## Date restriction

GTR: *"we should restrict the query by date to avoid including data taken since
the query was redone in ~April"*, and *"28 April is what sticks in my head. But
we could also use whatever date works empirically."* **The cutoff has not been
pinned empirically yet.** The check is to vary the date and find the value at
which the returned object list stops changing and matches the 474.

## What remains to be done

1. Write the query against the archive, with the date cutoff.
2. Run it and diff the result against `measurements.csv` — it must return
   exactly those 434 index values.
3. Record here any object the query cannot reach, with the documented check that
   admits or excludes it. GTR has accepted that a query plus documented checks
   is acceptable so long as it is repeatable.
4. Have the students reproduce step 2 independently.

Until all four are done this file stays marked NOT TESTED.
