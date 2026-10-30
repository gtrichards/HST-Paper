# Prompt for the next session: paper plots and Section 2

Paste this (or point the session at this file) to start.

---

## What I want from this session

Two things, and **only** these two:

1. **Update the plotting code** so the paper's figures are made from the
   measurements we just finished, instead of the old catalogue they currently
   read.
2. **Help with Section 2 only** — specifically, the part that describes how the
   sample was assembled and how we established that we are not missing HST
   spectra, and not including objects that have none. I want the *description of
   what was done*, because only you have that provenance.

**Do not write any other part of the paper.** No introduction, no discussion, no
conclusions, no abstract. If you think a result deserves prose, say so in chat
and leave it to me.

As a reminder the git repo for the paper is in /Users/gtr/Work/git/5bb25899051ceb07efc729f4

I will decide whether to add the IUE and GNIRS-DQS data after I have seen the
updated plots. Do not start on that unless I ask.

---

## Where things stand

The object-by-object pass over all 474 rows of `CIV_measurements_v22.xlsx` is
**complete**. Every object has a verdict and a written justification.

```
Data/decisions.csv            474 rows, one per object: folder, manmask,
                              reject_reason, note. The note is the full
                              reasoning. This is the authority.
Data/CIV_measurements_v23.csv 474 rows: row_num, designation, common_name, z,
                              instrument, adopted, sdss_flag, sul_flag,
                              cat_flags, CIV_blue, CIV_EW, CIVgood_manmask

manmask  3 = final (237)   2 = accepted with a stated reservation (112)
         1 = IUE-only (22)  0 = unmeasurable (103)
adopted sample = manmask 2 or 3 = 349 objects
```

Adopted-sample statistics as they now stand (349 objects):

```
blueshift  median +197.5  mean +401.5  sd 911.4  16-84%  -239.1 to +887.0
log EW     median 1.866 (sample sd 0.341, 16-84% 1.523-2.207)
```

Blueshift sign convention: `blueshift = ((1549.48 - lambda_half)/1549.48) * c`,
so **positive is blueshifted** and negative means the line sits redward of
systemic. Values are quantised at 69.03 km/s by construction — that is
deliberate, not a bug, and must not be "fixed".

### Where the numbers live

```
pipeline_output/chosen_visit/fits/<NNN_Name_INSTR>/records/BEST_*.json
```

349 files, one per adopted object, each holding the complete provenance:
`override` (the mask and component set used), `result` (civ_blue, civ_ew,
f2500, z), `monochromatic` (log L1450 / L1700 / L2500 with the calibration),
`windows` (per-line residuals), `civ_subcontinuum` (the veto),
`redshift_check_kms`.

`finalrun/build_paper_table.py` builds `CIV_measurements_v23.csv` from the
ledger plus these records. It joins on the **index prefix**, not the folder
name — the ledger's folder names predate the single-visit policy and some still
say `_COMBINED` while the fit lives in `_COS`. That is deliberate; do not
"fix" the folder strings.

---

## The actual plotting problem

`Plotting Code/make_fig1.py` (L versus z) and `make_fig2.py` currently read the
**old** HST catalogue:

```
Data/HST_CIV_Sulentic2007_HSLA2018_finalprops_f2500AP.csv   207 rows
Data/HST_CIV_Sulentic2007_HSLA2018_1Jan2023.csv             208 rows
    columns CIV_blue_tvm, CIV_EW_tvm, CIVgood_tvm
```

The new table is 474 rows with `CIV_blue`, `CIV_EW`, `CIVgood_manmask`. So the
figures are being made from roughly 200 superseded measurements while 349
current ones sit unused. `Notebooks/L_versus_z.ipynb` already references the
v23 table and manmask; `Plotting Code/` does not.

**There is a gap you will hit immediately: `CIV_measurements_v23.csv` has no
luminosity column, and Figure 1 is luminosity versus redshift.** The
luminosities exist, but only inside the `BEST_*.json` records. Census of the 349
adopted objects:

```
log L1450 measured : 345 (99%)   median 45.43
log L1700 measured : 247 (71%)   median 45.40
log L2500 measured :  97 (28%)   median 44.96

L1450 only 102 | L1450+L1700 147 | all three 96 | none 0
```

So the first real task is probably to extend `build_paper_table.py` to carry the
monochromatic luminosities into the table, then point the plotting code at it.
Decide that with me before doing it.

Some luminosities carry caveats recorded in the ledger — a handful of objects
have a continuum normalisation that rises to the red (the "419 family", 47 of
369 by one count) and their L1450 is flagged provisional in the note. If the
luminosity axis matters to a figure, those are worth surfacing rather than
silently plotting.

---

## Section 2: what I need and what I do not

I need a description of **how the sample was established and checked**, because
that provenance only exists in the ledger and in your work. Specifically:

- How objects were selected from the v22 spine of 474.
- How we determined that every object that should have an HST spectrum has one,
  and that no object without usable HST data was carried into the sample. There
  is real content here: three objects (377, 389, 403) were found by an
  end-of-pass completeness check to have no decisions row at all; all three turn
  out to have C IV falling in a G230L detector gap, with zero usable pixels in
  1500-1600 A and zero in the 1445-1465 A anchor, verified from the rebinned
  FITS rather than from a status column.
- The single-visit policy, and the splice exception for objects where one
  instrument carries the blue end and another the red.
- What manmask 0, 1, 2, 3 mean and how many objects are in each.
- The co-addition and rebinning provenance in outline (the v23 regenerated tree
  against the students' master).

Write that as a factual methods description. Nothing else.

### The reference for the C IV distance

```
DOI: 10.17918/CIVdistance
```

GTR supplied this on 2026-10-30 as the citable record for the C IV distance —
the quantity this whole pass measures, and the one `main.tex` already treats as
a defined concept (line 57: "Talk about \civ\ distance as a proxy for \lledd").
It is **not** yet in `refs.bib` (120 entries, none matching), and nothing in
either repository referenced it before.

Cite it wherever Section 2 defines the measured quantity. The blueshift
definition currently sits in `main.tex` around lines 329-332, alongside an
unresolved `[FIX: We use linear fit, should be using power-law fit?]` note that
is GTR's to settle, not yours.

---

## Standing constraints — these carry over and are not negotiable

- **Never push to `origin`.** That is the student's repository
  (`pratsosa/HST-Paper`). Push only to `fork` (`gtrichards`). Current branch:
  `gtr/manual-fix-gui-and-reproducibility`.
- **Dropbox access is per approved folder only.** Approved:
  `~/Dropbox/HST/RebinnedSpec_HSTSDSS_test15`, `~/Dropbox/HST/Summer2026`,
  `~/Dropbox/HST/Pratsos`, and three `Dropbox/Trevor/July2024/HST/` subfolders.
  Listing or searching outside an approved folder is itself the violation. If
  something is not in an approved folder, describe what is missing and ask.
- **Do not deviate from Rankine's prescription.** Bugs in this project's own
  code get fixed; the fitting prescription does not change. An unphysical
  reconstruction is reported, not engineered away.
- The 69 km/s quantisation is deliberate.
- **The manmask carries the doubt; the record states the measured value.** Never
  withhold a number or call a measured quantity unmeasurable in the ledger.
- There is no "UV frame" — only the correct systemic redshift. A redshift may be
  changed only when the catalogue value is genuinely ambiguous *and* the frame
  can actually be measured.
- Mg II is **fitted** when it falls in the HST spectrum and is only a **check**
  when it appears solely in SDSS.
- Keep presentations brief.

---

## Two results from the last session that bear on the plots

**The Mg II / SDSS question is settled and the answer is "it changes nothing".**
Letting Mg II choose the component set would leave the sample median blueshift
identical, move the mean by 14 km/s, and narrow the scatter by 2%. Only 56 of
349 objects would change at all, median change one quantum.

**Refitting against HST+SDSS jointly makes things worse, not better.** The C IV
fit degrades in 147 of 151 objects (97%), median residual rms 0.25 -> 0.44 on
identical pixels. The cause is weighting, not the cross-normalisation: SDSS
contributes ~2300 px at S/N ~40 against ~265 px at S/N ~17 under C IV, and the
damage scales with how much SDSS is added (rms ratio 1.04 at <300 px added,
1.99 at >2000) rather than with bridge quality. What *does* improve is the
continuum: scale inside the healthy 0.88-1.00 band rises from 67% to 75%.

Neither of these changes the adopted measurements. Both are recorded in
`finalrun/sweeps/` with their outputs in `pipeline_output/final/`.

---

## Open items, none blocking

- **The CENWAVE grouping defect** in the exposure level test: `by_ep` is keyed on
  `(epoch, grating)` rather than CENWAVE. It has cost three objects (365, 382,
  448), each recovered by hand.
- **`prepare_objects.py` prints no console line for mid-loop skips**
  (`no_chosen_visit_fit`, `no_records`, `no_redshift`). That is what hid index
  448. A one-line fix was offered and never answered.
- **Down-weighting SDSS** so HST keeps control of C IV is the obvious follow-up
  to the joint-fit experiment and has not been run.
- `manmask_status_v23_2026-09-21_superseded.csv` is a dated snapshot, not a
  current status file; its README explains why. `plot_inspect.py` and
  `plot_nocivid.py` read the old filename and will fail until repointed.
- A recovery archive of everything irreplaceable is at
  `~/CIV_pass_recovery_2026-10-29.zip` (169 MB, verified by restoring it and
  replaying two fits).

---

## How I like to work

Show me the figure or the number, briefly, and let me decide. Do not write
prose for the paper beyond Section 2's methods description. If a plotting
choice has two defensible options, show me both rather than picking silently.
If you find something wrong with my request, say so in a sentence and then do
the work anyway under a stated assumption.
