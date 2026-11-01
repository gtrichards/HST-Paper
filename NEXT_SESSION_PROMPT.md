# Prompt for the paper session: figures and Section 2

Paste this, or point the session at this file, to start.

Rewritten 2026-11-01. The fitting session is staying open for fitting; this is
the paper-side work moved across.

---

## What I want from this session

Two things, and **only** these two:

1. **The paper's figures** — finish Figure 1's design, then get the figures into
   the document. They are generated but the document does not use them.
2. **Section 2 only** — the part describing how the sample was assembled and how
   we established we are not missing HST spectra, and not including objects that
   have none. I want the description of what was done, because only the work
   itself has that provenance.

**Do not write any other part of the paper.** No introduction, no discussion, no
conclusions, no abstract. If a result seems to deserve prose, say so and leave
it to me.

I will decide about adding IUE and GNIRS-DQS after seeing the updated plots.
Don't start on that unless I ask.

**Eddington ratios are out of scope** for this paper — probably the next one.

Repositories:
```
paper (Overleaf)  /Users/gtr/Work/git/5bb25899051ceb07efc729f4   branch main
code + data       /Users/gtr/Work/git/HST-Paper                  branch gtr/manual-fix-gui-and-reproducibility
fitting tree      /Users/gtr/Work/projects/hstica/finalrun       (not a git repo; tracked copy lives in HST-Paper/finalrun)
```

---

## State of the measurements

The object-by-object pass over all 474 rows of `CIV_measurements_v22.xlsx` is
**complete**. Every object has a verdict and a written justification.

```
Data/decisions.csv            474 rows: folder, manmask, reject_reason, note.
                              The note is the full reasoning. This is the authority.
Data/CIV_measurements_v23.csv 474 rows: row_num, designation, common_name, z,
                              instrument, adopted, sdss_flag, sul_flag, cat_flags,
                              CIV_blue, CIV_EW, CIVgood_manmask

manmask  3 = final (237)   2 = accepted with a stated reservation (112)
         1 = IUE-only (22)  0 = unmeasurable (103)
adopted sample = manmask 2 or 3 = 349 objects

blueshift  median +197.5  mean +401.5  sd 911.4  16-84%  -239.1 to +887.0
log EW     median 1.866   (sd 0.341, 16-84% 1.523-2.207)
```

Sign convention: `blueshift = ((1549.48 - lambda_half)/1549.48) * c`, so positive
is blueshifted. Quantised at 69.03 km/s by construction — deliberate, not a bug.

Per-object provenance: `pipeline_output/chosen_visit/fits/<NNN_Name_INSTR>/records/BEST_*.json`,
349 files, each with the override (mask + component set), result (civ_blue,
civ_ew, f2500, z), monochromatic (log L1450/L1700/L2500), per-line residuals,
the sub-continuum veto, and redshift cross-checks.

---

## *** READ THIS BEFORE TOUCHING ANY FIGURE ***

```python
import richardsplot as rplot            # WRONG -- silently does nothing
from richardsplot import richardsplot as rplot   # correct
```

`richardsplot/__init__.py` is EMPTY, so the plain import resolves to it and the
house style never applies. The figure then comes out in matplotlib defaults:
sans-serif, ticks outward, no minor ticks, and `text.usetex` off — which is why
`\,` prints literally and `\AA` renders in math italic. GTR's own `testplot1.py`
has the same import, so this is long-standing and will bite again. All three
`paper_figures_v23` scripts and `Plotting Code/make_fig1.py` are fixed; anything
new must use the two-name form.

The style GTR wants (SuperMongo-like) is exactly what `richardsplot.py` sets:
serif via usetex, ticks `in`, major and minor ticks visible, `axes.linewidth`
1.5, tick sizes 10/5. Don't reinvent it; just import it correctly.

Several scripts need `PYTHONPATH=~/Work/git` (for `richardsplot` and
`CIV_Distance`).

---

## Where the figure code lives

`HST-Paper/paper_figures_v23/` is the current generation and is tracked. Do NOT
start from `Plotting Code/`, which is the older one.

```
hst_sample.py        load_adopted(); M_BH categories from the old catalogue
extract_f1450.py     -> Data/HST_F1450_v23.csv  (F1450 + F1450_est flag)
fix_old_f2500.py     -> Data/HST_old_catalogue_fluxes_corrected.csv
make_fig1_L1450.py   -> Figures/Fig1_L1450_versus_z.pdf
make_fig4.py         -> Figures/Fig4_CIV_logEW_versus_blue.pdf
make_fig5a_6a.py     -> Figures/Fig5a_CIV_blue_versus_L1450.pdf
                        Figures/Fig6a_CIV_dist_versus_L1450.pdf
make_table1.py       -> Tables/table1_master_sample_v23.tex + .csv
make_table2.py       -> superseded, see Tables below; still works
trevor_contour.py    plot_contour, copied verbatim from Trevor's CIVplot.ipynb
```

All four figures are current as of 2026-11-01.

---

## Figure 1 — mechanically finished, two design questions left for GTR

Done: all **349** objects plotted (was 345); the four with no measurable L1450
estimated from each object's own continuum slope and drawn with a ring so they
are visibly not measurements; style fixed; y-axis relabelled
`log L_nu(1450 A)` because the axis is erg/s/Hz and the old label could be read
as nu L_nu.

**Open, and GTR's to decide:**

1. **The colour coding** — HST+RM / HST Scaling-Relation / HST no-M_BH. GTR:
   "that is what I had intended, but I'm not sure that still makes sense."
   Note the Scaling-Relation group already uses a white marker face, so "open
   symbol" is taken as an encoding.
2. **Illustrating the RM bias** — the goal is to show the classical RM sample is
   biased against strong winds, mostly because those objects are lower
   luminosity, being at lower redshift. The data hint at it (RM points sit
   almost entirely below z ~ 0.4 and away from the luminous end) but it has NOT
   been quantified. GTR was offered numbers first and the session moved before
   answering.

**Settled, don't relitigate:**

- *"How can the most luminous HST objects be more luminous than the most
  luminous SDSS objects?"* They aren't. SDSS reaches lambdaL = 47.73 against
  HST's 47.29, and 38 SDSS objects exceed the brightest HST object — but that is
  0.03% of 125,729, outside the outermost (95%) contour, so they render as sparse
  dots and read as absent. The two samples also do not overlap in redshift AT
  ALL (HST 0.001-1.223, SDSS 1.624-3.500), because C IV only enters the SDSS
  window above z ~ 1.6. If the figure should stop implying SDSS is fainter, the
  fix is in the rendering (a 99% contour, or darker points above the top
  contour), not the data.
- **SDSS and GNIRS-DQS stay at L1350** and are labelled as such. Converting to
  L1450 would move them 0.015 dex for a typical continuum — invisible.
- GNIRS-DQS stays in as a reference; GTR wants to redo that analysis later.

---

## The luminosity audit — three errors, all now fixed

Worth knowing because old numbers are still in circulation and all three made
HST luminosities too LOW.

```
1. old F1450_spec   missing x f_lambda   -0.470 dex   flat in z
2. old F2500_AP     missing x f_lambda   -0.236 dex   flat in z
3. iterate_fit.py + ica/manual_fix_gui.py
                    missing x (1+z)      -0.116 dex median, up to 0.347, GROWS with z
```

**(1) and (2):** the old catalogue's flux columns are the co-adder's *Continuum
Normalisation* at that wavelength, with the spectrum never multiplied in.
Matched object by object: our `continuum_scale` alone equals old `F1450_spec` to
+0.000 dex (n=190) and old `F2500_AP` to -0.009 dex (n=73). The missing factor is
the model value there — 2.97 at 1450 A, 1.76 at 2500 A, which is just the quasar
continuum shape ((1450/3000)^-1.5 = 2.98). Corrected per object in
`Data/HST_old_catalogue_fluxes_corrected.csv` (195 of 207 rows; the other 12 have
no v23 fit and are dropped). `Plotting Code/make_fig1.py` and
`Notebooks/L_versus_z.ipynb` now use it.

**(3):** `rebinning/coadd.py:526` writes `Table([final_wave/(1+z), final_flux, ...])`
— only the WAVELENGTH is divided, so the calibrated flux is observed-frame flux
density on a rest-frame wavelength axis, and
`nu_rest L_nu = 4 pi D_L^2 lambda_rest (1+z) f_lambda(obs)`. Fixed at source in
both copies of the formula, and applied to 3035 stored records (5884 logL values)
by `finalrun/sweeps/fix_luminosity_1pz.py`, which is idempotent. The ledger notes
quote luminosities as literal text, so 318 rows carry an appended correction
paragraph giving each object's own corrected values.

Validation: Figure 1's independent calculation and the ledger now agree to
**0.0001 dex** (was 189 objects off by >0.1). Separately, HST observed flux
matches SDSS where the two overlap in rest wavelength — 26 objects, median ratio
1.064 (+0.027 dex), the scatter being genuine AGN variability.

**Still carrying error, deliberately not fixed** (Eddington ratios are out of
scope): the old catalogue's `logLbol_2500` and `logEddRatio_2500`, both derived
from F2500, are low by 0.236 dex — Eddington ratios there are a factor 1.7 low.

---

## Tables — done and live in Overleaf

Table 2 was merged into Table 1 and the result is in the document.

```
Tables/table1_master_sample_v23.tex   474 rows, 12 columns, landscape
```

- Wrapped in `\begin{longrotatetable}`, NOT `\rotate`: AASTeX does not apply
  `\rotate` to a `\startlongtable`, which is why it rendered portrait at first.
- Table 2's measurement columns (Blueshift, EW, D_CIV) moved in;
  `table2_civ_v23.tex` is no longer `\input`. `make_table2.py` still works, so
  the split version is a revert of `make_table1.py` away.
- The separate `Spec.` column was folded into `Inst.` by **bolding** the
  instrument the measurement came from. Two objects cannot be bolded and show the
  spectrum in parentheses: **#345 LEDA 50824** `COS/FOS (SPLICE)`, and
  **#452 ICRF J213745.1-143255** `COS/FOS (STIS)` — the second is a genuine
  defect in the instrument listing, since the adopted fit is `116_..._STIS` and
  STIS is not listed at all. Worth chasing for Section 2.
- Redshifts at **five decimals**: that column is the redshift used in the fit, an
  input to the velocities, and four decimals cost up to 14.6 km/s at z = 0.03
  (~29 km/s peak to peak, 0.42 of a quantum).
- `D_CIV = 0.00` for 16 objects is correct, not an error: the best-fit curve spans
  EW 13-316 A and these have EW > 316 A, so they project onto its endpoint. GTR
  is content to report 0 and may extend the curve in future. A one-line footnote
  saying so was offered and not yet added.

---

## The real remaining gap: the figures are not in the document

`main.tex` still `\plotone`s the older ICA-era figures. The paper uses AASTeX
`\plotone`, not `\includegraphics` — there is exactly one `\includegraphics` and
it is commented out. The four swaps:

```
main.tex:258  figures/L_versus_z_ICA_new.pdf                  -> Fig1_L1450_versus_z.pdf
main.tex:343  figures/CIV_logEW_versus_blue_ICA.pdf           -> Fig4_CIV_logEW_versus_blue.pdf
main.tex:363  figures/L_and_Lwise_versus_blue_ICA_labeled.pdf -> Fig5a_CIV_blue_versus_L1450.pdf
main.tex:374  figures/L_and_Lwise_versus_c4dist_ICA_labeled.pdf -> Fig6a_CIV_dist_versus_L1450.pdf
```

**Check each with GTR before swapping**: at least two of the old figures are
two-panel (`L_and_Lwise_...` shows L and L_WISE together) where the v23 versions
are single-panel, so a swap may falsify the caption. The other eight active
`\plotone` calls are figures this pass does not touch. The PDFs also have to be
copied into the Overleaf repo's `figures/` directory.

---

## Section 2 — what is wanted

A factual methods description of how the sample was established and checked:

- selection from the v22 spine of 474;
- how we established that every object that should have an HST spectrum has one,
  and that nothing without usable HST data was carried in. There is real content:
  three objects (377, 389, 403) were found by an end-of-pass completeness check
  to have no decisions row at all, and all three turn out to have C IV in a G230L
  detector gap — zero usable pixels in 1500-1600 A and zero in the 1445-1465 A
  anchor, verified from the rebinned FITS rather than a status column;
- the single-visit policy and the splice exception;
- what manmask 0/1/2/3 mean and the counts;
- co-addition and rebinning provenance in outline (the v23 regenerated tree
  against the students' master).

The C IV distance has a citable record: **DOI 10.17918/CIVdistance**, already in
`refs.bib` as `@MISC{CIVDistance}` (McCaffrey & Richards 2022) with the DOI
attached. `main.tex` already treats "C IV distance" as a defined concept
(line 57). Note the unresolved `[FIX: We use linear fit, should be using
power-law fit?]` next to the blueshift definition around main.tex:329 — that is
GTR's to settle, not yours.

---

## Standing constraints — not negotiable

- **Never push HST-Paper to `origin`** (the student's repository, pratsosa).
  Only to `fork` (gtrichards). The **Overleaf** repo is different: its `origin`
  IS Overleaf and pushing there is correct. **Fetch first** — collaborators edit
  in Overleaf and the remote moves; it moved three times during one session.
  Rebase onto their work, never over it.
- **Dropbox access is per approved folder only**: `~/Dropbox/HST/RebinnedSpec_HSTSDSS_test15`,
  `~/Dropbox/HST/Summer2026`, `~/Dropbox/HST/Pratsos`, and three
  `Dropbox/Trevor/July2024/HST/` subfolders. Listing or searching outside an
  approved folder is itself the violation. If something is missing, describe it
  and ask.
- **Do not deviate from Rankine's prescription.** Bugs in this project's own code
  get fixed; the fitting prescription does not change.
- The 69 km/s quantisation is deliberate.
- The manmask carries the doubt; the record states the measured value.
- There is no "UV frame", only the correct systemic redshift.
- Mg II is fitted when it falls in the HST spectrum, a check when only in SDSS.
- **Do not deviate far from the existing plot style** — GTR likes SuperMongo
  defaults (Hershey font, ticks in, major and minor ticks). See the richardsplot
  note above.
- Keep presentations brief.

---

## Settled, so nobody repeats them

- **Mg II / SDSS would change nothing.** Letting Mg II choose the component set
  leaves the sample median blueshift identical, moves the mean 14 km/s, narrows
  the scatter 2%. 56 of 349 objects change at all, median one quantum.
- **Joint HST+SDSS fitting makes things worse.** C IV degrades in 147 of 151
  objects, median residual rms 0.25 -> 0.44. The cause is weighting, not the
  cross-normalisation: SDSS contributes ~2300 px at S/N ~40 against ~265 px at
  S/N ~17 under C IV, and the damage scales with how much SDSS is added, not with
  bridge quality. The continuum does improve (scale in the healthy band 67% ->
  75%). Down-weighting SDSS is the obvious follow-up and has NOT been run.
- Both are recorded in `finalrun/sweeps/` with outputs in `pipeline_output/final/`.

---

## Open items, none blocking

- The CENWAVE grouping defect in the exposure level test: `by_ep` keyed on
  `(epoch, grating)` rather than CENWAVE; it has cost three objects (365, 382,
  448), each recovered by hand.
- `prepare_objects.py` prints no console line for mid-loop skips; that is what
  hid index 448. A one-line fix was offered and never answered.
- `plot_inspect.py` and `plot_nocivid.py` read `manmask_status_v23.csv`, which was
  renamed to `manmask_status_v23_2026-09-21_superseded.csv` (a stale snapshot; its
  README explains). They will fail until repointed — deliberately, so they fail
  loudly rather than plotting five-week-old state.
- A verified recovery archive of everything irreplaceable:
  `~/CIV_pass_recovery_2026-10-29.zip` (169 MB; restored and two fits replayed to
  confirm).

---

## How I like to work

Show me the figure or the number, briefly, and let me decide. Don't write paper
prose beyond Section 2's methods description. If a choice has two defensible
options, show me both rather than picking silently. If something is wrong with my
request, say so in a sentence and then do the work anyway under a stated
assumption.
