# `finalrun/` — the definitive-pass driver scripts

These are the scripts that drive the object-by-object pass over
`CIV_measurements_v22.xlsx`. They live and run in
`~/Work/projects/hstica/finalrun/`, which is **not** a git repository; this
copy exists so the working code is recoverable between laptop backups.

**The working copy is the one in `~/Work/projects/hstica/finalrun/`.** Treat
this directory as a snapshot, not as the place to edit. Re-copy after a
session that changes them.

Not included: `pipeline_output/`, which holds the fit iterations, figures and
tables (~90 MB and regenerable). The pass's outputs that matter for the paper
are `Data/CIV_measurements_v23.csv` in this repo.

What the pieces do:

- `run_batch.py` — first pass over a block of objects: co-add, copy in prior
  fits, propose masks, fit three variants, mark a recommended one.
- `iterate_fit.py` — one fit with an explicit override; the C IV
  sub-continuum veto and the no-C IV-data guard live here.
- `iterate_masks.py` — iterative absorption search (fit, detect, mask, refit).
- `mask_proposer.py` — absorption and cosmic-ray-spike proposals.
- `coadd_experiment.py` — inverse-variance combination on the shared lattice.
- `trial_z.py` — refit at a grid of trial redshifts to test whether an anchor
  offset is a redshift error or model shape.
- `pass_decisions.py`, `build_paper_table.py`, `finalize.py` — the ledger and
  the paper table.
- `check_redshift.py`, `query_redshifts.py` — NED/SIMBAD cross-checks.
- `refit_all.py` — replay recorded iterations after a change to the fit code.
- `object_paths.py` — index-prefixed folder naming shared by every script.
