# Redshift check

GTR's object-by-object check of the systemic redshifts of the measured
objects (artifact: https://claude.ai/artifact/BsGXCfqSusid2qoQpqebng; his
verdicts live in its `zchecks` database collection).

## What the page shows
For each measured object: the redshift used (Table 1, with the live
`ica/redshift_overrides.csv` applied on top), NED's preferred value and every
published redshift NED lists (with refcodes), our fits' C III] / Mg II offsets,
and line-region spectra plotted against velocity relative to the redshift used,
with each candidate redshift drawn.  In the UV only the C III] peak and Mg II
are shown (Si IV, C IV, He II can be blueshifted -- that is the measurement).

## Pipeline (run in a work directory WD with subfolders zcheck/, optical/, optical2/)
1. `build_data.py WD/zcheck` -- Table 1 + overrides + NED (needs `WD/zcheck/ned/*.json`
   from the NED fetch; that cache was lost, so `zcheck_data.json` here is the
   preserved build product -- rerunning build_data needs the NED fetch redone).
2. `build_spectra.py WD/zcheck` -- SDSS optical and UV (C III], Mg II) panels.
3. `ned_spectra.py WD` then `build_optical.py WD` -- NED-archive optical / H I
   spectra.  Excluded: Boroson & Green (1992) files (de-redshifted with their
   own z, not listed in NED); files whose header position is >120" from ours
   (NED attached other objects' spectra); files coarser than 250 km/s/pixel.
   Dispersion from W0/WPC or CDELT1 before CD1_1 (BG92's CD1_1 is rounded).
   6dF: the combined "SPECTRUM VR" extension.  Air -> vacuum unless the file
   says vacuum.  A file is treated as rest-frame (and excluded) only if a
   noise-weighted narrow-line template scores >2x better there.
4. `survey_archives.py WD`, `download.py WD/optical2`, `build_optical2.py WD`
   -- HST optical gratings on disk, SDSS DR18 re-query, LAMOST, ESO archive.
   ESO: air -> vacuum, topocentric -> barycentric (QC VRAD BARYCOR or computed
   for Paranal), outer 20 A and >50x-median spikes dropped; UVES binned to
   ~25 km/s.
5. `template.html` + the two JSON files -> the page (replace __DATA__ and
   __SPEC__); `check.js` syntax-checks its script with osascript -l JavaScript.

Downloaded spectra: `~/Work/projects/hstica/finalrun/data_v23/optical_lit/`.
The narrow-line redshift fit used for overrides: `../fit_narrow_redshift.py`.
