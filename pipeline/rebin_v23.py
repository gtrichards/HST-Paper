"""
pipeline/rebin_v23.py

Rebin the spectra that pipeline/retrieve_spectra.py fetched.

WHY THIS EXISTS, alongside pipeline/catalog_bridge.py

`catalog_bridge.build_object_list()` is the bridge the earlier rebin run used,
and it cannot be reused for the v22/v23 sample for three reasons, each of which
silently loses spectra rather than failing:

  1. it reads `master_catalog_v21.csv`, an older and shorter object list, so an
     object added in v22 has no redshift and is skipped;
  2. it keeps only `has_civ == True` rows.  The ICA components span 1260-3000 A
     and the component weights are constrained by the whole ultraviolet
     spectrum, so a spectrum that does not itself reach C IV still improves the
     C IV reconstruction.  Dropping those spectra degrades fits for objects
     that look fully covered;
  3. it reads `download_manifest.csv`, written by the older retrieval, so
     anything fetched by `retrieve_spectra.py` is invisible to it.

This module builds the same list of dicts -- name, redshift, instrument,
data_path -- from the v22 catalogue and one or more retrieval manifests, with
no coverage filter, and hands it to the unchanged
`rebinning.run_rebin.run_from_catalog_bridge`.  The rebinning itself is not
modified.

REDSHIFT

`best_z` from `master_catalog_v22.csv`.  `coadd.rebin` writes its output in the
rest frame, dividing the observed wavelength by (1+z), so a new spectrum must
use the same redshift as the spectra of the same object already rebinned or
the two will not lie on a common lattice.  All 576 files in the existing
rebinned master carry exactly the v21 `best_z`, and v21 and v22 agree on
`best_z` for every one of them, so `best_z` from v22 is the consistent choice.
The redshift overrides in `ica/redshift_overrides.csv` are applied at fit time,
not here, and must not be applied here.

OUTPUT

Defaults to `<data-dir>/RebinnedSpec_v23`, deliberately NOT the rebinned master.
Adding an instrument to an object changes its combined spectrum and therefore
its measurement, so the new files are reviewed before being merged into the
master.

Usage:
  python -m pipeline.rebin_v23 \
      --data-dir  /path/to/data_v23 \
      --manifest  .../manifest_tierA.csv .../manifest_tierB.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CATALOG = _REPO_ROOT / 'Data' / 'master_catalog_v22.csv'


def _load_catalog(path):
    cat = pd.read_csv(path, skipinitialspace=True)
    cat.columns = [c.strip() for c in cat.columns]
    for c in ('common_name', 'best_z'):
        if c not in cat.columns:
            raise KeyError('%s has no %r column' % (path, c))
    cat['common_name'] = cat['common_name'].astype(str).str.strip()
    return cat


def build_object_list(raw_root, manifests, catalog_path=_DEFAULT_CATALOG,
                      verbose=True):
    """
    One entry per (object, instrument) directory that holds retrieved FITS.

    `raw_root` is the MAST_v23 directory; `manifests` are the CSVs written by
    retrieve_spectra.py.  The manifest says what was asked for, the directory
    says what arrived; only pairs with files actually on disk are returned, so
    a partly failed retrieval rebins what it has rather than nothing.
    """
    raw_root = Path(raw_root)
    cat = _load_catalog(catalog_path)
    z_of = dict(zip(cat['common_name'], cat['best_z']))

    frames = []
    for m in manifests:
        d = pd.read_csv(m)
        if 'status' in d.columns:
            d = d[d['status'].isin(['ok', 'cached'])]
        frames.append(d[['common_name', 'inst_family']])
    if not frames:
        return []
    pairs = (pd.concat(frames, ignore_index=True)
               .drop_duplicates()
               .sort_values(['common_name', 'inst_family'])
               .reset_index(drop=True))

    out, no_z, no_files = [], [], []
    for _, r in pairs.iterrows():
        name, inst = str(r['common_name']), str(r['inst_family'])
        path = raw_root / name.replace('/', '_') / inst
        fits = sorted(path.glob('*.fits')) if path.exists() else []
        if not fits:
            no_files.append('%s/%s' % (name, inst))
            continue
        z = z_of.get(name)
        if z is None or pd.isna(z):
            no_z.append('%s/%s' % (name, inst))
            continue
        out.append(dict(name=name, redshift=float(z), instrument=inst,
                        data_path=path, n_files=len(fits)))

    if verbose:
        print('rebin_v23: %d (object, instrument) pairs ready from %d files'
              % (len(out), sum(o['n_files'] for o in out)))
        if no_files:
            print('  %d pair(s) with no FITS on disk: %s'
                  % (len(no_files), ', '.join(no_files[:6])
                     + (' ...' if len(no_files) > 6 else '')))
        if no_z:
            print('  %d pair(s) with no redshift in the catalogue: %s'
                  % (len(no_z), ', '.join(no_z[:6])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    ap.add_argument('--data-dir', required=True,
                    help='the directory given to retrieve_spectra.py; '
                         'raw FITS are read from <data-dir>/MAST_v23')
    ap.add_argument('--manifest', nargs='+', required=True,
                    help='one or more manifests written by retrieve_spectra.py')
    ap.add_argument('--catalog', default=str(_DEFAULT_CATALOG))
    ap.add_argument('--output-dir', default=None,
                    help='default <data-dir>/RebinnedSpec_v23')
    ap.add_argument('--sdss-spec-dir', default=None)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--no-skip-existing', dest='skip_existing',
                    action='store_false')
    ap.add_argument('--dry-run', action='store_true',
                    help='list what would be rebinned and stop')
    ap.set_defaults(skip_existing=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    raw_root = data_dir / 'MAST_v23'
    out_dir = Path(args.output_dir) if args.output_dir \
        else data_dir / 'RebinnedSpec_v23'
    sdss_dir = args.sdss_spec_dir or str(data_dir / 'SDSS_spec' / 'lite')

    print('[1/3] building the object list from %s' % raw_root)
    objects = build_object_list(raw_root, args.manifest, args.catalog)
    if not objects:
        print('nothing to rebin'); return 1

    if args.dry_run:
        print('\n[2/3] dry run -- would rebin into %s' % out_dir)
        for o in objects:
            print('  %-34s %-4s  %3d file(s)  z=%.5f'
                  % (o['name'], o['instrument'], o['n_files'], o['redshift']))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    print('\n[2/3] rebinning into %s' % out_dir)
    from rebinning.run_rebin import run_from_catalog_bridge
    n_good, n_fail, n_skip, failed = run_from_catalog_bridge(
        objects, str(out_dir), sdss_dir,
        skip_existing=args.skip_existing, workers=args.workers)

    print('\n[3/3] %d ok, %d failed, %d skipped' % (n_good, n_fail, n_skip))
    for inst, name, etype, msg in failed:
        print('  FAIL %-4s %-34s %s: %s' % (inst, name, etype, msg))
    return 0 if n_fail == 0 else 2


if __name__ == '__main__':
    sys.exit(main())
