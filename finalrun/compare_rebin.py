"""
compare_rebin.py -- does a freshly rebinned spectrum reproduce the one the
measurements so far were made on?

The rebinned master in Dropbox/Pratsos/RebinnedSpec_master was produced by the
students with an earlier checkout.  Before any object is measured against a
tree we rebuilt ourselves, the two have to be shown to agree on the objects
they share; if they do not, every measurement already made is against files we
cannot regenerate, and that has to be known before it happens rather than
after.

The master is opened read-only and never written to.

For each (object, instrument) present in BOTH directories this reports:
  n_old / n_new      pixel counts
  dlam_max           largest difference in rest-frame wavelength, in Angstrom
  z_old / z_new      the redshift each file was rebinned at
  dflux_med_pct      median |new-old|/|old| over pixels both call good, per cent
  dflux_max_pct      the worst such pixel
  mask_diff          pixels whose Bad Pixel Mask flag differs
  verdict            IDENTICAL, CLOSE (median < 0.1%), or DIFFERS

Usage:
  python compare_rebin.py --new  /path/to/RebinnedSpec_v23
                          [--old /Users/gtr/Dropbox/HST/Pratsos/RebinnedSpec_master]
                          [--csv pipeline_output/retrieval_v23/rebin_comparison.csv]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from astropy.table import Table

OLD_DEFAULT = '/Users/gtr/Dropbox/HST/Pratsos/RebinnedSpec_master'

W, F, E, M, Z = ('Rest-frame Wavelength', 'Coadded Flux (Arbitrary Units)',
                 'Coadded Flux Errors', 'Bad Pixel Mask', 'Redshift')


def _read(path):
    t = Table.read(path)
    return (np.asarray(t[W], float), np.asarray(t[F], float),
            np.asarray(t[E], float), np.asarray(t[M], float),
            float(np.nanmedian(np.asarray(t[Z], float))))


def compare_one(old_path, new_path):
    wo, fo, eo, mo, zo = _read(old_path)
    wn, fn, en, mn, zn = _read(new_path)
    row = dict(n_old=len(wo), n_new=len(wn), z_old=zo, z_new=zn)

    if len(wo) != len(wn):
        row.update(dlam_max=np.nan, dflux_med_pct=np.nan, dflux_max_pct=np.nan,
                   mask_diff=np.nan, verdict='DIFFERS (length)')
        return row

    row['dlam_max'] = float(np.nanmax(np.abs(wn - wo)))
    row['mask_diff'] = int(np.sum((mo != 0) != (mn != 0)))

    good = (mo == 0) & (mn == 0) & np.isfinite(fo) & np.isfinite(fn) & (np.abs(fo) > 0)
    if good.sum() < 10:
        row.update(dflux_med_pct=np.nan, dflux_max_pct=np.nan,
                   verdict='DIFFERS (no comparable pixels)')
        return row

    rel = np.abs(fn[good] - fo[good]) / np.abs(fo[good]) * 100.0
    row['dflux_med_pct'] = float(np.median(rel))
    row['dflux_max_pct'] = float(np.max(rel))

    if (row['dlam_max'] < 1e-6 and row['dflux_max_pct'] < 1e-6
            and row['mask_diff'] == 0):
        row['verdict'] = 'IDENTICAL'
    elif row['dflux_med_pct'] < 0.1 and row['dlam_max'] < 1e-3:
        row['verdict'] = 'CLOSE'
    else:
        row['verdict'] = 'DIFFERS'
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--new', required=True)
    ap.add_argument('--old', default=OLD_DEFAULT)
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    old_files = {f for f in os.listdir(args.old) if f.endswith('.fits')}
    new_files = {f for f in os.listdir(args.new) if f.endswith('.fits')}
    shared = sorted(old_files & new_files)

    print('old %s: %d files' % (args.old, len(old_files)))
    print('new %s: %d files' % (args.new, len(new_files)))
    print('shared: %d   new-only: %d   old-only: %d\n'
          % (len(shared), len(new_files - old_files), len(old_files - new_files)))

    rows = []
    for f in shared:
        stem = f[:-5]
        name, _, inst = stem.rpartition('_')
        try:
            r = compare_one(os.path.join(args.old, f), os.path.join(args.new, f))
        except Exception as exc:
            r = dict(verdict='ERROR: %s' % exc)
        r.update(common_name=name, instrument=inst, filename=f)
        rows.append(r)

    if not rows:
        print('no shared files to compare'); return 1

    d = pd.DataFrame(rows)
    cols = ['common_name', 'instrument', 'n_old', 'n_new', 'z_old', 'z_new',
            'dlam_max', 'dflux_med_pct', 'dflux_max_pct', 'mask_diff', 'verdict']
    d = d[[c for c in cols if c in d.columns]]
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(d.to_string(index=False))
    print('\nverdicts: %s' % d['verdict'].value_counts().to_dict())

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        d.to_csv(args.csv, index=False)
        print('-> %s' % args.csv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
