"""Compare two rebinned trees over the C IV window, spectrum by spectrum.

Written to size the damage from the NaN-corrupted weighted median in the
co-adder (see figure_coadd_nanmedian.py): it reports, for every spectrum the two
trees share, how much the C IV data actually moved and whether the co-add got
smoother.  Objects whose pixels barely move keep whatever verdict they already
have; only the movers need looking at again.

    python diff_trees.py --old data_v23/RebinnedSpec_v23 --new data_v23/RebinnedSpec_v24
"""
import argparse
import glob
import os

import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
CIV = (1500.0, 1600.0)


def read(path):
    with fits.open(path) as h:
        t = h[1].data
        return (np.asarray(t.field(0), float), np.asarray(t.field(1), float),
                np.asarray(t.field(2), float), np.asarray(t.field(3), float))


def roughness(f, e):
    """Pixel-to-pixel scatter divided by the quoted error: 1 means consistent."""
    if f.size < 10:
        return np.nan
    d = np.median(np.abs(np.diff(f))) / np.sqrt(2) / 0.6745
    m = np.median(e)
    return d / m if m > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old', default=os.path.join(HERE, 'data_v23', 'RebinnedSpec_v23'))
    ap.add_argument('--new', default=os.path.join(HERE, 'data_v23', 'RebinnedSpec_v24'))
    ap.add_argument('--out', default=os.path.join(HERE, 'pipeline_output', 'final',
                                                  'tree_diff_v23_v24.csv'))
    a = ap.parse_args()

    rows = []
    for p_new in sorted(glob.glob(os.path.join(a.new, '*.fits'))):
        stem = os.path.basename(p_new)[:-5]
        p_old = os.path.join(a.old, stem + '.fits')
        if not os.path.exists(p_old):
            rows.append(dict(stem=stem, status='new only'))
            continue
        wo, fo, eo, mo = read(p_old)
        wn, fn, en, mn = read(p_new)
        idx = {round(x, 4): i for i, x in enumerate(wo)}
        pair = [(idx[round(x, 4)], j) for j, x in enumerate(wn) if round(x, 4) in idx]
        if not pair:
            rows.append(dict(stem=stem, status='no shared lattice'))
            continue
        io = np.array([p[0] for p in pair]); jn = np.array([p[1] for p in pair])
        s = (wn[jn] >= CIV[0]) & (wn[jn] <= CIV[1])
        s &= np.isfinite(fo[io]) & np.isfinite(fn[jn]) & (eo[io] > 0) & (en[jn] > 0)
        if s.sum() < 10:
            rows.append(dict(stem=stem, status='no usable C IV pixels'))
            continue
        d = np.abs(fn[jn][s] - fo[io][s]) / eo[io][s]
        rows.append(dict(
            stem=stem, status='ok', n_civ=int(s.sum()),
            frac_gt1sig=float((d > 1).mean()), frac_gt3sig=float((d > 3).mean()),
            median_dsig=float(np.median(d)),
            rough_old=roughness(fo[io][s], eo[io][s]),
            rough_new=roughness(fn[jn][s], en[jn][s]),
            usable_old=int(((wo >= CIV[0]) & (wo <= CIV[1]) & (mo == 0) & (eo > 0)).sum()),
            usable_new=int(((wn >= CIV[0]) & (wn <= CIV[1]) & (mn == 0) & (en > 0)).sum()),
        ))

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    ok = df[df.status == 'ok']
    print('%d spectra compared (%d skipped)' % (len(ok), len(df) - len(ok)))
    if len(ok):
        print('  median |delta|/sigma per pixel : %.2f' % ok.median_dsig.median())
        print('  roughness  old %.2f -> new %.2f (median)'
              % (ok.rough_old.median(), ok.rough_new.median()))
        print('  spectra with >25%% of C IV pixels moving >1 sigma : %d'
              % (ok.frac_gt1sig > 0.25).sum())
        print('  spectra with  >5%% of C IV pixels moving >3 sigma : %d'
              % (ok.frac_gt3sig > 0.05).sum())
    print('-> %s' % a.out)


if __name__ == '__main__':
    main()
