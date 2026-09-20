"""
pipeline/retrieve_spectra.py

Retrieve HST UV spectra from MAST for the quasar-winds sample, as a single
stated procedure that can be repeated from the paper.

WHY THIS EXISTS, alongside the older pipeline/download_spectra.py
-----------------------------------------------------------------
`download_spectra.py` does not query MAST for observations.  It recovers
obs_ids by cross-matching against three cached CSVs
(`v21_cache_mast_{cos,stis,fos}.csv`) that were produced by an earlier,
undocumented run, and it reads `master_catalog_v21.csv` and keeps only its
`has_civ=True` rows.  Two consequences, both measured in
`Data/missing_spectra_audit.csv`:

  * The v21 catalogue holds 257 objects against the 474 of the current sample,
    so 229 objects are absent from the caches entirely and were never retrieved.
  * Restricting to `has_civ=True` is the wrong criterion.  The ICA fits its
    components over 1260-3000 A, so a spectrum that does not reach C IV still
    constrains the reconstruction; an object only needs C IV in *some* spectrum.

More importantly for the paper: a retrieval that depends on cache files cannot
be written down as a method.  This script states the query instead, so the data
set can be rebuilt from the catalogue alone.

WHAT IT DOES
------------
For every object in the catalogue:
  1. cone-search MAST for HST spectra within SEARCH_RADIUS_ARCSEC of its
     position, per instrument family (COS, STIS, FOS);
  2. list the data products of every matching observation;
  3. keep SCIENCE products at calibration level 2 or 3 -- level 4 is the
     HASP/HLSP co-add, excluded because the pipeline does its own co-addition --
     and, for COS and STIS, only the extracted-spectrum subgroups;
  4. download what is not already on disk, into
     <data-dir>/MAST_v23/<common_name>/;
  5. write a manifest naming every product retrieved, so completeness can be
     checked without re-querying.

The filters in steps 3 reproduce those of `download_spectra.py` exactly, so
spectra retrieved by either route are interchangeable.

USAGE
-----
    python -m pipeline.retrieve_spectra --data-dir /path/to/data            # all
    python -m pipeline.retrieve_spectra --data-dir /path/to/data \
        --objects-from Data/missing_spectra_audit.csv                       # repair
    python -m pipeline.retrieve_spectra --data-dir /path/to/data --dry-run  # plan

`--dry-run` performs the queries and writes the plan but downloads nothing; it
is the way to check what a retrieval would fetch before spending the bandwidth.

Requires astroquery.  Run under the same environment as the rest of the
pipeline.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Retrieval parameters (these are the numbers the paper must quote) ─────────

SEARCH_RADIUS_ARCSEC = 3.0      # cone radius about the catalogue position
INSTRUMENTS = ('COS', 'STIS', 'FOS')
CALIB_LEVELS = [2, 3]           # 2,3 = calibrated exposures; 4 = HASP/HLSP co-adds
PRODUCT_SUBGROUPS = {           # None = filter on type and level only
    'COS':  ['X1D', 'X1DSUM'],  # FOS product subgroup names are non-standard,
    'STIS': ['X1D', 'SX1'],     # so FOS is filtered by productType/calib_level
    'FOS':  None,               # alone, as in download_spectra.py
}

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = _REPO_ROOT / 'Data' / 'master_catalog_v22.csv'


# ── Catalogue ────────────────────────────────────────────────────────────────

def load_catalog(path):
    """Objects to retrieve: name and position, every row, no C IV filtering.

    Every object in the sample is retrieved whether or not C IV is expected to
    be covered, because any UV spectrum constrains the ICA components.  Objects
    that genuinely cannot be measured are excluded later, at fitting time, on
    the evidence of the spectra themselves.
    """
    cat = pd.read_csv(path, skipinitialspace=True)
    cat.columns = [c.strip() for c in cat.columns]
    need = {'common_name', 'ra_deg', 'dec_deg'}
    missing = need - set(cat.columns)
    if missing:
        raise SystemExit('catalogue %s lacks column(s): %s' % (path, ', '.join(sorted(missing))))
    cat = cat[['common_name', 'ra_deg', 'dec_deg']].dropna()
    # MAST writes -1.0 for an unknown coordinate; drop those and anything
    # outside the valid ranges rather than cone-searching a nonsense position.
    ok = ((cat['ra_deg'] >= 0) & (cat['ra_deg'] <= 360) &
          (cat['dec_deg'] >= -90) & (cat['dec_deg'] <= 90) &
          (cat['ra_deg'] != -1.0) & (cat['dec_deg'] != -1.0))
    dropped = (~ok).sum()
    if dropped:
        print('  %d object(s) dropped for invalid coordinates' % dropped)
    return cat[ok].drop_duplicates('common_name').reset_index(drop=True)


def restrict_to_audit(cat, audit_path):
    """Keep only the objects an audit CSV lists as missing spectra."""
    aud = pd.read_csv(audit_path)
    want = set(aud['common_name'].astype(str))
    out = cat[cat['common_name'].astype(str).isin(want)].reset_index(drop=True)
    print('  restricted to %d of %d objects listed in %s'
          % (len(out), len(cat), os.path.basename(audit_path)))
    return out


# ── MAST ─────────────────────────────────────────────────────────────────────

def query_observations(obs_module, name, ra, dec):
    """Cone-search each instrument family; return one DataFrame of observations."""
    radius_deg = SEARCH_RADIUS_ARCSEC / 3600.0
    frames = []
    for inst in INSTRUMENTS:
        try:
            t = obs_module.query_criteria(
                obs_collection='HST',
                instrument_name='%s*' % inst,
                dataproduct_type='spectrum',
                s_ra=[ra - radius_deg, ra + radius_deg],
                s_dec=[dec - radius_deg, dec + radius_deg],
            )
        except Exception as exc:
            print('    WARNING  %s / %s: query failed: %s' % (name, inst, exc))
            continue
        if len(t) == 0:
            continue
        df = t.to_pandas()
        df['inst_family'] = inst
        df['common_name'] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _prefer_summed(prods):
    """Within one observation, keep X1DSUM in preference to its X1D siblings.

    For COS, X1DSUM is the sum of the sub-exposures of an association and X1D
    holds those sub-exposures individually; retrieving both fetches the same
    photons twice and, if both reach the co-adder, would double-weight them.
    Measured on the 64-object repair set, this is the common case: 1046 of 1327
    exposures returned two products each.

    Applied per obsID so that an observation without a summed product keeps its
    X1Ds. STIS is left alone -- its X1D and SX1 are different extraction modes
    (MAMA versus CCD ACCUM), not a sum and its parts.
    """
    if prods.empty or 'obsID' not in prods.columns:
        return prods
    keep_idx = []
    for _, grp in prods.groupby(['obsID', 'inst_family'], sort=False):
        sub = grp['productSubGroupDescription'].astype(str)
        if (grp['inst_family'] == 'COS').any() and (sub == 'X1DSUM').any():
            keep_idx.extend(grp.index[sub == 'X1DSUM'])
        else:
            keep_idx.extend(grp.index)
    return prods.loc[sorted(keep_idx)]


def filter_products(prods):
    """SCIENCE products at calibration level 2-3, extracted spectra only.

    The type, level and subgroup filters are identical to `_filter_products` in
    download_spectra.py, so spectra retrieved by either route are
    interchangeable.  The one deliberate difference is `_prefer_summed`: where
    COS offers both an X1DSUM and its constituent X1Ds, only the sum is taken.
    """
    if prods.empty:
        return prods
    prods = prods[(prods['productType'] == 'SCIENCE') &
                  (prods['calib_level'].isin(CALIB_LEVELS))]
    keep = []
    for inst, subgroups in PRODUCT_SUBGROUPS.items():
        m = prods['inst_family'] == inst
        if subgroups is None:
            keep.append(prods[m])
        else:
            keep.append(prods[m & prods['productSubGroupDescription'].isin(subgroups)])
    out = pd.concat(keep, ignore_index=False) if keep else pd.DataFrame()
    return _prefer_summed(out).reset_index(drop=True)


def products_for(obs_module, obs):
    """Product list for every observation, tagged with object and instrument."""
    rows = []
    for _, o in obs.iterrows():
        try:
            p = obs_module.get_product_list(o['obsid']).to_pandas()
        except Exception as exc:
            print('    WARNING  %s / %s: product list failed: %s'
                  % (o['common_name'], o.get('obs_id', '?'), exc))
            continue
        if p.empty:
            continue
        p['common_name'] = o['common_name']
        p['inst_family'] = o['inst_family']
        rows.append(p)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=os.environ.get('HST_PAPER_DATA_DIR'),
                    help='root for downloaded FITS; files land in <data-dir>/MAST_v23/<object>/')
    ap.add_argument('--catalog', default=str(DEFAULT_CATALOG),
                    help='object catalogue: needs common_name, ra_deg, dec_deg (default: %s)'
                         % DEFAULT_CATALOG.name)
    ap.add_argument('--objects-from', default=None, metavar='CSV',
                    help='retrieve only the objects named in this CSV\'s common_name column, '
                         'e.g. Data/missing_spectra_audit.csv')
    ap.add_argument('--dry-run', action='store_true',
                    help='query and write the plan, download nothing')
    ap.add_argument('--manifest', default=None,
                    help='where to write the manifest (default: <data-dir>/MAST_v23/retrieval_manifest.csv)')
    ap.add_argument('--sleep', type=float, default=0.0,
                    help='seconds to pause between objects, if MAST is rate-limiting')
    args = ap.parse_args()

    if not args.data_dir and not args.dry_run:
        ap.error('--data-dir is required (or set HST_PAPER_DATA_DIR), unless --dry-run')

    from astroquery.mast import Observations

    print('[1/4] Catalogue: %s' % args.catalog)
    cat = load_catalog(args.catalog)
    if args.objects_from:
        cat = restrict_to_audit(cat, args.objects_from)
    print('      %d objects to retrieve' % len(cat))

    out_root = Path(args.data_dir) / 'MAST_v23' if args.data_dir else None
    manifest_path = Path(args.manifest) if args.manifest else (
        out_root / 'retrieval_manifest.csv' if out_root else Path('retrieval_manifest.csv'))

    print('\n[2/4] Querying MAST (cone radius %.1f", instruments %s) ...'
          % (SEARCH_RADIUS_ARCSEC, '/'.join(INSTRUMENTS)))
    rows, no_obs, no_prod = [], [], []
    for i, o in cat.iterrows():
        name = str(o['common_name'])
        obs = query_observations(Observations, name, float(o['ra_deg']), float(o['dec_deg']))
        if obs.empty:
            no_obs.append(name); continue
        prods = filter_products(products_for(Observations, obs))
        if prods.empty:
            no_prod.append(name); continue
        for _, p in prods.iterrows():
            rows.append(dict(common_name=name, inst_family=p['inst_family'],
                             obs_id=p.get('obs_id', ''), productFilename=p['productFilename'],
                             dataURI=p['dataURI'], calib_level=p['calib_level'],
                             size_mb=round(float(p.get('size', 0) or 0) / 1e6, 3)))
        print('  %-30s %3d products  (%s)'
              % (name, len(prods), '/'.join(sorted(set(prods['inst_family'])))))
        if args.sleep:
            time.sleep(args.sleep)

    plan = pd.DataFrame(rows)
    print('\n      %d products across %d objects' % (len(plan), plan['common_name'].nunique() if len(plan) else 0))
    if no_obs:
        print('      %d object(s) with NO matching observation: %s'
              % (len(no_obs), ', '.join(no_obs[:8]) + (' ...' if len(no_obs) > 8 else '')))
    if no_prod:
        print('      %d object(s) whose observations yielded no products passing the filter: %s'
              % (len(no_prod), ', '.join(no_prod[:8]) + (' ...' if len(no_prod) > 8 else '')))

    if args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        plan.assign(status='planned').to_csv(manifest_path, index=False)
        print('\n[3/4] dry run: nothing downloaded')
        print('[4/4] plan -> %s' % manifest_path)
        return 0

    print('\n[3/4] Downloading into %s ...' % out_root)
    statuses = []
    for name, grp in plan.groupby('common_name'):
        dest = out_root / str(name).replace('/', '_')
        dest.mkdir(parents=True, exist_ok=True)
        todo = [r for _, r in grp.iterrows() if not (dest / r['productFilename']).exists()]
        if not todo:
            statuses += [('cached', r['dataURI']) for _, r in grp.iterrows()]
            print('  %-30s all %d products already present' % (name, len(grp)))
            continue
        # One file at a time via download_file, NOT download_products.
        # download_products wants the astropy product Table it produced itself;
        # handing it anything else (a pandas frame, a numpy recarray) fails
        # deep inside its own filtering with "Cannot compare structured or void
        # to non-void arrays", which is what silently lost the whole first
        # Tier A run.  download_file takes a plain dataURI and a destination
        # path, which is all the manifest carries anyway, and it lets a single
        # bad product fail without taking the object's other exposures with it.
        n_ok = n_bad = 0
        for r in todo:
            target = dest / r['productFilename']
            try:
                st_, msg, _url = Observations.download_file(
                    r['dataURI'], local_path=str(target), cache=True)
                if str(st_).upper() == 'COMPLETE' and target.exists():
                    statuses.append(('ok', r['dataURI'])); n_ok += 1
                else:
                    statuses.append(('error', r['dataURI'])); n_bad += 1
                    print('    ! %s: %s %s' % (r['productFilename'], st_, msg or ''))
            except Exception as exc:
                statuses.append(('error', r['dataURI'])); n_bad += 1
                print('    ! %s: %s' % (r['productFilename'], exc))
            if args.sleep:
                time.sleep(args.sleep)
        print('  %-30s fetched %d of %d%s'
              % (name, n_ok, len(grp), '  (%d FAILED)' % n_bad if n_bad else ''))

    st = dict((uri, s) for s, uri in statuses)
    plan['status'] = plan['dataURI'].map(lambda u: st.get(u, 'cached'))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(manifest_path, index=False)
    n_err = (plan['status'] == 'error').sum()
    print('\n[4/4] manifest -> %s   (%d ok/cached, %d errors)'
          % (manifest_path, len(plan) - n_err, n_err))
    if n_err:
        print('      re-run to retry the failures; existing files are skipped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
