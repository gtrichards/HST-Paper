"""
pipeline/retrieve_sdss.py

Find and fetch the SDSS spectrum for each object in the sample, so the
HST+SDSS grouping can be fitted.

WHY THIS EXISTS

`rebinning/coadd.py` already knows how to use an SDSS spectrum: given a
plate-MJD-fibre name it fits an SDSS continuum, morphs it onto the same scale
as the ultraviolet and splices the two together.  What has never been supplied
is the name.  The catalogue bridge passes `fn_sdss=None` for every object, so
the whole sample is currently HST-only -- including the 315 objects whose
category is `sdss`, two thirds of the working list.

This script supplies the missing half: it resolves each object to an SDSS
spectroscopic observation and downloads the "lite" spectrum into the layout
`coadd.rebin` expects,

    <data-dir>/SDSS_spec/lite/<plate>/spec-<plate>-<mjd>-<fibre>.fits

which is the same convention the older Sulentic route used.

IDENTIFICATION IS NOT AUTOMATIC

Coordinate matching is not safe for identifying these AGN on its own -- the
sample contains confusable pairs a few arcseconds apart, and the HST retrieval
already turned up two cone matches that were a different pointing entirely (a
target named NONE at 2", a DEUTERIUM calibration exposure at 3.4").  So this
script *records* rather than decides: every match carries its separation, its
SDSS class and redshift, the number of candidates within the radius, and the
difference between the SDSS and catalogue redshifts.  A match is marked
`review` rather than `ok` when it is far, when more than one candidate lies
inside the radius, when the class is not QSO or GALAXY, or when the redshifts
disagree by more than 0.01.  Nothing downstream should consume a `review` row
without a human having looked at it.

The v22 sheet's `sdss_flag` is an independent check: non-zero means GTR
recorded an SDSS spectrum for that object (2 = Mg II covered, 1 = not).  An
object flagged in the sheet with no match found here, or a match found for an
object with a blank flag, is worth reporting rather than silently accepting.

USAGE

    python -m pipeline.retrieve_sdss --data-dir /path/to/data_v23 \
        --manifest .../sdss_manifest.csv [--limit N] [--dry-run]

The manifest is a running record and a later run resumes from it, as with
pipeline/retrieve_spectra.py.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CATALOG = os.path.join(_REPO_ROOT, 'Data', 'master_catalog_v22.csv')

SEARCH_RADIUS_ARCSEC = 3.0
DATA_RELEASE = 17
GOOD_CLASSES = ('QSO', 'GALAXY')
Z_TOLERANCE = 0.01          # |z_sdss - z_catalog| above which the match is suspect

COLUMNS = ['common_name', 'ra_deg', 'dec_deg', 'plate', 'mjd', 'fiberID', 'run2d',
           'z_sdss', 'sdss_class', 'sep_arcsec', 'n_candidates', 'n_spectra',
           'science_primary', 'sn_median', 'dz', 'sdss_flag', 'verdict',
           'fn_sdss', 'status']


def load_catalog(path):
    cat = pd.read_csv(path, skipinitialspace=True)
    cat.columns = [c.strip() for c in cat.columns]
    cat['common_name'] = cat['common_name'].astype(str).str.strip()
    ok = ((cat['ra_deg'] >= 0) & (cat['ra_deg'] <= 360) &
          (cat['dec_deg'] >= -90) & (cat['dec_deg'] <= 90) &
          (cat['ra_deg'] != -1.0) & (cat['dec_deg'] != -1.0))
    if (~ok).sum():
        print('  %d object(s) dropped for invalid coordinates' % int((~ok).sum()), flush=True)
    return cat[ok].drop_duplicates('common_name').reset_index(drop=True)


def sdss_flags(xlsx_path):
    """GTR's sdss_flag per object: non-zero means an SDSS spectrum exists."""
    try:
        import openpyxl
    except ImportError:
        return {}
    try:
        ws = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)['CIV_measurements_v22']
    except Exception:
        return {}
    rows = list(ws.iter_rows(values_only=True))
    hdr = list(rows[0])
    if 'common_name' not in hdr or 'sdss_flag' not in hdr:
        return {}
    ci, fi = hdr.index('common_name'), hdr.index('sdss_flag')
    return {str(r[ci]).strip(): r[fi] for r in rows[1:] if r[ci]}


def query_one(SDSS, coord, name):
    """Every SDSS spectroscopic match inside the search radius, nearest first."""
    from astropy import units as u
    try:
        t = SDSS.query_region(coord, radius=SEARCH_RADIUS_ARCSEC * u.arcsec,
                              spectro=True, data_release=DATA_RELEASE,
                              # objID is deliberately absent: asking for it
                              # makes the SkyServer query fail outright with
                              # "Number of header columns (1) inconsistent with
                              # data columns".  cache=False because astroquery
                              # will otherwise serve that failure back for every
                              # later query too.
                              specobj_fields=['plate', 'mjd', 'fiberID', 'run2d',
                                              'z', 'class', 'ra', 'dec',
                                              'sciencePrimary', 'snMedian'],
                              cache=False)
    except Exception as exc:
        print('    WARNING  %s: SDSS query failed: %s' % (name, exc), flush=True)
        return None
    if t is None or len(t) == 0:
        return None
    d = t.to_pandas()
    d['sep_arcsec'] = coord.separation(_coords(d)).arcsec
    # An object observed more than once returns one row per spectrum.  That is
    # not ambiguity, and counting rows as candidates marked most of the sample
    # for review; repeats are separated by position instead.  Among repeats of
    # one object SDSS's own sciencePrimary flag says which spectrum to use, with
    # median signal-to-noise as the tie-break.
    d['sciencePrimary'] = pd.to_numeric(d.get('sciencePrimary'), errors='coerce').fillna(0)
    d['snMedian'] = pd.to_numeric(d.get('snMedian'), errors='coerce').fillna(0)
    d = d.sort_values(['sciencePrimary', 'snMedian', 'sep_arcsec'],
                      ascending=[False, False, True]).reset_index(drop=True)
    return d


def _coords(d):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    return SkyCoord(np.asarray(d['ra'], float) * u.deg, np.asarray(d['dec'], float) * u.deg)


def judge(row, n_objects, z_cat, flag):
    """Is this match safe to use without a human looking at it?"""
    reasons = []
    if row['sep_arcsec'] > 2.0:
        reasons.append('%.1f arcsec away' % row['sep_arcsec'])
    if n_objects > 1:
        reasons.append('%d distinct objects in the cone' % n_objects)
    cls = str(row.get('class', '')).strip().upper()
    if cls not in GOOD_CLASSES:
        reasons.append('class %s' % (cls or 'unknown'))
    if np.isfinite(z_cat) and abs(float(row['z']) - z_cat) > Z_TOLERANCE:
        reasons.append('dz = %.4f' % (float(row['z']) - z_cat))
    if flag in (None, '', 0) or (isinstance(flag, float) and np.isnan(flag)):
        reasons.append('sdss_flag is blank in the v22 sheet')
    return ('ok', '') if not reasons else ('review', '; '.join(reasons))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=os.environ.get('HST_PAPER_DATA_DIR'), required=False,
                    help='spectra land in <data-dir>/SDSS_spec/lite/<plate>/')
    ap.add_argument('--catalog', default=DEFAULT_CATALOG)
    ap.add_argument('--xlsx', default='/Users/gtr/Dropbox/HST/Summer2026/CIV_measurements_v22.xlsx',
                    help="the v22 sheet, read only for its sdss_flag column")
    ap.add_argument('--objects-from', default=None, metavar='CSV')
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--limit', type=int, default=0, metavar='N',
                    help='stop after N objects this run (0 = no limit)')
    ap.add_argument('--dry-run', action='store_true', help='resolve but download nothing')
    ap.add_argument('--sleep', type=float, default=0.0)
    args = ap.parse_args()

    if not args.data_dir and not args.dry_run:
        ap.error('--data-dir is required unless --dry-run')

    from astroquery.sdss import SDSS
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    cat = load_catalog(args.catalog)
    if args.objects_from:
        want = set(pd.read_csv(args.objects_from)['common_name'].astype(str))
        cat = cat[cat['common_name'].isin(want)].reset_index(drop=True)
    flags = sdss_flags(args.xlsx)
    print('[1/3] %d objects; sdss_flag read for %d of them' % (len(cat), len(flags)), flush=True)

    manifest_path = args.manifest or os.path.join(args.data_dir or '.', 'sdss_manifest.csv')
    os.makedirs(os.path.dirname(manifest_path) or '.', exist_ok=True)
    done = set()
    if os.path.exists(manifest_path):
        prev = pd.read_csv(manifest_path)
        done = set(prev['common_name'].astype(str))
        prev.to_csv(manifest_path, index=False)
        print('  resuming: %d object(s) already resolved' % len(done), flush=True)
    else:
        pd.DataFrame(columns=COLUMNS).to_csv(manifest_path, index=False)

    out_root = os.path.join(args.data_dir, 'SDSS_spec', 'lite') if args.data_dir else None
    print('\n[2/3] resolving against SDSS DR%d (radius %.1f")' % (DATA_RELEASE, SEARCH_RADIUS_ARCSEC),
          flush=True)

    n_done = n_ok = n_review = n_none = n_fetched = 0
    for _, o in cat.iterrows():
        name = str(o['common_name'])
        if name in done:
            continue
        coord = SkyCoord(float(o['ra_deg']) * u.deg, float(o['dec_deg']) * u.deg)
        d = query_one(SDSS, coord, name)
        z_cat = float(o.get('best_z', np.nan))
        flag = flags.get(name)

        if d is None or len(d) == 0:
            row = dict(common_name=name, ra_deg=o['ra_deg'], dec_deg=o['dec_deg'],
                       sdss_flag=flag, verdict='none', status='no_match')
            n_none += 1
            if flag not in (None, '', 0) and not (isinstance(flag, float) and np.isnan(flag)):
                print('  %-30s NO SDSS MATCH but the v22 sheet flags one (flag=%s)'
                      % (name, flag), flush=True)
        else:
            best = d.iloc[0]
            # Distinct objects, not distinct spectra.  objID would say this
            # directly but including it makes the SkyServer query fail, so
            # repeats are identified by position: rows within 1" of the chosen
            # match are the same source observed again.
            n_objects = 1 + int((_coords(d).separation(_coords(d.iloc[[0]])[0]).arcsec > 1.0).sum())
            verdict, why = judge(best, n_objects, z_cat, flag)
            fn = '%04d/spec-%04d-%05d-%04d.fits' % (int(best['plate']), int(best['plate']),
                                                    int(best['mjd']), int(best['fiberID']))
            row = dict(common_name=name, ra_deg=o['ra_deg'], dec_deg=o['dec_deg'],
                       plate=int(best['plate']), mjd=int(best['mjd']),
                       fiberID=int(best['fiberID']), run2d=str(best.get('run2d', '')),
                       z_sdss=float(best['z']), sdss_class=str(best.get('class', '')),
                       sep_arcsec=round(float(best['sep_arcsec']), 3),
                       n_candidates=n_objects, n_spectra=len(d),
                       science_primary=int(best.get('sciencePrimary', 0)),
                       sn_median=round(float(best.get('snMedian', np.nan)), 2),
                       dz=round(float(best['z']) - z_cat, 5) if np.isfinite(z_cat) else np.nan,
                       sdss_flag=flag, verdict=verdict, fn_sdss=fn, status='resolved')
            n_ok += verdict == 'ok'
            n_review += verdict == 'review'
            if verdict == 'review':
                print('  %-30s REVIEW: %s' % (name, why), flush=True)

            if not args.dry_run and out_root:
                dest = os.path.join(out_root, '%04d' % int(best['plate']))
                os.makedirs(dest, exist_ok=True)
                target = os.path.join(dest, os.path.basename(fn))
                if os.path.exists(target) and os.path.getsize(target) > 0:
                    row['status'] = 'cached'
                    n_fetched += 1
                else:
                    try:
                        sp = SDSS.get_spectra(plate=int(best['plate']), mjd=int(best['mjd']),
                                              fiberID=int(best['fiberID']),
                                              data_release=DATA_RELEASE)
                        if sp:
                            sp[0].writeto(target, overwrite=True)
                            row['status'] = 'ok'
                            n_fetched += 1
                        else:
                            row['status'] = 'download_empty'
                    except Exception as exc:
                        row['status'] = 'error'
                        print('    ! %s: %s' % (fn, exc), flush=True)

        pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(
            manifest_path, mode='a', header=False, index=False)
        n_done += 1
        if args.sleep:
            time.sleep(args.sleep)
        if args.limit and n_done >= args.limit:
            print('  stopping after %d object(s) this run (--limit)' % n_done, flush=True)
            break

    print('\n[3/3] %d resolved this run: %d ok, %d need review, %d with no SDSS match; '
          '%d spectra on disk' % (n_done, n_ok, n_review, n_none, n_fetched), flush=True)
    print('      manifest -> %s' % manifest_path, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
