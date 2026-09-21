"""One row per object: what its manmask is (or will be), checked against the
regenerated spectra.  Covers the 110 already worked and the 364 not yet reached."""
import os
import numpy as np
import pandas as pd
from astropy.table import Table

N = 'data_v23/RebinnedSpec_v23/'
W, E, MK = ('Rest-frame Wavelength', 'Coadded Flux Errors', 'Bad Pixel Mask')
CIV = 1549.48

files = [f for f in os.listdir(N) if f.endswith('.fits')]
by_obj = {}
for f in files:
    by_obj.setdefault(f[:-5].rpartition('_')[0], []).append(f)

def probe(obj):
    """usable C IV pixels, number of spectra, ICA-range pixels"""
    civ = ica = 0
    for f in by_obj.get(obj, []):
        t = Table.read(N + f); w = np.asarray(t[W], float)
        if not len(w):
            continue
        good = (np.asarray(t[MK]) == 0) & (np.asarray(t[E], float) > 0)
        civ = max(civ, int(((w >= CIV - 25) & (w <= CIV + 25) & good).sum()))
        ica = max(ica, int(((w >= 1260) & (w <= 3000) & good).sum()))
    return civ, len(by_obj.get(obj, [])), ica

q = pd.read_csv('pipeline_output/work_queue.csv')
q['common_name'] = q.name_mast_key.astype(str).str.strip()
dec = pd.read_csv('pipeline_output/final/pass_decisions.csv')
dec['common_name'] = dec.common_name.astype(str).str.strip()
done = dict(zip(dec.common_name, dec.manmask))
reason = dict(zip(dec.common_name, dec.reject_reason.fillna('')))

ex = pd.read_csv('pipeline_output/retrieval_v23/exclusion_recheck.csv')
changed = dict(zip(ex.common_name.astype(str).str.strip(), ex.what_changed))

rows = []
for _, r in q.iterrows():
    o = r.common_name
    civ, nsp, ica = probe(o)
    mm = done.get(o)
    worked = int(r['order']) <= 110 and mm is not None

    if nsp == 0:
        expect, why = 1, 'iue_only: no HST observation in the archive'
    elif civ < 20:
        expect, why = 0, 'no_civ_data: %d usable pixels at C IV' % civ
    else:
        expect, why = None, 'fittable: %d px at C IV across %d spectra' % (civ, nsp)

    if worked:
        # An exclusion is only worth revisiting if the DATA changed.  Most
        # manmask-0 objects were excluded on quality -- low S/N, a
        # sub-continuum reconstruction, an unstable blueshift -- and having
        # C IV pixels was never in doubt for them, so presence of C IV is not
        # the test.  exclusion_recheck.csv holds the comparison of each
        # excluded object's spectra before and after the regeneration.
        if mm == 1 and expect != 1:
            status = 'RECONSIDER - carried as IUE-only but HST data exists'
        elif mm == 0 and changed.get(o, 'nothing') != 'nothing':
            status = 'RECONSIDER - %s' % changed[o]
        elif mm in (0, 1):
            status = 'confirmed'
        else:
            status = 'measured'
    else:
        status = 'not yet reached'

    rows.append(dict(index=int(r['order']), common_name=o, name_pub=r['name_pub'],
                     manmask_now=mm, manmask_expected=expect, status=status,
                     civ_px=civ, n_spectra=nsp, ica_px=ica, basis=why,
                     reason_now=reason.get(o, '')))

d = pd.DataFrame(rows).sort_values('index')
d.to_csv('pipeline_output/final/manmask_status_v23.csv', index=False)

print('=== indices 1-110, already worked ===')
w = d[d.status.isin(['confirmed', 'measured']) | d.status.str.startswith('RECONSIDER')]
w = w[w['index'] <= 110]
print(w.groupby(['manmask_now', 'status']).size().to_string())
print()
print('=== indices 111-474, not yet reached: what they will be ===')
nr = d[d.status == 'not yet reached']
print('  manmask 1 (no HST data at all)     : %d' % int((nr.manmask_expected == 1).sum()))
print('  manmask 0 (HST data, no C IV)      : %d' % int((nr.manmask_expected == 0).sum()))
print('  fittable                           : %d' % int(nr.manmask_expected.isna().sum()))
print()
print('-> pipeline_output/final/manmask_status_v23.csv')
