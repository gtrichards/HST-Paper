"""Re-test every manmask-0 exclusion against the regenerated spectra.

Each exclusion's reason was judged against the master, which we now know was
sometimes truncated or missing instruments.  For each excluded object this
compares what the master gave it against what the new tree gives it, on the
quantities the reasons actually rest on: C IV coverage, how much of the ICA
range 1260-3000 A is covered, and the signal-to-noise in the C IV window.
No fitting: this only measures the spectra.
"""
import os
import numpy as np
import pandas as pd
from astropy.table import Table

M = '/Users/gtr/Dropbox/HST/Pratsos/RebinnedSpec_master/'
N = 'data_v23/RebinnedSpec_v23/'
W, F, E, MK = ('Rest-frame Wavelength', 'Coadded Flux (Arbitrary Units)',
               'Coadded Flux Errors', 'Bad Pixel Mask')
CIV, LO, HI = 1549.48, 1500.0, 1600.0


def files_for(D, obj):
    return [f for f in os.listdir(D)
            if f.endswith('.fits') and f[:-5].rpartition('_')[0] == obj]


def probe(D, fs):
    """Best C IV coverage, ICA-range coverage and C IV-window S/N over an
    object's spectra.  Taken per file, not co-added, since the fit uses one
    spectrum (or a merge of them) and the best single file is the fair bound."""
    civ_px, ica_px, snr = 0, 0, 0.0
    for f in fs:
        t = Table.read(D + f)
        w = np.asarray(t[W], float)
        if not len(w):
            continue
        # "Covers C IV" has to mean usable pixels at the line, not a wavelength
        # range that happens to straddle it.  Several excluded objects span
        # 1549 A with a detector gap exactly there, which is why their C IV
        # signal-to-noise reads zero; counting the range as coverage would have
        # reported their exclusions as questionable when they are correct.
        core = (w >= CIV - 25) & (w <= CIV + 25) & (np.asarray(t[MK]) == 0) \
            & np.isfinite(np.asarray(t[F], float)) & (np.asarray(t[E], float) > 0)
        civ_px = max(civ_px, int(core.sum()))
        ica_px = max(ica_px, int(((w >= 1260) & (w <= 3000)).sum()))
        sel = (w >= LO) & (w <= HI) & (np.asarray(t[MK]) == 0)
        if sel.sum() > 10:
            fl, er = np.asarray(t[F], float)[sel], np.asarray(t[E], float)[sel]
            good = np.isfinite(fl) & np.isfinite(er) & (er > 0)
            if good.sum() > 10:
                snr = max(snr, float(np.median(np.abs(fl[good]) / er[good])))
    return civ_px, ica_px, snr


dec = pd.read_csv('/Users/gtr/Work/git/HST-Paper/Data/pass_decisions.csv')
dec['common_name'] = dec.common_name.astype(str).str.strip()
rows = []
for _, r in dec[dec.manmask == 0].iterrows():
    obj = r.common_name
    fo, fn = files_for(M, obj), files_for(N, obj)
    co, io, so = probe(M, fo) if fo else (False, 0, 0.0)
    cn, inw, sn = probe(N, fn) if fn else (False, 0, 0.0)

    changed = []
    if cn >= 20 and co < 20:
        changed.append('C IV now covered (%d px, was %d)' % (cn, co))
    if inw > io * 1.2 and inw - io > 100:
        changed.append('ICA range +%d px' % (inw - io))
    if so > 0 and sn > so * 1.3:
        changed.append('S/N x%.1f' % (sn / so))
    elif so == 0 and sn > 0:
        changed.append('S/N now measurable')
    if len(fn) > len(fo):
        changed.append('+%d spectra' % (len(fn) - len(fo)))

    rows.append(dict(index=int(r['index']), common_name=obj,
                     reject_reason=r.reject_reason,
                     files_old=len(fo), files_new=len(fn),
                     civ_px_old=co, civ_px_new=cn,
                     ica_px_old=io, ica_px_new=inw,
                     snr_old=round(so, 2), snr_new=round(sn, 2),
                     what_changed='; '.join(changed) or 'nothing'))

d = pd.DataFrame(rows).sort_values('index')
d.to_csv('pipeline_output/retrieval_v23/exclusion_recheck.csv', index=False)
with pd.option_context('display.width', 250, 'display.max_colwidth', 40):
    print(d[['index', 'common_name', 'reject_reason', 'files_old', 'files_new',
             'civ_px_old', 'civ_px_new', 'ica_px_old', 'ica_px_new',
             'snr_old', 'snr_new', 'what_changed']].to_string(index=False))
print('\n-> pipeline_output/retrieval_v23/exclusion_recheck.csv')
