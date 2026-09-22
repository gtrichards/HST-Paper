"""Where do an object's C IV pixels die?  Follows one instrument's exposures
through the reader, the rebin and the per-exposure continuum, counting usable
pixels within 25 A of C IV at each stage.  Written while chasing why 35
rebinned spectra had no C IV when their raw files did."""
import sys, numpy as np
sys.path.insert(0, '/Users/gtr/Work/git/HST-Paper')
from rebinning import coadd, spec_morph, read_spec_data as rsd
obj, inst, z = sys.argv[1], sys.argv[2], float(sys.argv[3])
lo, hi = 1524 * (1 + z), 1574 * (1 + z)
path = '/Users/gtr/Work/projects/hstica/finalrun/data_v23/MAST_v23/%s/%s/' % (obj, inst)
w, f, e, m = rsd.read_data_flat(obj, path, inst, z)
print('%s %s z=%.4f: %d rows after the reader' % (obj, inst, z, w.shape[0]))
for i in range(w.shape[0]):
    s = (w[i] >= lo) & (w[i] <= hi)
    if s.sum() == 0:
        continue
    r1 = int(((m[i] == 0) & (e[i] > 0) & (f[i] != 0))[s].sum())
    nw, nf, ne, nm = coadd.LowerResHSTRebin_TVM.HSTLowResRebin(w[i], f[i], e[i], m[i], obj, z)
    s2 = (nw >= lo) & (nw <= hi)
    r2 = int(((nm == 0) & (ne > 0) & np.isfinite(nf))[s2].sum())
    cont = spec_morph.cont_filtered(nw, nf, z, obj)
    r3 = int(((cont > 0) & np.isfinite(cont))[s2].sum())
    print('  row %2d: raw-good %4d -> rebinned-good %4d -> continuum>0 %4d   (rest %.0f-%.0f A)'
          % (i, r1, r2, r3, w[i][w[i] > 0].min() / (1 + z), w[i].max() / (1 + z)))
