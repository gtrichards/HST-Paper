"""Mrk 1502: the co-add under the recommended rule, against the adopted product."""
import glob, json, os, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, "/Users/gtr/Work/git/richardsplot"); import richardsplot  # noqa
import matplotlib.pyplot as plt
from astropy.table import Table
HERE="/Users/gtr/Work/projects/hstica/finalrun"
S="/private/tmp/claude-502/-Users-gtr-Work-projects-hstica/d417042d-a1d6-4929-8903-4ecb7e5424bd/scratchpad/handbook"
OUT=HERE+"/pipeline_output/final/talk/talk_D_Mrk1502_recommended_coadd.png"
def load(p):
    t=Table.read(p); w=np.asarray(t["Rest-frame Wavelength"],float); f=np.asarray(t["Coadded Flux (Arbitrary Units)"],float)
    e=np.asarray(t["Coadded Flux Errors"],float); m=np.asarray(t["Bad Pixel Mask"],float)
    g=(m==0)&np.isfinite(f)&np.isfinite(e)&(e>0); return w,f,e,g
A=load(HERE+"/pipeline_output/chosen_visit/rebin/Mrk 1502_FOS.fits"); B=load(S+"/rebin/Mrk 1502_FOS.fits")
fits_={}
for p in glob.glob(S+"/fits/*/records/*.json"):
    d=json.load(open(p)); fits_[d["label"]]=d["result"]
fig,axs=plt.subplots(2,2,figsize=(14,8.4),gridspec_kw=dict(width_ratios=[1.35,1],height_ratios=[1.6,1]))
(axF,axZ),(axE,axEz)=axs
def panel(ax,lo,hi,zoom=False):
    for (w,f,e,g),c,lw,lab,zo in ((A,"0.65",2.6,"adopted product (code at a99e8ff)",1),(B,"k",0.9,"recommended: + Handbook-range truncation (G190H on the red Digicon from 1590 \\AA\\ observed)",3)):
        k=g&(w>=lo)&(w<=hi); ax.plot(w[k],f[k],color=c,lw=lw,label=lab,zorder=zo)
        if zoom: ax.fill_between(w[k],f[k]-e[k],f[k]+e[k],color=c,alpha=0.15,lw=0,zorder=zo-0.5)
    ax.axvspan(1500,1600,color="#e8eef7",zorder=0); ax.axvline(1549.48,color="0.6",lw=0.5,ls=":"); ax.set_xlim(lo,hi)
panel(axF,1420,1720); panel(axZ,1490,1540,zoom=True)
axF.legend(fontsize=8,frameon=False,loc="lower left"); axF.set_ylabel(r"co-added $F_\lambda$ (arbitrary)")
axF.set_title("Mrk 1502 (I Zw 1), index 70: what the co-add should look like",fontsize=11)
axZ.set_title(r"zoom: 1490--1540 \AA, with the co-added $1\sigma$ band",fontsize=10)
# the hole (filled) and the remaining spike
ka=A[3]&(A[0]>1497.5)&(A[0]<1500.2); kb=B[3]&(B[0]>1497.5)&(B[0]<1500.2)
axZ.annotate("1498--1500 \\AA: %d px in the adopted product,\n%d px now -- G130H fills them once the\nred-Digicon G190H diodes beyond 1590 \\AA\\ are dropped"%(ka.sum(),kb.sum()),
             xy=(1499.0,float(np.nanmedian(B[1][kb])) if kb.any() else 3.3),xytext=(1491.0,1.95),fontsize=7.5,arrowprops=dict(arrowstyle="->",lw=0.8))
ks=B[3]&(B[0]>1514.3)&(B[0]<1515.0)
if ks.any():
    axZ.annotate("1514.6 \\AA\\ remains: first G190H-only pixel,\ninside the quoted range (1605 \\AA\\ obs) but in\nfootnote g's zone; S/N %.1f, carried honestly"%(float(B[1][ks][0]/B[2][ks][0])),
                 xy=(float(B[0][ks][0]),float(B[1][ks][0])),xytext=(1519.0,2.3),fontsize=7.5,color="firebrick",arrowprops=dict(arrowstyle="->",lw=0.8,color="firebrick"))
r0=fits_.get("handbook",{}); r1=fits_.get("handbook_mod",{})
axF.text(0.985,0.96,"fits on this product (your store masks)\n default prior: %+.1f km/s, EW %.2f \\AA\n mod prior:     %+.1f km/s, EW %.2f \\AA\n(adopted product: %+.1f / %.2f ; %+.1f / %.2f)\ndata half-flux centroid: %+.1f"%(
    r0.get("civ_blue",np.nan),r0.get("civ_ew",np.nan),r1.get("civ_blue",np.nan),r1.get("civ_ew",np.nan),1506.2,26.64,1162.4,26.35,1440.5),
    transform=axF.transAxes,ha="right",va="top",fontsize=8,family="monospace",bbox=dict(fc="white",ec="0.7",pad=3))
for ax,(lo,hi) in ((axE,(1420,1720)),(axEz,(1490,1540))):
    for (w,f,e,g),c,lw in ((A,"0.65",2.2),(B,"k",0.9)):
        k=g&(w>=lo)&(w<=hi); ax.plot(w[k],np.abs(f[k])/e[k],color=c,lw=lw)
    ax.set_yscale("log"); ax.axvspan(1500,1600,color="#e8eef7",zorder=0); ax.set_xlim(lo,hi); ax.set_xlabel(r"Rest-frame Wavelength (\AA)")
axE.set_ylabel("co-added S/N per pixel")
fig.tight_layout(); fig.savefig(OUT,dpi=150); print("wrote",OUT)
