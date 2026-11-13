"""Before/after table for the NREAD rebuild, over every replayed object."""
import argparse, glob, json, os
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
LIVE=os.path.join(HERE,"pipeline_output","chosen_visit","fits")
def adopted(tree,i):
    for pat in ("BEST_*.json","REJECT_ALL_*.json"):
        h=glob.glob(os.path.join(tree,"%03d_*"%i,"records",pat))
        if h: return sorted(h)[0]
    return None
def vals(p):
    d=json.load(open(p)); r=d.get("result",{}) or {}; m=d.get("monochromatic",{}) or {}
    s=d.get("civ_subcontinuum",{}) or {}
    def mono(k):
        v=m.get(k) or {}
        return float(v["logL"]) if v.get("covered") and v.get("logL")==v.get("logL") else np.nan
    return dict(spectrum=d.get("name",""),blue=r.get("civ_blue",np.nan),ew=r.get("civ_ew",np.nan),
                l1450=mono("1450"),veto=s.get("n_px",np.nan),civ_px=s.get("n_civ_px",np.nan))
ap=argparse.ArgumentParser(); ap.add_argument("--snap",required=True)
ap.add_argument("--label",default="nread"); ap.add_argument("--csv",default=None)
a=ap.parse_args()
idx=[int(x) for x in open(os.path.join(a.snap,"INDICES.txt")).read().split()]
rows=[]
for i in idx:
    po=adopted(os.path.join(a.snap,"fits"),i)
    dirs=glob.glob(os.path.join(LIVE,"%03d_*"%i))
    if not dirs or po is None: continue
    cand=sorted(glob.glob(os.path.join(dirs[0],"records","iter*_%s.json"%a.label)))
    if not cand: continue
    o,n=vals(po),vals(cand[-1])
    rows.append(dict(order=i,spectrum=o["spectrum"],
        blue_before=o["blue"],blue_after=n["blue"],d_blue=n["blue"]-o["blue"],
        ew_before=o["ew"],ew_after=n["ew"],d_ew=n["ew"]-o["ew"],
        l1450_before=o["l1450"],l1450_after=n["l1450"],d_l1450=n["l1450"]-o["l1450"],
        veto_before=o["veto"],veto_after=n["veto"],
        civ_px_before=o["civ_px"],civ_px_after=n["civ_px"]))
d=pd.DataFrame(rows); d["d_quanta"]=d.d_blue/69.03
print("NREAD REBUILD -- CHANGE IN THE ADOPTED MEASUREMENTS")
print("  objects with an adopted record replayed: %d"%len(d))
print("  measurements that MOVED                : %d"%int((d.d_blue.abs()>1e-6).sum()))
print("  blueshift |delta| median %.2f quanta, 90th %.2f, max %.2f"
      %(d.d_quanta.abs().median(),d.d_quanta.abs().quantile(.9),d.d_quanta.abs().max()))
print("  EW |delta| median %.2f A, max %.2f A"%(d.d_ew.abs().median(),d.d_ew.abs().max()))
print("  log L1450 |delta| median %.3f dex, max %.3f dex"
      %(d.d_l1450.abs().median(),d.d_l1450.abs().max()))
print("  net log L1450 shift: mean %+.4f dex (%.1f standard errors from zero)"
      %(d.d_l1450.mean(), d.d_l1450.mean()/(d.d_l1450.std(ddof=1)/np.sqrt(d.d_l1450.notna().sum()))))
print()
vg=d[(d.veto_before>0)&(d.veto_after==0)]; vb=d[(d.veto_before==0)&(d.veto_after>0)]
print("  veto newly PASSES: %d    newly FAILS: %d"%(len(vg),len(vb)))
need=d[(d.d_quanta.abs()>=2)|((d.veto_before==0)&(d.veto_after>0))|((d.veto_before>0)&(d.veto_after==0))]
print()
print("*** NEEDS A LOOK (moved >=2 quanta, or veto changed state): %d of %d ***"%(len(need),len(d)))
print("  %-5s %-26s %9s %9s %8s %7s %7s %9s"%("ord","spectrum","blue_old","blue_new","d_quanta","veto_o","veto_n","dL1450"))
for _,r in need.reindex(need.d_quanta.abs().sort_values(ascending=False).index).iterrows():
    print("  %-5d %-26s %9.1f %9.1f %8.2f %7s %7s %9s"
          %(r.order,str(r.spectrum)[:26],r.blue_before,r.blue_after,r.d_quanta,
            r.veto_before,r.veto_after,("%+.3f"%r.d_l1450) if r.d_l1450==r.d_l1450 else "--"))
if a.csv: d.to_csv(a.csv,index=False); print(); print("wrote %s"%a.csv)
