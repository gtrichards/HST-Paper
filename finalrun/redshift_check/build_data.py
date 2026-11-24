import json, glob, os, sys, numpy as np, pandas as pd
S=sys.argv[1]; C=299792.458
t=pd.read_csv("/Users/gtr/Work/git/HST-Paper/Tables/table1_master_sample_v23.csv"); t=t[t.CIVgood_manmask.isin([2,3])]
q=pd.read_csv("/Users/gtr/Work/git/HST-Paper/archive/provenance/work_queue.csv"); order=dict(zip(q.row_num,q.order))
ov=pd.read_csv("/Users/gtr/Work/git/HST-Paper/ica/redshift_overrides.csv")
FIT=["/Users/gtr/Work/projects/hstica/finalrun/pipeline_output/chosen_visit/fits","/Users/gtr/Work/projects/hstica/finalrun/iue/fits"]
def best(o):
    for root in FIT:
        for d in glob.glob("%s/%03d_*"%(root,o)):
            b=sorted(glob.glob(d+"/records/BEST_*.json"))
            if b: return json.load(open(b[-1]))
keep=["Published Redshift","Published Redshift Uncertainty","Refcode","Frequency Targeted","Spectral Range","Measurement Mode Features","Measurement Mode Technique","Qualifiers","Name in publication"]
rows=[]
for _,r in t.iterrows():
    z=float(r.z); o=order.get(r.row_num)
    # The live overrides file wins over Table 1, which lags until the refit is
    # promoted; the fit-record offsets below were then measured at the old z.
    _m=ov[ov.object.astype(str).str.strip().isin([str(r.common_name).strip(), str(r.common_name).replace(" ","")])]
    z_table=z
    if len(_m) and abs(float(_m.z.iloc[0])-z)>1e-6: z=float(_m.z.iloc[0])
    rec={"row":int(r.row_num),"idx":int(o) if o is not None else None,"name":r.common_name,"des":r.designation,"z":z,"zcode":("O" if z!=z_table else r.z_code),"z_table":z_table,"zsrc":r.z_src_text if isinstance(r.z_src_text,str) else "","q":int(r.CIVgood_manmask),"sdss":int(r.sdss_flag) if r.sdss_flag==r.sdss_flag else 0,"inst":r.spectrum}
    m=ov[ov.object.astype(str).str.strip().isin([str(r.common_name).strip(), str(r.common_name).replace(" ","")])]
    if len(m): rec["override"]={"z":float(m.z.iloc[0]),"source":str(m.source.iloc[0]),"note":str(m.note.iloc[0])}
    b=best(o) if o is not None else None
    if b:
        rc=b.get("redshift_check_kms",{}) or {}
        if z!=z_table: rec["fit_note"]="measured at the previous z %.6f; refit pending"%z_table
        rec["fit"]={k:(None if rc.get(k) is None or (isinstance(rc.get(k),float) and not np.isfinite(rc.get(k))) else round(float(rc.get(k)),1)) for k in ["ciii_peak_dv","ciii_xcorr_dv","ciii_xcorr_snr","mgii_xcorr_dv","mgii_xcorr_snr"]}
    fn="%s/ned/%03d.json"%(S,r.row_num); n=json.load(open(fn)) if os.path.exists(fn) else {}
    rec["ned"]={"name":n.get("ned_name"),"how":n.get("how"),"sep":round(n["sep"],1) if n.get("sep") is not None else None,"z":n.get("ned_z")}
    if n.get("ned_z") is not None: rec["dv_ned"]=round(C*(n["ned_z"]-z)/(1+z))
    pubs=[]
    for p in n.get("ned_redshifts",[]) or []:
        try: pz=float(p.get("Published Redshift"))
        except Exception: continue
        if not np.isfinite(pz): continue
        e={k:p.get(k) for k in keep if p.get(k) not in (None,"")}; e["dv"]=round(C*(pz-z)/(1+z)); pubs.append(e)
    rec["pubs"]=pubs
    if pubs:
        dv=np.array([p["dv"] for p in pubs]); rec.update(pub_n=len(pubs),pub_lo=float(dv.min()),pub_hi=float(dv.max()),pub_med=float(np.median(dv)))
    rows.append(rec)
import math
def clean(o):
    """JSON for a browser: JSON.parse rejects NaN/Infinity, so write null."""
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    return o
json.dump(clean(rows),open(S+"/zcheck_data.json","w"),separators=(",",":"),default=str,allow_nan=False)
dv=np.array([abs(x.get("dv_ned",np.nan)) for x in rows],float)
print("objects",len(rows)," with NED z",np.isfinite(dv).sum()," |dv|>300:",(dv>300).sum()," >1000:",(dv>1000).sum(), " KB",os.path.getsize(S+"/zcheck_data.json")//1024)
