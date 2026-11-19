"""E(B-V) for every Table 1 object from IRSA's dust service (Schlafly &
Finkbeiner 2011 recalibration of SFD).  One cached JSON per object."""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ipac.irsa.irsa_dust import IrsaDust
# Usage: python fetch_ebv.py CACHE_DIR, then collect the JSON files into
# Data/EBV_SandF_v23.csv (row_num, ebv_sandf, ebv_sfd).  Needs astroquery.
OUT=sys.argv[1]; os.makedirs(OUT, exist_ok=True)
t=pd.read_csv("/Users/gtr/Work/git/HST-Paper/Data/CIV_measurements_v23.csv")
def coord(des):
    s=des[1:]; i=max(s.find("+"),s.find("-")); ra,dec=s[:i],s[i:]
    return SkyCoord("%s:%s:%s %s:%s:%s"%(ra[0:2],ra[2:4],ra[4:],dec[0:3],dec[3:5],dec[5:]),unit=(u.hourangle,u.deg))
for _,r in t.iterrows():
    fn=os.path.join(OUT,"%03d.json"%r.row_num)
    if os.path.exists(fn): continue
    for k in range(4):
        try:
            q=IrsaDust.get_query_table(coord(r.designation),section="ebv")
            rec={"row_num":int(r.row_num),"ebv_sandf":float(q["ext SandF mean"][0]),"ebv_sfd":float(q["ext SFD mean"][0])}
            json.dump(rec,open(fn,"w")); print(r.row_num,rec["ebv_sandf"],flush=True); break
        except Exception as ex:
            print("retry",r.row_num,str(ex)[:80],flush=True); time.sleep(5)
print("done")
