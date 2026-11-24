import json, sys, io, time, urllib.request, warnings; warnings.filterwarnings("ignore")
from astropy.io.votable import parse_single_table
from astropy.coordinates import SkyCoord; import astropy.units as u
import pyvo
S=sys.argv[1]; T=json.load(open(S+"/optical2/targets.json"))
eso=pyvo.dal.TAPService("https://archive.eso.org/tap_obs")
def coord(des):
    s=des[1:]; i=max(s.find("+"),s.find("-")); ra,dec=s[:i],s[i:]
    return SkyCoord("%s:%s:%s %s:%s:%s"%(ra[0:2],ra[2:4],ra[4:],dec[0:3],dec[3:5],dec[5:]),unit=(u.hourangle,u.deg))
out=[]
for t in T:
    c=coord(t['des']); rec={"row":t['row'],"name":t['name'],"z":t['z'],"lamost":[],"eso":[]}
    for dr in ("dr9","dr10"):
        try:
            x=urllib.request.urlopen("https://www.lamost.org/%s/v2.0/voservice/conesearch?ra=%.6f&dec=%.6f&sr=0.0014"%(dr,c.ra.deg,c.dec.deg),timeout=60).read()
            tb=parse_single_table(io.BytesIO(x)); a=tb.array
            names=[f.name for f in tb.fields]
            for row in a: rec["lamost"].append({n:str(row[i]) for i,n in enumerate(names) if n in ("catalogue_obsid","catalogue_class","catalogue_z","catalogue_snrg","catalogue_snrr","catalogue_ra","catalogue_dec")})
            if rec["lamost"]: rec["lamost_dr"]=dr; break
        except Exception as e: rec["lamost_err"]=str(e)[:80]
    lines=[3728.48,4862.68,5008.24]
    try:
        q=("select obs_collection, instrument_name, em_min*1e10 as lo, em_max*1e10 as hi, em_res_power, t_exptime, target_name, access_url, dp_id "
           "from ivoa.ObsCore where dataproduct_type='spectrum' and intersects(s_region, circle('', %.6f, %.6f, 0.0014))=1")%(c.ra.deg,c.dec.deg)
        r=eso.search(q).to_table()
        for x in r:
            cov=[l for l in lines if x['lo'] < l*(1+t['z'])-30 < l*(1+t['z'])+30 < x['hi']]
            if cov: rec["eso"].append({k:(str(x[k]) if k in ('obs_collection','instrument_name','target_name','access_url','dp_id') else float(x[k])) for k in ('obs_collection','instrument_name','lo','hi','em_res_power','t_exptime','target_name','access_url','dp_id')})
    except Exception as e: rec["eso_err"]=str(e)[:80]
    out.append(rec); print(t['row'],t['name'],"LAMOST",len(rec["lamost"]),"ESO",len(rec["eso"]),flush=True)
json.dump(out,open(S+"/optical2/archives.json","w"))
print("LAMOST hits:",sum(bool(r['lamost']) for r in out)," ESO hits:",sum(bool(r['eso']) for r in out))
