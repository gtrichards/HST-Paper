import json, os, sys, time, urllib.request
S=sys.argv[1]; F=S+"/files"; man=[]
def get(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest)>1000: return True
    for k in range(3):
        try:
            urllib.request.urlretrieve(url, dest); return os.path.getsize(dest)>1000
        except Exception as e: time.sleep(3)
    return False
A=json.load(open(S+"/archives.json"))
for x in A:
    for l in x['lamost']:
        oid=l.get('catalogue_obsid'); dr=x.get('lamost_dr','dr9')
        dest="%s/%03d_lamost_%s.fits.gz"%(F,x['row'],oid)
        ok=get("https://www.lamost.org/%s/v2.0/spectrum/fits/%s"%(dr,oid), dest)
        man.append({"row":x['row'],"src":"LAMOST "+dr.upper(),"file":os.path.basename(dest),"ok":ok,"meta":l})
    es=sorted(x['eso'], key=lambda e:-e['t_exptime'])[:3]
    for e in es:
        dest="%s/%03d_eso_%s.fits"%(F,x['row'],e['dp_id'].replace(":","_"))
        ok=get("https://dataportal.eso.org/dataportal_new/file/"+e['dp_id'], dest)
        man.append({"row":x['row'],"src":"ESO "+e['instrument_name'],"file":os.path.basename(dest),"ok":ok,"meta":e})
for x in json.load(open(S+"/sdss_requery.json")):
    r=x['sdss'][0]; plate=int(r['plate']); run2d=r['run2d']
    sub="sdss" if plate<3510 else "eboss"
    url="https://data.sdss.org/sas/dr18/spectro/%s/redux/%s/spectra/lite/%04d/spec-%04d-%s-%04d.fits"%(sub,run2d,plate,plate,r['mjd'],int(r['fiberid']))
    dest="%s/%03d_sdss_spec-%04d-%s-%04d.fits"%(F,x['row'],plate,r['mjd'],int(r['fiberid']))
    ok=get(url,dest); man.append({"row":x['row'],"src":"SDSS DR18","file":os.path.basename(dest),"ok":ok,"meta":r,"url":url})
json.dump(man,open(S+"/downloads.json","w"))
print("downloaded ok:",sum(m['ok'] for m in man)," failed:",[ (m['row'],m['src']) for m in man if not m['ok']])
