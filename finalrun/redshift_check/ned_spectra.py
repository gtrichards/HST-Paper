"""For each measured object without an SDSS spectrum: list NED's archived
spectra, keep the ones whose wavelength range reaches the optical, download
the FITS, and record the bibliographic source.  One JSON per object."""
import json, os, sys, time, urllib.request, warnings
warnings.filterwarnings("ignore")
from astroquery.ipac.ned import Ned
S = sys.argv[1]; OUT = os.path.join(S, "optical"); os.makedirs(OUT + "/fits", exist_ok=True)
rows = json.load(open(S + "/zcheck/zcheck_data.json"))
spec = json.load(open(S + "/zcheck/zcheck_spectra.json"))
todo = [r for r in rows if not any(p["src"] == "SDSS" for p in spec[str(r["row"])]["panels"])]
print("objects without SDSS:", len(todo), flush=True)
for r in todo:
    fn = os.path.join(OUT, "%03d.json" % r["row"])
    if os.path.exists(fn): continue
    nm = (r.get("ned") or {}).get("name") or r["name"]
    rec = {"row": r["row"], "name": r["name"], "ned_name": nm, "spectra": []}
    for k in range(3):
        try:
            urls = Ned.get_image_list(nm, item="spectra")
            rec["n_listed"] = len(urls)
            for i, u in enumerate(urls):
                dest = os.path.join(OUT, "fits", "%03d_%02d.fits" % (r["row"], i))
                if not os.path.exists(dest):
                    urllib.request.urlretrieve(u, dest)
                rec["spectra"].append({"url": u, "file": os.path.basename(dest)})
            break
        except Exception as ex:
            rec["error"] = str(ex)[:150]; time.sleep(4)
    json.dump(rec, open(fn, "w"))
    print(r["row"], r["name"], "listed", rec.get("n_listed"), rec.get("error", "")[:60], flush=True)
print("done")
