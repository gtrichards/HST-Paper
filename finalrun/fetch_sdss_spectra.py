"""Resolve and download the SDSS spectrum for every object SDSS can actually help.

sdss_flag in the work queue is not a boolean: 1 means an SDSS spectrum exists,
2 means it reaches into the ICA range.  The split is at z ~ 0.36, which is where
SDSS's 3800 A blue cutoff brings Mg II 2800 A into the band -- below that the
spectrum is entirely redward of the 1260-3000 A basis and contributes nothing to
the fit.  GTR's instruction is nonetheless to hold every spectrum that exists -- "I
just want the SDSS spectra on disk if there is one. Then if they are
useful, we will have them" -- so both flags are fetched by default, and
--flag=2 or --flag=1 selects one of them.

The master catalogue carries no plate/MJD/fiber -- run_rebin.py says so in its
own docstring, which is why coadd.rebin has always been called with
fn_sdss=None -- so each object is resolved by a cone search on its designation
before the spectrum is fetched.

    python fetch_sdss_spectra.py              # all flag-2 objects
    python fetch_sdss_spectra.py 286 300      # a range

Writes data_v23/SDSS_spec/lite/<plate>/spec-<plate>-<mjd>-<fiber>.fits and a
manifest at pipeline_output/final/sdss_spectra.csv giving, per index, the
identifiers, the SDSS redshift, and the rest range the spectrum covers.
"""
import csv, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.join(HERE, "pipeline_output", "work_queue.csv")
OUT = os.path.join(HERE, "data_v23", "SDSS_spec", "lite")
MAN = os.path.join(HERE, "pipeline_output", "final", "sdss_spectra.csv")
SQL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
SAS = "https://dr18.sdss.org/sas/dr18/spectro/sdss/redux"


def radec(desig):
    """'J145108.70+270927.0' -> degrees."""
    m = re.match(r"J(\d{2})(\d{2})(\d{2}\.?\d*)([+-])(\d{2})(\d{2})(\d{2}\.?\d*)",
                 str(desig).strip())
    if not m:
        return None
    hh, mm, ss, sgn, dd, dm, ds = m.groups()
    ra = (int(hh) + int(mm) / 60.0 + float(ss) / 3600.0) * 15.0
    dec = int(dd) + int(dm) / 60.0 + float(ds) / 3600.0
    return ra, (-dec if sgn == "-" else dec)


def curl(args, timeout=120):
    r = subprocess.run(["curl", "-sL", "--max-time", str(timeout)] + args,
                       capture_output=True, text=False)
    return r.returncode, r.stdout


def resolve(ra, dec, radius=0.5):
    q = ("SELECT TOP 1 s.plate, s.mjd, s.fiberid, s.z, s.class, s.run2d "
         "FROM SpecObj s JOIN dbo.fGetNearbySpecObjEq(%f,%f,%f) n "
         "ON s.specObjID=n.specObjID ORDER BY n.distance" % (ra, dec, radius))
    rc, out = curl(["-G", SQL, "--data-urlencode", "cmd=" + q,
                    "--data-urlencode", "format=csv"], timeout=90)
    if rc != 0:
        return None
    lines = [l for l in out.decode("utf-8", "replace").splitlines()
             if l and not l.startswith("#")]
    if len(lines) < 2:
        return None
    hdr = [h.strip() for h in lines[0].split(",")]
    val = [v.strip() for v in lines[1].split(",")]
    d = dict(zip(hdr, val))
    try:
        return dict(plate=int(d["plate"]), mjd=int(d["mjd"]),
                    fiber=int(d["fiberid"]), z=float(d["z"]),
                    cls=d.get("class", ""), run2d=d.get("run2d", "26"))
    except (KeyError, ValueError):
        return None


def fetch(info):
    sub = "%04d" % info["plate"]
    fn = "spec-%04d-%05d-%04d.fits" % (info["plate"], info["mjd"], info["fiber"])
    dest_dir = os.path.join(OUT, sub)
    dest = os.path.join(dest_dir, fn)
    if os.path.exists(dest) and os.path.getsize(dest) > 20000:
        return dest, "cached"
    os.makedirs(dest_dir, exist_ok=True)
    for r2 in (str(info.get("run2d") or "26"), "26", "v5_13_2"):
        url = "%s/%s/spectra/lite/%s/%s" % (SAS, r2, sub, fn)
        rc, body = curl(["-o", dest, url], timeout=180)
        if rc == 0 and os.path.exists(dest) and os.path.getsize(dest) > 20000:
            return dest, "ok(run2d=%s)" % r2
    if os.path.exists(dest):
        os.remove(dest)
    return None, "download failed"


def main():
    args = [a for a in sys.argv[1:]]
    flags = ("1", "2")
    if args and args[0].startswith("--flag"):
        flags = tuple(args[0].split("=", 1)[1].split(","))
        args = args[1:]
    lo, hi = (int(args[0]), int(args[1])) if len(args) > 1 else (1, 10 ** 9)
    queue = {int(r["order"]): r for r in csv.DictReader(open(QUEUE))}
    todo = [i for i in sorted(queue)
            if str(queue[i].get("sdss_flag")) in flags and lo <= i <= hi]
    if not todo:
        print("nothing to do"); return
    print("fetching SDSS spectra for %d objects (sdss_flag in %s): %d-%d"
          % (len(todo), ",".join(flags), todo[0], todo[-1]), flush=True)
    rows, bad = [], []
    for n, i in enumerate(todo, 1):
        q = queue[i]
        rd = radec(q.get("designation"))
        if not rd:
            bad.append((i, "unparsable designation %r" % q.get("designation")))
            print("%3d/%d  %03d  BAD DESIGNATION %s" % (n, len(todo), i, q.get("designation")), flush=True)
            continue
        info = resolve(*rd)
        if not info:
            bad.append((i, "no SDSS spectrum within 0.5 arcmin"))
            print("%3d/%d  %03d  %-28s no match" % (n, len(todo), i, str(q.get("name_pub"))[:28]), flush=True)
            time.sleep(0.4)
            continue
        path, how = fetch(info)
        zq = float(q["z"]) if q.get("z") else float("nan")
        rest_lo, rest_hi = 3800.0 / (1 + zq), 9100.0 / (1 + zq)
        rows.append(dict(index=i, row_num=q.get("row_num"), name_pub=q.get("name_pub"),
                         designation=q.get("designation"),
                         plate=info["plate"], mjd=info["mjd"], fiber=info["fiber"],
                         run2d=info.get("run2d"), sdss_class=info.get("cls"),
                         z_queue=zq, z_sdss=info["z"],
                         dv_kms=round((info["z"] - zq) / (1 + zq) * 299792.458, 1),
                         fn_sdss="%04d/spec-%04d-%05d-%04d.fits"
                                 % (info["plate"], info["plate"], info["mjd"], info["fiber"]),
                         rest_min=round(rest_lo, 1), rest_max=round(rest_hi, 1),
                         covers_2500=bool(rest_lo <= 2500), covers_mgii=bool(rest_lo <= 2800),
                         status=how))
        if path is None:
            bad.append((i, "download failed for %s" % rows[-1]["fn_sdss"]))
        print("%3d/%d  %03d  %-26s plate %4d mjd %5d fiber %4d  z %.4f (dv %+6.0f)  rest %5.0f-%5.0f  %s"
              % (n, len(todo), i, str(q.get("name_pub"))[:26], info["plate"], info["mjd"],
                 info["fiber"], info["z"], rows[-1]["dv_kms"], rest_lo, rest_hi, how), flush=True)
        time.sleep(0.4)
    if rows:
        os.makedirs(os.path.dirname(MAN), exist_ok=True)
        # Merge with any existing manifest rather than replacing it: the flag-2
        # objects were fetched first and a later flag-1 run must not erase them.
        prev = {}
        if os.path.exists(MAN):
            for r in csv.DictReader(open(MAN)):
                try:
                    prev[int(r["index"])] = r
                except (KeyError, ValueError):
                    pass
        for r in rows:
            prev[int(r["index"])] = r
        allrows = [prev[k] for k in sorted(prev)]
        with open(MAN, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(allrows)
        rows = allrows
        # Rows read back from the CSV are strings, so "False" is truthy -- count
        # on the value, not its truthiness.
        def _t(v):
            return v is True or str(v).strip().lower() == "true"
        print("\n%d spectra -> %s" % (len(rows), MAN))
        print("   covering Mg II 2800 A: %d" % sum(1 for r in rows if _t(r["covers_mgii"])))
        print("   covering 2500 A:       %d" % sum(1 for r in rows if _t(r["covers_2500"])))
    if bad:
        print("\n%d problems:" % len(bad))
        for i, w in bad[:20]:
            print("   %03d  %s" % (i, w))


if __name__ == "__main__":
    sys.exit(main())
