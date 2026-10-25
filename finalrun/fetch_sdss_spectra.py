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


#: Nominal wavelength limits, used only when the file is not on disk.  The
#: legacy spectrograph runs 3800-9100 A and BOSS 3600-10400; the original code
#: applied the legacy pair to every spectrum, which under-reported BOSS coverage
#: by about 650 A in the rest frame and made rest_max wrong by more than 50 A for
#: 297 of 315 files.  That mattered twice over: it hid [O III] 5007 on sixteen
#: objects above z 0.84 whose BOSS spectra do reach it, and it marked
#: covers_2500 False for twenty-two objects at z 0.45-0.52 whose BOSS spectra
#: reach 2500 A at the blue end -- index 317 among them, one of the rows flagged
#: as short of a luminosity.
NOMINAL = {"legacy": (3800.0, 9100.0), "boss": (3600.0, 10400.0)}


def coverage(path, z, run2d=None):
    """(rest_lo, rest_hi, how) for one spectrum, MEASURED from the file.

    Falls back to the nominal limits for the spectrograph when the file is not
    on disk, and says which it used, so a manifest row can never silently claim
    a coverage that was assumed rather than observed.
    """
    import numpy as np
    if path and os.path.exists(path):
        try:
            from astropy.io import fits
            with fits.open(path) as h:
                d = h[1].data
                lam = 10.0 ** np.asarray(d["loglam"], float)
                iv = np.asarray(d["ivar"], float)
            good = lam[np.isfinite(lam) & np.isfinite(iv) & (iv > 0)]
            if good.size:
                return (float(good.min() / (1 + z)), float(good.max() / (1 + z)), "file")
        except Exception:
            pass
    #: run2d 26 and 103/104 are legacy reductions; v5_* and integers >= 100 that
    #: are not 103/104 are BOSS.  When in doubt assume legacy, the narrower of
    #: the two, so coverage is under-claimed rather than over-claimed.
    key = "boss" if str(run2d).strip().startswith("v5") else "legacy"
    lo, hi = NOMINAL[key]
    return (lo / (1 + z), hi / (1 + z), "nominal(%s)" % key)


def recompute(write=True, quiet=False):
    """Recompute the coverage columns of an existing manifest from the files.

    Separate from the fetch so the table can be corrected without re-resolving
    or re-downloading 315 spectra.  Adds coverage_from and covers_oiii to rows
    written before those existed.
    """
    if not os.path.exists(MAN):
        if not quiet:
            print("no manifest at %s" % MAN)
        return 0
    rows = list(csv.DictReader(open(MAN)))
    changed = 0
    for r in rows:
        try:
            z = float(r.get("z_queue"))
        except (TypeError, ValueError):
            continue
        fn = str(r.get("fn_sdss") or "").strip()
        lo, hi, how = coverage(os.path.join(OUT, fn) if fn else None, z, r.get("run2d"))
        new = dict(rest_min="%.1f" % round(lo, 1), rest_max="%.1f" % round(hi, 1),
                   coverage_from=how,
                   covers_2500=str(bool(lo <= 2490 and hi >= 2510)),
                   covers_mgii=str(bool(lo <= 2780 and hi >= 2820)),
                   covers_oiii=str(bool(lo <= 4940 and hi >= 5010)))
        if any(str(r.get(k, "")) != v for k, v in new.items()):
            changed += 1
        r.update(new)
    if write and rows:
        cols = list(rows[0].keys())
        for c in ("coverage_from", "covers_oiii"):
            if c not in cols:
                cols.append(c)
        with open(MAN, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
    if not quiet:
        def _t(v):
            return v is True or str(v).strip().lower() == "true"
        print("recomputed coverage for %d rows; %d changed" % (len(rows), changed))
        print("   measured from the file : %d" % sum(1 for r in rows
                                                     if r.get("coverage_from") == "file"))
        print("   covering Mg II 2800 A  : %d" % sum(1 for r in rows if _t(r["covers_mgii"])))
        print("   covering 2500 A        : %d" % sum(1 for r in rows if _t(r["covers_2500"])))
        print("   covering [O III] 5007  : %d" % sum(1 for r in rows if _t(r["covers_oiii"])))
    return changed


def main():
    args = [a for a in sys.argv[1:]]
    if args and args[0] == "--recompute-coverage":
        return 0 if recompute() is not None else 1
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
        _fn = "%04d/spec-%04d-%05d-%04d.fits" % (info["plate"], info["plate"],
                                                 info["mjd"], info["fiber"])
        rest_lo, rest_hi, cov_how = coverage(os.path.join(OUT, _fn), zq,
                                             info.get("run2d"))
        rows.append(dict(index=i, row_num=q.get("row_num"), name_pub=q.get("name_pub"),
                         designation=q.get("designation"),
                         plate=info["plate"], mjd=info["mjd"], fiber=info["fiber"],
                         run2d=info.get("run2d"), sdss_class=info.get("cls"),
                         z_queue=zq, z_sdss=info["z"],
                         dv_kms=round((info["z"] - zq) / (1 + zq) * 299792.458, 1),
                         fn_sdss="%04d/spec-%04d-%05d-%04d.fits"
                                 % (info["plate"], info["plate"], info["mjd"], info["fiber"]),
                         rest_min=round(rest_lo, 1), rest_max=round(rest_hi, 1),
                         coverage_from=cov_how,
                         #: Both ends, not just the blue one.  The old test asked
                         #: only whether the blue cutoff was blueward of the line,
                         #: so an object whose spectrum ended short of 2500 A in
                         #: the red was still marked as covering it.
                         covers_2500=bool(rest_lo <= 2490 and rest_hi >= 2510),
                         covers_mgii=bool(rest_lo <= 2780 and rest_hi >= 2820),
                         covers_oiii=bool(rest_lo <= 4940 and rest_hi >= 5010),
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
