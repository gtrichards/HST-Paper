"""Is what is on disk what was built?  Re-check every file against MANIFEST.csv.

The point of a checksum manifest is to make "did anything change between
versions, and was it intended" answerable rather than a matter of memory.  It is
the same instinct as the checksum row GTR asked for in Table 1, applied to the
whole deposit, and this script is the reader that makes it useful.

    python verify_archive.py                 # check every file
    python verify_archive.py --quick         # size only, no hashing
    python verify_archive.py --against OLD.csv   # diff two manifests

EXIT STATUS is 0 when the archive matches and 1 when it does not, so this can be
run before a deposit is uploaded.
"""
import argparse
import csv
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "MANIFEST.csv")


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load(p):
    return {r["path"]: r for r in csv.DictReader(open(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--against", default=None, metavar="MANIFEST.csv")
    a = ap.parse_args()

    if not os.path.exists(MAN):
        sys.exit("no MANIFEST.csv -- run build_archive.py")
    man = load(MAN)

    if a.against:
        old = load(a.against)
        added = sorted(set(man) - set(old))
        gone = sorted(set(old) - set(man))
        changed = sorted(p for p in set(man) & set(old)
                         if man[p]["sha256"] != old[p]["sha256"])
        print("MANIFEST DIFF against %s" % a.against)
        print("  unchanged : %d" % len(set(man) & set(old) - set(changed)))
        print("  changed   : %d" % len(changed))
        print("  added     : %d" % len(added))
        print("  removed   : %d" % len(gone))
        for tag, lst in (("changed", changed), ("added", added), ("removed", gone)):
            for p in lst[:40]:
                print("    %-9s %s" % (tag, p))
            if len(lst) > 40:
                print("    %-9s ... and %d more" % (tag, len(lst) - 40))
        return 0

    missing, wrong_size, wrong_hash, ok = [], [], [], 0
    for rel, r in sorted(man.items()):
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            missing.append(rel)
            continue
        if os.path.getsize(p) != int(r["bytes"]):
            wrong_size.append(rel)
            continue
        if not a.quick and sha256(p) != r["sha256"]:
            wrong_hash.append(rel)
            continue
        ok += 1

    # files on disk that the manifest does not know about
    extra = []
    for dirpath, _, files in os.walk(HERE):
        for f in files:
            rel = os.path.relpath(os.path.join(dirpath, f), HERE)
            if rel == "MANIFEST.csv" or rel.startswith(".git"):
                continue
            if rel not in man:
                extra.append(rel)

    print("ARCHIVE VERIFICATION%s" % ("  (sizes only)" if a.quick else ""))
    print("  manifest entries : %d" % len(man))
    print("  verified         : %d" % ok)
    print("  MISSING          : %d" % len(missing))
    print("  WRONG SIZE       : %d" % len(wrong_size))
    if not a.quick:
        print("  WRONG CHECKSUM   : %d" % len(wrong_hash))
    print("  not in manifest  : %d" % len(extra))
    for tag, lst in (("MISSING", missing), ("SIZE", wrong_size),
                     ("CHECKSUM", wrong_hash), ("EXTRA", extra)):
        for p in lst[:20]:
            print("    %-9s %s" % (tag, p))
        if len(lst) > 20:
            print("    %-9s ... and %d more" % (tag, len(lst) - 20))
    bad = missing or wrong_size or wrong_hash
    print()
    print("ARCHIVE %s" % ("DOES NOT MATCH ITS MANIFEST" if bad else "matches its manifest"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
