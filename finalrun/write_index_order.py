"""Rebuild index_order.csv so the interactive fitter lists objects by index.

ica/manual_fix.py and manual_fix_gui.py READ this file to order the object list
and to label each entry with its index number, but nothing wrote it: it was made
by hand once and went stale the moment an override changed which instrument an
object is fitted on.  H1821+643 (index 127) moved from COS to FOS and then
appeared at the END of the fitter's list with no index at all, because the file
still mapped the old stem.

Run it after any override that changes the chosen instrument; fit_chosen_visit
calls it automatically.

    python write_index_order.py
"""
import csv, glob, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def write(rebin=None, queue=None, quiet=False):
    rebin = rebin or os.path.join(HERE, "pipeline_output", "chosen_visit", "rebin")
    queue = queue or os.path.join(HERE, "pipeline_output", "work_queue.csv")
    if not os.path.isdir(rebin):
        return 0
    by_name = {}
    for r in csv.DictReader(open(queue)):
        try:
            by_name[r["name_mast_key"]] = int(r["order"])
        except (KeyError, TypeError, ValueError):
            continue
    rows, unknown = [], []
    for p in sorted(glob.glob(os.path.join(rebin, "*.fits"))):
        stem = os.path.basename(p)[:-5]
        # stems are "<name_mast_key>_<INST>"; SPLICE products keep the same key
        name = stem.rsplit("_", 1)[0]
        i = by_name.get(name)
        if i is None:
            unknown.append(stem)
            continue
        rows.append((stem, i))
    rows.sort(key=lambda t: (t[1], t[0]))
    out = os.path.join(rebin, "index_order.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "index"])
        w.writerows(rows)
    if not quiet:
        print("wrote %s: %d stems" % (out, len(rows)))
        if unknown:
            print("  no queue entry for %d stem(s): %s"
                  % (len(unknown), ", ".join(unknown[:6])))
    return len(rows)


if __name__ == "__main__":
    sys.exit(0 if write() else 1)
