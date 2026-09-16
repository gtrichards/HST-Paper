"""
Where an object's files live, and what they are called.

Folders and final figures are prefixed with the work-queue order number, so that
after a few dozen objects the directory still sorts into the order the work was
done and a given object can be found without hunting. The order is looked up
from work_queue.csv rather than passed around, so every script names the same
folder for the same object without having to agree on anything.

    001_NGC4395_COMBINED/          iteration figures, records/ beneath
    final/plots/001_NGC 4395.png   the adopted figure

An object with no queue entry -- a scratch or trial spectrum -- gets order 000
rather than failing, so exploratory work does not need a queue row to exist.
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.join(HERE, "pipeline_output", "work_queue.csv")

_INDEX = None


def safe(s):
    return "".join(c if (c.isalnum() or c in "+.-_") else "_" for c in str(s))


def _index():
    """Map every name an object may be known by to its queue row."""
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        if os.path.exists(QUEUE):
            for r in csv.DictReader(open(QUEUE)):
                # _COMBINED is the inverse-variance merge of every spectrum;
                # _SPLICE is a hand-built join of pieces of them (first used
                # on UGC 12163: FOS for the line, STIS for the continuum).
                # Both are object-level names, not stems on disk.
                for suffix in ("COMBINED", "SPLICE"):
                    _INDEX["%s_%s" % (r["name_mast_key"], suffix)] = r
                    _INDEX[safe("%s_%s" % (r["name_mast_key"], suffix))] = r
                for st in r["stems"].split(";"):
                    if st:
                        _INDEX[st] = r
                        _INDEX[safe(st)] = r
    return _INDEX


def queue_row(name):
    return _index().get(name) or _index().get(safe(name))


def order_for(name):
    r = queue_row(name)
    return int(r["order"]) if r else 0


def folder_for(name):
    """Iteration folder for a spectrum stem or <object>_COMBINED name."""
    return "%03d_%s" % (order_for(name), safe(name))


def pub_name(name):
    r = queue_row(name)
    return r["name_pub"] if r else name
