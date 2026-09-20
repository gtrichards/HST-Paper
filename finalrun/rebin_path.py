"""
rebin_path.py -- the one place the rebinned-spectrum directory is named.

Every fitting and diagnostic script in finalrun/ reads its spectra from the
same directory, and each used to hardcode it.  That made switching the whole
pass to a different rebin tree a nine-file edit, with the obvious risk that one
script is left pointing at the old one and quietly measures different data from
its neighbours.

Default is the rebinned master the students produced, which is read-only as far
as this project is concerned.  Override it for a whole session with

    export HSTICA_REBIN=/path/to/RebinnedSpec_v23

so that a regenerated tree can be adopted, or reverted, in one place.  Scripts
that take an explicit path on the command line still win over both.
"""

import os

MASTER = "/Users/gtr/Dropbox/HST/Pratsos/RebinnedSpec_master"

REBIN = os.environ.get("HSTICA_REBIN", MASTER)

#: True when the directory in use is not the master, so a script can say so.
IS_OVERRIDDEN = REBIN != MASTER


def describe():
    """One line naming the directory in use, for a run's log header."""
    return "rebin directory: %s%s" % (
        REBIN, "   (HSTICA_REBIN override)" if IS_OVERRIDDEN else "   (master)")
