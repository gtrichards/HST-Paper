"""
rebin_path.py -- the one place the rebinned-spectrum directory is named.

Every fitting and diagnostic script in finalrun/ reads its spectra from the
same directory, and each used to hardcode it.  That made switching the whole
pass to a different rebin tree a nine-file edit, with the obvious risk that one
script is left pointing at the old one and quietly measures different data from
its neighbours.

Default is the regenerated tree.  The master the students produced is still
named here as MASTER, read-only, as the reference the regenerated tree was
checked against.  Override the choice for a whole session with

    export HSTICA_REBIN=/path/to/some/other/tree

so that the working tree can be changed, or reverted to MASTER, in one place.
Scripts that take an explicit path on the command line still win over both.
"""

import os

#: The students' rebinned spectra. Kept as the reference to compare against and
#: never written to; superseded as the working tree on 2026-09-21.
MASTER = "/Users/gtr/Dropbox/HST/Pratsos/RebinnedSpec_master"

#: The regenerated tree: every object retrieved from MAST by
#: pipeline/retrieve_spectra.py and rebinned by pipeline/rebin_v23.py, 647
#: spectra against the master's 576. It reproduces all 576 of the master's
#: files -- 183 bit-identical, 161 agreeing to 0.1% -- and where it differs the
#: cause is archive recalibration since the master was built, or a master file
#: that was truncated. Adopted as the default by GTR on 2026-09-21.
V23 = "/Users/gtr/Work/projects/hstica/finalrun/data_v23/RebinnedSpec_v23"

DEFAULT = V23

REBIN = os.environ.get("HSTICA_REBIN", DEFAULT)

#: True when the directory in use is not the default one.
IS_OVERRIDDEN = REBIN != DEFAULT


def describe():
    """One line naming the directory in use, for a run's log header."""
    if IS_OVERRIDDEN:
        tag = "   (HSTICA_REBIN override)"
    elif REBIN == V23:
        tag = "   (regenerated v23 tree)"
    else:
        tag = ""
    return "rebin directory: %s%s" % (REBIN, tag)


#: Where per-object fit iterations are written.  Overridable so a fresh first
#: pass can be run without colliding with records already adopted: run_batch.py
#: adds iterations to an object's folder and renames its pick to BEST_, so a
#: second pass into the same folder would leave two BEST_ records and make the
#: adopted fit ambiguous.
#:
#:     export HSTICA_ITERDIR=/path/to/pipeline_output/fit_iterations_v23
#:
_HERE = os.path.dirname(os.path.abspath(__file__))
ITERDIR = os.environ.get("HSTICA_ITERDIR",
                         os.path.join(_HERE, "pipeline_output", "fit_iterations"))
ITERDIR_IS_OVERRIDDEN = ITERDIR != os.path.join(_HERE, "pipeline_output", "fit_iterations")
