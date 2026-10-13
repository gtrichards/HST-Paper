"""
rebinning/spec_cuts_cos.py
Migrated from: Trevor Code/SpecCuts_COS.py
Migration date: 2026-04-30
No algorithmic changes — only header added.

Note: Included because cut_edge_pix.py has a COS code path. No COS objects
are in the 2026AP run, but the module is needed for the import to succeed.
"""

BlueEdges = {
    "B21425+26 - o65637010": 9999
}

#: Rest-frame red cutoffs, applied as RedEdges[exp_id] * (1+z) in
#: cut_edge_pix.Cut_Edge_Pix.  The key is the exp_id that reader builds,
#: "<object name> - <exposure base>.<segment row>".
#:
#: Why these are needed at all: spec_morph.cont_filtered normalises each
#: exposure by a continuum that is interpolated LINEARLY across the C IV
#: emission-line exclusion window, 1460-1685 A rest (emission_lines2).  If an
#: exposure's usable range ends inside that window, the red end of the
#: interpolation is anchored on dead detector, the continuum ramps to zero
#: across C IV and goes negative past 1685 A, and dividing by it sends the
#: normalised exposure to +/-1e12.  coadd.rebin then masks the lot, so the
#: rebinned product carries ZERO usable C IV pixels and the ICA fit reports a
#: blueshift extrapolated from the components with no data behind it.
#:
#: 2MASS J08434975+2619106 (index 254, z=0.2564) is the worked case: COS G140L
#: throughput collapses from about 2000 A observed = 1602 A rest, inside the
#: exclusion window, and the rebinned product had 0 of 280 C IV pixels while
#: both raw exposures show a clean C IV line rising from 2.5e-15 to 9e-15.
#: Trimming at 1602 A rest (2013 A observed) stops the interpolation reaching
#: the dead region and recovers 280 pixels at S/N 3.21.  The result is
#: insensitive to the exact cut: every trim from 1595 to 1650 A rest returns
#: the same blueshift to the last decimal.
RedEdges = {
    "2MASS J08434975+2619106 - lb1402rhq.0": 1602.0,
    "2MASS J08434975+2619106 - lb1402rnq.0": 1602.0,
    #: ATO J200.4373+03.5154 (index 260, z=0.2686) is the same failure with a
    #: different face.  Here the C IV window itself is clean -- S/N 3.4-5.2 and
    #: no negative pixels through 1560 A -- but cont_filtered's interpolation
    #: anchors past 1685 A, where this spectrum reaches -7.7e+09, so the
    #: normalised flux steps down by a factor of SIX at exactly 1460 A, the
    #: exclusion boundary.  The reconstruction then sat entirely below its own
    #: continuum, the EW clamped to exactly 0.00 and the veto rejected at
    #: -139 per cent.  Trimming at 1590 A rest (2017 A observed) clears the
    #: veto, keeps 251 C IV pixels and corrects log L1450 by 1.5 dex, from
    #: 43.944 to 45.418.  The cut is placed before the 1600-1620 A bin, where
    #: 49 per cent of pixels are already negative; 1605 would buy 22 more
    #: pixels at the cost of reaching into it, for the same blueshift to one
    #: quantum.
    "ATO J200.4373+03.5154 - lb1418isq.0": 1590.0,
}
