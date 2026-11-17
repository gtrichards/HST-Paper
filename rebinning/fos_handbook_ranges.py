"""The wavelength range each FOS grating is specified for, by detector.

Source: FOS Instrument Handbook, Version 6.0, Table 1-3 ("FOS Dispersers"),
observed-frame Angstroms.  The archive's calibrated arrays run past these
limits -- G190H on the red Digicon is delivered from 1571.9 A although the
Handbook's range begins at diode 503, 1590 A, with footnote g: "The red Digicon
fused silica faceplate strongly absorbs some light shortward of 1650 A."

WHY THIS EXISTS (2026-11-17).  Every red-Digicon G190H exposure whose array
reaches below 1590 A -- 72 of 195 in the staged sample, and none of any other
grating or detector -- carries its first six pixels with quoted errors 1e-7 of
the median.  Under 1/sigma^2 weighting those six pixels outvote any number of
good exposures that cover the same wavelengths (fourteen, on I Zw 1).  They are
the diodes beyond the end of the specified range.  Truncating each exposure to
its Handbook range removes them by definition, with a sentence any HST user
accepts, and removes nothing inside the range.

Keys are (DETECTOR, FGWA_ID) as the headers spell them.  An exposure whose
detector or grating is not listed is left untouched.  PRISM is not listed.
"""

RANGES = {
    # blue Digicon
    ("BLUE",  "H13"): (1140.0, 1606.0),
    ("BLUE",  "H19"): (1573.0, 2330.0),
    ("BLUE",  "H27"): (2221.0, 3301.0),
    ("BLUE",  "H40"): (3240.0, 4822.0),
    ("BLUE",  "H57"): (4574.0, 6872.0),
    ("BLUE",  "L15"): (1140.0, 2508.0),
    ("BLUE",  "L65"): (3540.0, 9022.0),
    # red (amber) Digicon
    ("AMBER", "H19"): (1590.0, 2312.0),
    ("AMBER", "H27"): (2222.0, 3277.0),
    ("AMBER", "H40"): (3235.0, 4781.0),
    ("AMBER", "H57"): (4569.0, 6818.0),
    ("AMBER", "H78"): (6270.0, 8500.0),
    ("AMBER", "L15"): (1571.0, 2424.0),
    ("AMBER", "L65"): (3540.0, 7075.0),
}


def handbook_range(header):
    """(low, high) in observed Angstroms for this exposure, or (None, None)."""
    det = str(header.get("DETECTOR", "")).strip().upper()
    grat = str(header.get("FGWA_ID", "")).strip().upper()
    return RANGES.get((det, grat), (None, None))
