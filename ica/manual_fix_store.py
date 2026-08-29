# Single source of truth for per-object manual fixes.
#
# Why this module exists
# ----------------------
# There were two unconnected records of "what we did by hand to this object":
#
#   * MANUAL_FIX_CONFIG (ica/manual_fix_config.py) -- a hand-edited Python dict
#     read by the batch runner. Knows only `custom_mask_pixels` (a flat array of
#     wavelengths) and `forced_components`.
#   * manual_fix_overrides.json -- written by the Save button in manual_fix_gui.
#     Knows mask ranges, single pixels, *un*mask ranges and pixels, and forced
#     components. Nothing read it, so GUI decisions never reached a batch run.
#
# resolve_override() normalises both into one shape, so the GUI and the batch
# runner apply byte-identical overrides and a batch fit reproduces a GUI fit.
#
# Precedence: a JSON entry replaces the MANUAL_FIX_CONFIG entry for that object
# outright (it is the newer, interactively-reviewed record, and the GUI edits it
# as a whole). Objects in both are reported by check_conflicts() rather than
# merged silently -- merging two pixel lists would double-apply masks.

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ica.manual_fix_config import MANUAL_FIX_CONFIG

# Kept next to the config it complements.
DEFAULT_OVERRIDES_JSON = Path(__file__).resolve().parent / "manual_fix_overrides.json"

# Keys of the normalised override dict returned by resolve_override().
_EMPTY = dict(mask_ranges=[], mask_pixels=[],
              unmask_ranges=[], unmask_pixels=[],
              comps_use=None, source="none")


def load_overrides(path=DEFAULT_OVERRIDES_JSON):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_overrides(path, overrides):
    with open(path, "w") as f:
        json.dump(overrides, f, indent=2)


def _from_json_entry(entry):
    return dict(
        mask_ranges=[list(r) for r in entry.get("mask_ranges", [])],
        mask_pixels=[float(w) for w in entry.get("mask_pixels", [])],
        unmask_ranges=[list(r) for r in entry.get("unmask_ranges", [])],
        unmask_pixels=[float(w) for w in entry.get("unmask_pixels", [])],
        comps_use=entry.get("forced_components") or None,
        source="json",
    )


def _from_config_entry(config):
    """MANUAL_FIX_CONFIG's custom_mask_pixels is an array of wavelengths, each
    masking its nearest pixel -- exactly what mask_pixels means here, so the
    batch behaviour is preserved when an object has no JSON entry."""
    pixels = config.get("custom_mask_pixels")
    return dict(
        mask_ranges=[],
        mask_pixels=[] if pixels is None else [float(w) for w in np.asarray(pixels).ravel()],
        unmask_ranges=[],
        unmask_pixels=[],
        comps_use=config.get("forced_components") or None,
        source="config",
    )


def resolve_override(name, overrides=None, overrides_path=DEFAULT_OVERRIDES_JSON):
    """Normalised manual fix for `name`, from the JSON store if present, else
    from MANUAL_FIX_CONFIG, else empty. `overrides` lets a caller pass an
    already-loaded JSON dict (e.g. the GUI's in-memory copy)."""
    if overrides is None:
        overrides = load_overrides(overrides_path)
    if name in overrides:
        return _from_json_entry(overrides[name])
    if name in MANUAL_FIX_CONFIG:
        return _from_config_entry(MANUAL_FIX_CONFIG[name])
    return dict(_EMPTY, mask_ranges=[], mask_pixels=[],
                unmask_ranges=[], unmask_pixels=[])


def is_empty(ov):
    """True when an override would leave the fit untouched."""
    return not (ov["mask_ranges"] or ov["mask_pixels"] or ov["unmask_ranges"]
                or ov["unmask_pixels"] or ov["comps_use"])


def describe(ov):
    """One-line human summary, for batch logs and provenance columns."""
    if is_empty(ov):
        return "none"
    bits = []
    for key, label in (("mask_ranges", "mask range"), ("mask_pixels", "mask pix"),
                       ("unmask_ranges", "unmask range"), ("unmask_pixels", "unmask pix")):
        if ov[key]:
            bits.append("%d %s%s" % (len(ov[key]), label,
                                     "s" if len(ov[key]) != 1 else ""))
    if ov["comps_use"]:
        bits.append("comps=%s" % ov["comps_use"])
    return "%s: %s" % (ov["source"], ", ".join(bits))


# ---------------------------------------------------------------------------
# Measurements written by the GUI's Save button
# ---------------------------------------------------------------------------
# The override JSON records the *inputs* to a manual fix. This CSV records what
# those inputs measured, so a result survives moving on to the next object. The
# first seven columns deliberately match ICA_Results_Rebin_master.csv, so a GUI
# row and a batch row can be compared or joined directly.
DEFAULT_MEASUREMENTS_CSV = Path(__file__).resolve().parent / "manual_fix_measurements.csv"

MEASUREMENT_COLUMNS = [
    "object_name", "spec_name", "redshift", "CIV_blueshift", "CIV_EW", "f2500",
    "components_used", "override_source", "override_summary", "plot_file",
    "saved_utc",
]


def load_measurements(path=DEFAULT_MEASUREMENTS_CSV):
    """Existing rows as {object_name: row-dict}, preserving insertion order."""
    import csv
    rows = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows[row.get("object_name", "")] = row
    return rows


def upsert_measurement(row, path=DEFAULT_MEASUREMENTS_CSV):
    """Insert or replace the row for row['object_name'] and rewrite the file.

    One row per object: re-saving an object after another re-fit updates its
    numbers in place rather than appending a second, contradictory record.
    Rewriting whole is fine at this scale (hundreds of rows) and keeps the file
    consistent if a run is interrupted mid-session.
    """
    import csv
    rows = load_measurements(path)
    rows[row["object_name"]] = {c: row.get(c, "") for c in MEASUREMENT_COLUMNS}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MEASUREMENT_COLUMNS)
        w.writeheader()
        for r in rows.values():
            w.writerow({c: r.get(c, "") for c in MEASUREMENT_COLUMNS})
    return len(rows)


# ---------------------------------------------------------------------------
# Systemic redshift overrides
# ---------------------------------------------------------------------------
# A sparse table of deliberate redshift corrections, layered over whatever the
# rebinned FITS carries (which came from master_catalog best_z at rebin time).
# Only objects listed here are changed; everything else is untouched.
#
# Deliberately NOT read from CIV_measurements_v22.xlsx: that column is rounded
# to 4 decimal places for most objects, so using it wholesale would shift
# hundreds of redshifts by up to ~14 km/s that nobody intended to change.
# Edit there if you like, then import with sandbox/import_redshifts_from_xlsx.py.
DEFAULT_REDSHIFT_CSV = Path(__file__).resolve().parent / "redshift_overrides.csv"

REDSHIFT_COLUMNS = ["object", "z", "source", "note", "updated_utc"]


def load_redshift_overrides(path=DEFAULT_REDSHIFT_CSV):
    """{object key: {z, source, note}}. Keys may be a FITS stem
    ('3C345_FOS') or a bare object name ('3C345'); the bare name applies to
    every instrument of that object, which is what a systemic redshift means."""
    import csv
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = str(row.get("object", "")).strip()
            if not key:
                continue
            try:
                z = float(row["z"])
            except (KeyError, TypeError, ValueError):
                continue
            if z <= 0:
                continue
            out[key] = dict(z=z, source=str(row.get("source", "")).strip(),
                            note=str(row.get("note", "")).strip())
    return out


def resolve_redshift(name, z_default, overrides=None,
                     path=DEFAULT_REDSHIFT_CSV):
    """Redshift to use for `name`, and where it came from.

    Tries the full spectrum stem first, then the object name with the
    instrument suffix stripped. Returns (z, source) where source is "catalog"
    when no override applies.
    """
    if overrides is None:
        overrides = load_redshift_overrides(path)
    if not overrides:
        return float(z_default), "catalog"
    for key in (str(name).strip(), str(name).strip().rsplit("_", 1)[0]):
        hit = overrides.get(key)
        if hit is not None:
            return float(hit["z"]), (hit["source"] or "override")
    return float(z_default), "catalog"


def save_redshift_overrides(rows, path=DEFAULT_REDSHIFT_CSV):
    """Write the override table. `rows` is an iterable of dicts."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REDSHIFT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in REDSHIFT_COLUMNS})


def check_conflicts(overrides_path=DEFAULT_OVERRIDES_JSON):
    """Objects defined in both stores. The JSON wins; this makes that visible
    instead of silent, so a stale config entry cannot quietly be ignored."""
    return sorted(set(load_overrides(overrides_path)) & set(MANUAL_FIX_CONFIG))
