# Manual-fix GUI for interactive ICA CIV fits (Phase 2 tooling).
# New file (2026-07-10). Modifies NO existing module: GuiFixProcessor subclasses
# ica.manual_fix.ICAManualFixProcessor and adds a non-plotting fit method that
# returns arrays for the embedded matplotlib canvas.
#
# What it does
# ------------
#   * Loads a rebinned spectrum, runs the *exact* ICA pipeline used by the batch
#     runner (setup_object -> main_ICA -> get_CIV_parameters), and shows the
#     usual diagnostic panels in a Qt window with the real matplotlib toolbar.
#   * Sidebar lets you add wavelength ranges to mask (right-drag on a plot) and
#     force a component set (auto/mod/low/high), then Re-fit (runs off the UI
#     thread so the ~5-10 s fit never freezes the window).
#   * Save writes the per-object override to a JSON file (mask ranges + forced
#     components). This is the artifact you edit; it is designed to be merged
#     back into the batch pipeline (see feed_into_batch() docstring below).
#
# Run
# ---
#   python -m ica.manual_fix_gui --rebin-dir "F:\Richards-data\HST Paper\RebinnedSpec_master"
#
# The GUI operates in master mode (object list = FITS stems in --rebin-dir),
# matching `python -m ica.run_all_objects --master`.

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# This GUI is built on PySide6, but PyQt6 is also installed in some environments.
# Matplotlib's Qt backend picks PyQt6 by default, which yields a toolbar whose C++
# type is unrelated to our PySide6 window. Pin the binding before matplotlib loads.
os.environ.setdefault("QT_API", "pyside6")
from PySide6 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

_UNMASK_COLOR = QtGui.QColor(0, 120, 0)  # list-item text colour for unmask rows

from ica.manual_fix import ICAManualFixProcessor
from ica import run_ica
from ica import civ_bal_regions as CIV_BAL_regions
from ica import manual_fix_store
# Re-exported so existing callers of these names keep working; the store module
# is now the single source of truth shared with the batch runner.
from ica.manual_fix_store import (
    DEFAULT_OVERRIDES_JSON, DEFAULT_MEASUREMENTS_CSV,
    load_overrides, save_overrides, resolve_override, upsert_measurement)

# Right mouse button for the SpanSelector so it never collides with the toolbar's
# left-drag pan/zoom. Left-drag = zoom (toolbar); right-drag = add mask range;
# right-*click* (press and release without dragging) = mask the nearest pixel.
_SPAN_BUTTON = 3

# A right-button press/release that moves less than this many screen pixels counts
# as a click rather than a drag, so it masks one pixel instead of a range. The
# SpanSelector ignores zero-width spans, so the two gestures never both fire.
_CLICK_SLOP_PX = 3

# Scroll-wheel zoom: one notch shrinks/grows the visible span by this factor,
# keeping the wavelength under the cursor fixed.
_ZOOM_STEP = 1.3

# Individual pixels (and the nearest-pixel hover cursor) are drawn only once the
# view is zoomed in far enough that markers are distinguishable. This also keeps
# hover redraws off the table when the whole spectrum is in view.
_PIXEL_MARK_MAX = 400

# Figure geometry matched to the batch diagnostic plot (create_diagnostic_plot
# uses figsize=(18, 10.5) with the same GridSpec(7, 12) panel layout) so GUI
# fits are directly comparable to the saved ICA plots.
_FIG_W, _FIG_H = 18.0, 10.5
_FIG_ASPECT = _FIG_W / _FIG_H

# Status banner styles (busy / success / error).
_STATUS_BUSY = "QLabel{background:#fff3cd; color:#7a5b00; padding:6px; border:1px solid #e0c060;}"
_STATUS_OK = "QLabel{background:#d4edda; color:#155724; padding:6px; border:1px solid #9ad0a5;}"
_STATUS_ERR = "QLabel{background:#f8d7da; color:#721c24; padding:6px; border:1px solid #d69098;}"


# ---------------------------------------------------------------------------
# Fit engine (headless; safe to call from a worker thread)
# ---------------------------------------------------------------------------
class GuiFixProcessor(ICAManualFixProcessor):
    """ICAManualFixProcessor fitted from overrides the GUI holds in memory.

    The fit itself lives in ICAManualFixProcessor.fit_with_overrides(), which
    process_object() also calls -- so this is literally the same code the batch
    runner executes, not a parallel copy that has to be kept in step. Touches no
    matplotlib, so it is safe to call off the UI thread.
    """

    def fit_for_gui(self, name, mask_ranges=None, mask_pixels=None, comps_use=None,
                    unmask_ranges=None, unmask_pixels=None):
        return self.fit_with_overrides(
            name, mask_ranges=mask_ranges, mask_pixels=mask_pixels,
            comps_use=comps_use, unmask_ranges=unmask_ranges,
            unmask_pixels=unmask_pixels)


# ---------------------------------------------------------------------------
# Worker (runs one fit on a background thread)
# ---------------------------------------------------------------------------
class _FitSignals(QtCore.QObject):
    done = QtCore.Signal(dict)
    failed = QtCore.Signal(str)


class _FitTask(QtCore.QRunnable):
    def __init__(self, proc, name, mask_ranges, mask_pixels, comps_use,
                 unmask_ranges, unmask_pixels):
        super().__init__()
        self.proc = proc
        self.name = name
        self.mask_ranges = mask_ranges
        self.mask_pixels = mask_pixels
        self.comps_use = comps_use
        self.unmask_ranges = unmask_ranges
        self.unmask_pixels = unmask_pixels
        self.signals = _FitSignals()

    @QtCore.Slot()
    def run(self):
        try:
            res = self.proc.fit_for_gui(
                self.name, self.mask_ranges, self.mask_pixels, self.comps_use,
                unmask_ranges=self.unmask_ranges, unmask_pixels=self.unmask_pixels)
            self.signals.done.emit(res)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Override persistence
# ---------------------------------------------------------------------------
def load_overrides(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_overrides(path, overrides):
    with open(path, "w") as f:
        json.dump(overrides, f, indent=2)


def _robust_flux_ylims(flux, wave, errs=None):
    """ylow/yhigh for the flux panels, matching create_diagnostic_plot's formula
    (ylow = 1st pct; yhigh = 99th pct + median over wave>1400) but NaN-safe.

    Falls back to the full array when wave>1400 has no finite flux -- e.g. COS
    spectra whose rest-frame coverage is entirely below 1400 A -- so an object
    with no CIV coverage renders instead of raising "Axis limits cannot be NaN".
    """
    sub = flux[wave > 1400]
    if not np.any(np.isfinite(sub)):
        sub = flux
    # Drop wild pixels before taking percentiles. A percentile is no defence
    # against a block of garbage: 2MASX-J00391586-5117013_COS carries 723
    # unflagged pixels at 2000-2360 A reaching 92,704 on a median of 2.9 --
    # 23% of the spectrum -- so its 99th percentile put the top of the panel
    # at 30,000 and the object could not be worked by hand. Anything above
    # 40x the median is not the object: the sharpest real line in the sample
    # (Mrk 1310's C IV) peaks at ~33x continuum and survives this cut.
    # Pixels whose ERROR is non-physical are not measurements and must not set
    # the scale either: the garbage blocks near 2050 A in Mrk 231_COS and
    # NGC 985_COS carry NEGATIVE errors (medians -661 and -90,604), and the
    # one in 2MASX-J00391586-5117013_COS errors of 161,499 on a median of
    # 0.42. A flux cut alone leaves the low tail of such a block in, which on
    # Mrk 231 put the panel top at 32 on a continuum of 2.9 and squashed
    # everything below 1900 A. Real sharp lines carry positive errors within
    # ~12x the median (Mrk 1310, Mrk 40), so they are untouched by this.
    if errs is not None:
        errs = np.asarray(errs, dtype=float)
        if errs.shape == flux.shape and np.any(np.isfinite(errs) & (errs > 0)):
            med_e = float(np.nanmedian(errs[np.isfinite(errs) & (errs > 0)]))
            junk = ~np.isfinite(errs) | (errs <= 0) | (errs > 100.0 * med_e)
            flux = np.where(junk, np.nan, flux)
            sub = flux[wave > 1400]
            if not np.any(np.isfinite(sub)):
                sub = flux
    finite = flux[np.isfinite(flux)]
    if finite.size:
        med = float(np.nanmedian(finite))
        if med > 0:
            keep = sub[np.isfinite(sub) & (sub < 40.0 * med)]
            if keep.size >= 0.5 * np.isfinite(sub).sum():
                sub = keep
            flux = flux[np.isfinite(flux) & (flux < 40.0 * med)] if np.isfinite(flux).sum() else flux
    ylow = max(0.0, float(np.nanpercentile(flux, 1))) if np.any(np.isfinite(flux)) else 0.0
    yhigh = (float(np.nanpercentile(sub, 99) + np.nanmedian(sub))
             if np.any(np.isfinite(sub)) else ylow + 1.0)
    if not np.isfinite(ylow):
        ylow = 0.0
    if not np.isfinite(yhigh) or yhigh <= ylow:
        yhigh = ylow + 1.0
    return ylow, yhigh


def _robust_ylims(arr):
    """NaN-safe (1st pct, 99th pct) limits for a single array (e.g. errors)."""
    if np.any(np.isfinite(arr)):
        lo = max(0.0, float(np.nanpercentile(arr, 1)))
        hi = float(np.nanpercentile(arr, 99))
    else:
        lo, hi = 0.0, 1.0
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


# ---------------------------------------------------------------------------
# Aspect-ratio-locked container
# ---------------------------------------------------------------------------
class _AspectRatioWidget(QtWidgets.QWidget):
    """Holds one child widget and keeps it at a fixed width/height aspect ratio
    by adding symmetric margins (letterbox/pillarbox), centering it. Used to stop
    the embedded canvas from stretching to the window shape, so the plot panels
    keep the same proportions as the batch diagnostic figure."""

    def __init__(self, child, aspect, locked=True):
        super().__init__()
        self._aspect = aspect
        self._locked = locked
        self._lay = QtWidgets.QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.addWidget(child)

    def set_locked(self, locked):
        """Toggle the aspect lock. Unlocked, the canvas fills the whole area,
        which gives the full-width top panel every horizontal pixel available."""
        self._locked = bool(locked)
        if not self._locked:
            self._lay.setContentsMargins(0, 0, 0, 0)
        else:
            self._apply(self.width(), self.height())

    def _apply(self, w, h):
        if h > 0:
            if w / h > self._aspect:      # too wide -> pillarbox (side margins)
                extra = max(0, w - int(round(h * self._aspect)))
                self._lay.setContentsMargins(extra // 2, 0, extra - extra // 2, 0)
            else:                         # too tall -> letterbox (top/bottom)
                extra = max(0, h - int(round(w / self._aspect)))
                self._lay.setContentsMargins(0, extra // 2, 0, extra - extra // 2)

    def resizeEvent(self, event):
        if self._locked:
            self._apply(event.size().width(), event.size().height())
        super().resizeEvent(event)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class ManualFixWindow(QtWidgets.QMainWindow):
    def __init__(self, proc, names, overrides_path,
                 measurements_path=None, plots_dir=None):
        super().__init__()
        self.proc = proc
        self.names = names
        self.overrides_path = overrides_path
        self.measurements_path = measurements_path or str(DEFAULT_MEASUREMENTS_CSV)
        # Diagnostic PNGs go in their own subfolder so they never clobber the
        # batch runner's plots for the same objects.
        self.plots_dir = plots_dir or os.path.join(
            getattr(proc, "output_path", "."), "ManualFix")
        self.overrides = load_overrides(overrides_path)
        # Point the processor at the GUI's live dict so both see the same state
        # and nothing is re-read from disk mid-session.
        self.proc._overrides = self.overrides
        self.proc.overrides_path = overrides_path
        self._override_source = "none"
        self.pool = QtCore.QThreadPool.globalInstance()

        # per-object working state
        self.mask_ranges = []      # list of [lo, hi] to mask
        self.mask_pixels = []      # wavelengths; each masks its nearest pixel
        self.unmask_ranges = []    # list of [lo, hi] to force-unmask (incl. NAL/BAL)
        self.unmask_pixels = []    # wavelengths; each unmasks its nearest pixel
        self.comps_use = "auto"
        self._saved_lims = None    # (xlim, ylim) per panel, to preserve zoom on refit
        self._busy = False

        # Plotted grid, kept so a cursor position can be snapped to a real pixel.
        # This is main_ICA's output grid (what is on screen), which equals the
        # input grid unless the spectrum needed re-binning to 69 km/s.
        self._wave_plot = None
        self._flux_plot = None
        self._hover_idx = None
        self._default_lims = None  # per-panel (xlim, ylim) as _draw() set them

        # Last completed fit, plus the override state that produced it. Save
        # refuses to record measurements when the masks have moved on since,
        # so a number in the CSV always corresponds to the override beside it.
        self._last_res = None
        self._fit_state = None
        self._pending_state = None

        self.setWindowTitle("ICA Manual Fix")
        self.resize(1600, 820)  # landscape, close to the 18:10.5 plot aspect
        self._build_ui()

        if self.names:
            self.obj_combo.setCurrentIndex(0)
            self._load_object(self.names[0])

    # ---- UI construction -------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # --- left: figure + toolbar ---
        left = QtWidgets.QVBoxLayout()
        self.fig = Figure(figsize=(_FIG_W, _FIG_H), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        left.addWidget(self.toolbar)
        # The aspect lock keeps the canvas proportioned like the saved batch ICA
        # plots, but it pillarboxes away horizontal room the full-width top panel
        # wants. Default to unlocked; the sidebar checkbox restores batch parity.
        self._aspect_box = _AspectRatioWidget(self.canvas, _FIG_ASPECT, locked=False)
        left.addWidget(self._aspect_box, stretch=1)
        root.addLayout(left, stretch=1)

        # The full-spectrum panel spans all 12 columns (it used to stop at column
        # 8, leaving the top-right quadrant empty) so single pixels are as wide
        # apart on screen as possible when hunting for ones to mask.
        gs = GridSpec(7, 12, figure=self.fig)
        self.ax_full = self.fig.add_subplot(gs[:3, :])
        self.ax_civ = self.fig.add_subplot(gs[3:6, :8])
        self.ax_err = self.fig.add_subplot(gs[6, :8], sharex=self.ax_civ)
        self.ax_fit = self.fig.add_subplot(gs[3:, 8:])
        self._panels = [self.ax_full, self.ax_civ, self.ax_err, self.ax_fit]
        self._spec_panels = (self.ax_full, self.ax_civ)

        # right-drag on the two spectrum panels adds a mask range
        # useblit=False: blitting caches an axes-background bitmap that goes
        # stale after a refit clears+replots, leaving the canvas showing the old
        # fit until a forced draw. Negligible perf cost for this occasional drag.
        self._spans = [
            SpanSelector(ax, self._on_span, "horizontal", useblit=False,
                         button=_SPAN_BUTTON,
                         props=dict(alpha=0.25, facecolor="orange"))
            for ax in self._spec_panels
        ]

        # Nearest-pixel masking, scroll zoom and the hover cursor. Right-click
        # is tracked press->release so a click can be told from a span drag.
        self._press = None
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._connect_xlim_callbacks()

        # --- right: sidebar (width-capped so the plot gets the room) ---
        side_widget = QtWidgets.QWidget()
        side_widget.setMaximumWidth(300)
        side = QtWidgets.QVBoxLayout(side_widget)
        root.addWidget(side_widget)

        # object picker
        side.addWidget(QtWidgets.QLabel("<b>Object</b>"))
        row = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("◀")
        self.next_btn = QtWidgets.QPushButton("▶")
        self.prev_btn.setFixedWidth(36)
        self.next_btn.setFixedWidth(36)
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(+1))
        self.obj_combo = QtWidgets.QComboBox()
        self.obj_combo.setEditable(True)
        self.obj_combo.addItems(self.names)
        self.obj_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.obj_combo.completer().setCompletionMode(
            QtWidgets.QCompleter.PopupCompletion)
        self.obj_combo.activated.connect(
            lambda _i: self._load_object(self.obj_combo.currentText()))
        row.addWidget(self.prev_btn)
        row.addWidget(self.obj_combo, stretch=1)
        row.addWidget(self.next_btn)
        side.addLayout(row)

        # masks (ranges + single pixels)
        side.addWidget(QtWidgets.QLabel("<b>Masks</b> (right-click / right-drag / type)"))
        hint = QtWidgets.QLabel(
            "Right-<i>click</i> a plot to mask the single nearest pixel; "
            "right-<i>drag</i> for a range. Scroll to zoom (option+scroll = "
            "vertical); individual pixels appear once zoomed in.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555; font-size:11px;")
        side.addWidget(hint)
        self.mask_list = QtWidgets.QListWidget()
        self.mask_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.mask_list.setMaximumHeight(160)
        side.addWidget(self.mask_list)

        # text entry: "1548.5" -> pixel(s); "1530-1545" -> range
        trow = QtWidgets.QHBoxLayout()
        self.mask_input = QtWidgets.QLineEdit()
        self.mask_input.setPlaceholderText("1548.5   or   1548.5 1549.2   or   1530-1545")
        self.mask_input.returnPressed.connect(self._add_mask_from_text)
        add_btn = QtWidgets.QPushButton("Add")
        add_btn.clicked.connect(self._add_mask_from_text)
        trow.addWidget(self.mask_input, stretch=1)
        trow.addWidget(add_btn)
        side.addLayout(trow)

        # Unmask mode: same text/drag interface, but force-unmasks pixels the
        # pipeline masked (e.g. a NAL flag) instead of adding a mask.
        self.unmask_mode_cb = QtWidgets.QCheckBox("Unmask mode (re-include pixels)")
        self.unmask_mode_cb.setToolTip(
            "When checked, the text box / right-click / right-drag force-include "
            "pixels the pipeline masked (including NAL/BAL flags) instead of "
            "adding a mask.")
        side.addWidget(self.unmask_mode_cb)

        self.aspect_cb = QtWidgets.QCheckBox("Lock plot aspect (match batch PNG)")
        self.aspect_cb.setToolTip(
            "Letterbox the canvas to the 18:10.5 batch diagnostic figure shape. "
            "Off (default) lets the plot fill the window, which gives the "
            "full-width top panel more room for picking out single pixels.")
        self.aspect_cb.toggled.connect(self._aspect_box.set_locked)
        side.addWidget(self.aspect_cb)

        self.reset_zoom_btn = QtWidgets.QPushButton("Reset zoom  (R)")
        self.reset_zoom_btn.setToolTip(
            "Return every panel to this object's default view. Use after "
            "scrolling off the end of the spectrum.")
        self.reset_zoom_btn.clicked.connect(self._reset_zoom)
        side.addWidget(self.reset_zoom_btn)
        # Shortcut too: the whole point is getting back quickly, and the pointer
        # is over the plot (not the sidebar) when you scroll out of range.
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, self._reset_zoom)

        mrow = QtWidgets.QHBoxLayout()
        rm = QtWidgets.QPushButton("Remove selected")
        clr = QtWidgets.QPushButton("Clear all")
        rm.clicked.connect(self._remove_selected_mask)
        clr.clicked.connect(self._clear_masks)
        mrow.addWidget(rm)
        mrow.addWidget(clr)
        side.addLayout(mrow)

        # components
        side.addWidget(QtWidgets.QLabel("<b>Components</b>"))
        self.comp_group = QtWidgets.QButtonGroup(self)
        crow = QtWidgets.QHBoxLayout()
        for label in ("auto", "mod", "low", "high"):
            rb = QtWidgets.QRadioButton(label)
            rb.toggled.connect(self._on_comp_changed)
            self.comp_group.addButton(rb)
            crow.addWidget(rb)
            if label == "auto":
                rb.setChecked(True)
        side.addLayout(crow)

        # actions
        self.refit_btn = QtWidgets.QPushButton("Re-fit")
        self.refit_btn.setStyleSheet("font-weight:bold; padding:8px;")
        # lambda so the button's `checked` bool isn't passed as preserve_zoom
        self.refit_btn.clicked.connect(lambda: self._refit())
        side.addWidget(self.refit_btn)

        self.save_btn = QtWidgets.QPushButton("Save override + measurements")
        self.save_btn.setToolTip(
            "Write the masks to the override JSON, this object's CIV "
            "measurements to the measurements CSV, and the diagnostic PNG.")
        self.save_btn.clicked.connect(self._save)
        side.addWidget(self.save_btn)

        self.save_png_cb = QtWidgets.QCheckBox("Also save diagnostic PNG")
        self.save_png_cb.setChecked(True)
        self.save_png_cb.setToolTip(
            "Render the plot to %s at the object's default view (not the "
            "current zoom, so archived plots stay comparable)." % self.plots_dir)
        side.addWidget(self.save_png_cb)

        self.status = QtWidgets.QLabel("Ready.")
        self.status.setWordWrap(True)
        side.addWidget(self.status)

        self.results = QtWidgets.QLabel("")
        self.results.setWordWrap(True)
        self.results.setStyleSheet("font-family: monospace;")
        side.addWidget(self.results)

        side.addStretch(1)

    # ---- object / state handling ----------------------------------------
    def _step(self, delta):
        if self._busy or not self.names:
            return
        i = (self.obj_combo.currentIndex() + delta) % len(self.names)
        self.obj_combo.setCurrentIndex(i)
        self._load_object(self.names[i])

    def _load_object(self, name):
        if name not in self.names:
            return
        self.current = name
        # Resolve through the shared store so an object whose fix still lives in
        # MANUAL_FIX_CONFIG opens with those masks visible and editable, instead
        # of looking un-fixed and inviting a Save that would contradict the batch.
        ov = resolve_override(name, overrides=self.overrides)
        self.mask_ranges = ov["mask_ranges"]
        self.mask_pixels = ov["mask_pixels"]
        self.unmask_ranges = ov["unmask_ranges"]
        self.unmask_pixels = ov["unmask_pixels"]
        self.comps_use = ov["comps_use"] or "auto"
        self._override_source = ov["source"]
        self._sync_mask_list()
        for rb in self.comp_group.buttons():
            if rb.text() == self.comps_use:
                rb.setChecked(True)
        self.results.setText("")
        self._refit(preserve_zoom=False)  # new object -> autoscale, don't inherit

    def _sync_mask_list(self):
        # Each row stores its (kind, value) so removal works regardless of
        # ordering (no reliance on row index -> list index). kinds:
        # "range"/"pixel" (mask) and "unmask_range"/"unmask_pixel" (unmask).
        self.mask_list.clear()
        for r in self.mask_ranges:
            it = QtWidgets.QListWidgetItem("mask range     %.2f – %.2f Å" % (r[0], r[1]))
            it.setData(QtCore.Qt.UserRole, ("range", r))
            self.mask_list.addItem(it)
        for wl in self.mask_pixels:
            it = QtWidgets.QListWidgetItem("mask pixel     %.3f Å" % wl)
            it.setData(QtCore.Qt.UserRole, ("pixel", wl))
            self.mask_list.addItem(it)
        for r in self.unmask_ranges:
            it = QtWidgets.QListWidgetItem("unmask range   %.2f – %.2f Å" % (r[0], r[1]))
            it.setData(QtCore.Qt.UserRole, ("unmask_range", r))
            it.setForeground(_UNMASK_COLOR)
            self.mask_list.addItem(it)
        for wl in self.unmask_pixels:
            it = QtWidgets.QListWidgetItem("unmask pixel   %.3f Å" % wl)
            it.setData(QtCore.Qt.UserRole, ("unmask_pixel", wl))
            it.setForeground(_UNMASK_COLOR)
            self.mask_list.addItem(it)

    def _on_span(self, xmin, xmax):
        if xmax - xmin <= 0:
            return
        # A drag narrower than the pixel spacing can fall entirely *between* two
        # grid points: the mask is applied as (wave >= lo) & (wave <= hi), so it
        # then selects nothing and silently does no masking at all. Zoomed in,
        # a few pixels of hand-shake is well under one 0.36 A spectral pixel, so
        # this is the common outcome of a click that wasn't quite still. Treat
        # any span covering no pixel as the single-pixel pick it meant to be.
        wave = self._wave_plot
        if wave is not None and len(wave):
            covered = int(np.searchsorted(wave, xmax, side="right")
                          - np.searchsorted(wave, xmin, side="left"))
            if covered == 0:
                idx = self._nearest_index(0.5 * (xmin + xmax))
                if idx is not None:
                    self._add_pixel(float(wave[idx]))
                    return
        rng = [round(float(xmin), 3), round(float(xmax), 3)]
        if self.unmask_mode_cb.isChecked():
            self.unmask_ranges.append(rng)
            msg = "Unmask range added. Re-fit to apply."
        else:
            self.mask_ranges.append(rng)
            msg = "Mask range added. Re-fit to apply."
        self._sync_mask_list()
        self._draw_mask_overlays()
        self.canvas.draw()
        self.status.setText(msg)

    def _add_mask_from_text(self):
        text = self.mask_input.text().strip()
        if not text:
            return
        try:
            entries = self._parse_mask_text(text)
        except ValueError as e:
            self._set_status("Bad mask input: %s" % e, _STATUS_ERR)
            return
        unmask = self.unmask_mode_cb.isChecked()
        for kind, val in entries:
            if kind == "range":
                (self.unmask_ranges if unmask else self.mask_ranges).append(val)
            else:
                (self.unmask_pixels if unmask else self.mask_pixels).append(val)
        self.mask_input.clear()
        self._sync_mask_list()
        self._draw_mask_overlays()
        self.canvas.draw()
        self.status.setText("Added %d %s(s). Re-fit to apply."
                            % (len(entries), "unmask" if unmask else "mask"))

    @staticmethod
    def _parse_mask_text(text):
        """Parse a mask text entry into a list of (kind, value) tuples.

        "1530-1545" or "1530:1545"  -> [("range", [1530.0, 1545.0])]
        "1548.5"                    -> [("pixel", 1548.5)]
        "1548.5 1549.2, 1550"       -> three ("pixel", wl) entries
        """
        text = text.strip()
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*[-:]\s*([0-9]*\.?[0-9]+)\s*$", text)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            if lo == hi:
                raise ValueError("range endpoints are equal")
            return [("range", [round(lo, 3), round(hi, 3)])]

        out = []
        for tok in re.split(r"[,\s]+", text):
            if not tok:
                continue
            try:
                out.append(("pixel", round(float(tok), 4)))
            except ValueError:
                raise ValueError("could not parse %r" % tok)
        if not out:
            raise ValueError("no values found")
        return out

    def _remove_selected_mask(self):
        for item in self.mask_list.selectedItems():
            kind, val = item.data(QtCore.Qt.UserRole)
            if kind == "range":
                self.mask_ranges = [r for r in self.mask_ranges if r is not val]
            elif kind == "unmask_range":
                self.unmask_ranges = [r for r in self.unmask_ranges if r is not val]
            elif kind == "pixel" and val in self.mask_pixels:
                self.mask_pixels.remove(val)
            elif kind == "unmask_pixel" and val in self.unmask_pixels:
                self.unmask_pixels.remove(val)
        self._sync_mask_list()
        self._draw_mask_overlays()
        self.canvas.draw()

    def _clear_masks(self):
        self.mask_ranges = []
        self.mask_pixels = []
        self.unmask_ranges = []
        self.unmask_pixels = []
        self._sync_mask_list()
        self._draw_mask_overlays()
        self.canvas.draw()

    def _on_comp_changed(self, checked):
        if checked:
            btn = self.comp_group.checkedButton()
            if btn:
                self.comps_use = btn.text()

    # ---- fitting ---------------------------------------------------------
    def _refit(self, preserve_zoom=True):
        if self._busy:
            return
        self._busy = True
        self._set_busy(True)
        # Preserve the current zoom only when re-fitting the SAME object. On an
        # object switch (preserve_zoom=False) the panels still hold the previous
        # object's artists, so capturing here would wrongly transplant that
        # object's x/y-limits onto the new one -> autoscale to the new data.
        self._saved_lims = [(ax.get_xlim(), ax.get_ylim()) for ax in self._panels] \
            if (preserve_zoom and self._has_drawn()) else None

        # Snapshot what is being fitted; promoted to _fit_state on completion so
        # Save can tell whether the on-screen numbers still match the masks.
        self._pending_state = self._override_snapshot()

        task = _FitTask(self.proc, self.current,
                        [list(r) for r in self.mask_ranges],
                        list(self.mask_pixels), self.comps_use,
                        [list(r) for r in self.unmask_ranges],
                        list(self.unmask_pixels))
        task.signals.done.connect(self._on_fit_done)
        task.signals.failed.connect(self._on_fit_failed)
        self.pool.start(task)

    def _has_drawn(self):
        return any(ax.lines for ax in self._panels)

    def _override_snapshot(self):
        """Hashable picture of the current override, for staleness comparison."""
        return (tuple(tuple(r) for r in self.mask_ranges),
                tuple(self.mask_pixels),
                tuple(tuple(r) for r in self.unmask_ranges),
                tuple(self.unmask_pixels),
                self.comps_use, self.current)

    def _fit_is_current(self):
        return (self._last_res is not None and self._fit_state is not None
                and self._fit_state == self._override_snapshot())

    @QtCore.Slot(dict)
    def _on_fit_done(self, res):
        self._last_res = res
        self._fit_state = self._pending_state
        try:
            self._draw(res)
            self.results.setText(
                "EW      = %7.2f Å\n"
                "blueshift = %7.1f km/s\n"
                "f2500   = %.3e\n"
                "z       = %.4f" % (res["civ_ew"], res["civ_blue"],
                                    res["f2500"], res["z"]))
            if getattr(self, "_civ_has_data", True):
                self._set_status("✓  Fit complete.", _STATUS_OK)
            else:
                self._set_status("⚠  No data in CIV region (1500–1600 Å) — "
                                 "CIV values unreliable.", _STATUS_BUSY)
        except Exception:
            # Never swallow a draw error silently: surface it and still leave the
            # canvas repainted with whatever was drawn before the failure.
            self._set_status("✗  Draw error (see console).", _STATUS_ERR)
            traceback.print_exc()
            try:
                self.canvas.draw()
            except Exception:
                pass
        finally:
            self._busy = False
            self._set_busy(False)

    @QtCore.Slot(str)
    def _on_fit_failed(self, tb):
        self._busy = False
        self._set_busy(False)
        self._set_status("✗  Fit FAILED (see console).", _STATUS_ERR)
        print(tb, file=sys.stderr)

    def _set_controls_enabled(self, on):
        for w in (self.refit_btn, self.prev_btn, self.next_btn,
                  self.obj_combo, self.save_btn):
            w.setEnabled(on)

    def _set_status(self, text, style):
        self.status.setText(text)
        self.status.setStyleSheet(style)

    def _set_busy(self, busy):
        """Toggle the fitting-in-progress visuals: disable controls, relabel the
        Re-fit button, colour the status banner, and set a wait cursor."""
        self._set_controls_enabled(not busy)
        if busy:
            self.refit_btn.setText("Fitting…")
            self._set_status("⏳  Fitting %s …" % self.current, _STATUS_BUSY)
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        else:
            self.refit_btn.setText("Re-fit")
            QtWidgets.QApplication.restoreOverrideCursor()

    # ---- nearest-pixel picking, zoom and hover ---------------------------
    def _connect_xlim_callbacks(self):
        """(Re)subscribe to xlim_changed on the zoomable panels.

        Axes.clear() installs a brand-new CallbackRegistry, silently dropping
        every prior connection, so this has to run again after each _draw().
        """
        for ax in (self.ax_full, self.ax_civ, self.ax_err):
            ax.callbacks.connect("xlim_changed", self._on_xlim_changed)

    def _nearest_index(self, x):
        """Index of the plotted pixel whose wavelength is closest to x, or None."""
        wave = self._wave_plot
        if wave is None or x is None or len(wave) == 0:
            return None
        i = int(np.searchsorted(wave, x))
        if i <= 0:
            return 0
        if i >= len(wave):
            return len(wave) - 1
        # searchsorted brackets x; pick whichever neighbour is actually closer
        return i if abs(wave[i] - x) < abs(x - wave[i - 1]) else i - 1

    def _visible_count(self, ax):
        """How many plotted pixels fall inside ax's current x limits."""
        wave = self._wave_plot
        if wave is None or len(wave) == 0:
            return 0
        lo, hi = sorted(ax.get_xlim())
        return int(np.searchsorted(wave, hi) - np.searchsorted(wave, lo))

    def _picking_locked(self):
        """True while the toolbar's Pan/Zoom modes own the mouse. Those bind the
        right button too (right-drag = zoom out), and they take the canvas
        widgetlock, which already suppresses the SpanSelector -- so pixel picking
        has to stand down as well or one right-click would do two things."""
        return bool(getattr(self.toolbar, "mode", "")) \
            or self.canvas.widgetlock.locked()

    def _on_button_press(self, event):
        if (event.button == _SPAN_BUTTON and event.inaxes in self._spec_panels
                and not self._picking_locked()):
            self._press = (event.x, event.y)
        else:
            self._press = None

    def _on_button_release(self, event):
        """Right-click (no drag) masks the single pixel nearest the cursor.
        A right-*drag* is left to the SpanSelector, which adds a range instead."""
        press, self._press = self._press, None
        if (event.button != _SPAN_BUTTON or press is None
                or event.inaxes not in self._spec_panels):
            return
        if max(abs(event.x - press[0]), abs(event.y - press[1])) > _CLICK_SLOP_PX:
            return  # a drag; the SpanSelector is handling it
        idx = self._nearest_index(event.xdata)
        if idx is None:
            return
        self._add_pixel(float(self._wave_plot[idx]))

    def _add_pixel(self, wl):
        """Record one nearest-pixel mask/unmask at the (snapped) wavelength wl."""
        unmask = self.unmask_mode_cb.isChecked()
        target = self.unmask_pixels if unmask else self.mask_pixels
        kind = "Unmask" if unmask else "Mask"
        # Clicking the same pixel twice is a mis-click far more often than it is
        # a deliberate duplicate, so toggle it back off instead of stacking.
        near = [w for w in target if abs(w - wl) < 1e-6]
        if near:
            for w in near:
                target.remove(w)
            msg = "%s removed at %.3f Å." % (kind, wl)
        else:
            target.append(wl)
            msg = "%s pixel %.3f Å added. Re-fit to apply." % (kind, wl)
        self._sync_mask_list()
        self._draw_mask_overlays()
        self.canvas.draw_idle()
        self._set_status(msg, _STATUS_OK)

    def _on_scroll(self, event):
        """Scroll to zoom about the cursor: x by default, y with a modifier held.

        Any modifier -- Option/Alt, Control, Command or Shift -- selects the
        vertical axis. Shift alone was the original binding, but macOS remaps
        Shift+wheel to a horizontal scroll before Qt sees it, so the event
        arrives with a zero vertical delta and matplotlib emits nothing; on a
        Mac the binding was simply dead. Option is the one to reach for there.
        ax_err shares x with ax_civ, so zooming either keeps them aligned."""
        ax = event.inaxes
        if ax is None or ax not in (self.ax_full, self.ax_civ, self.ax_err):
            return
        factor = 1.0 / _ZOOM_STEP if event.button == "up" else _ZOOM_STEP
        mods = ("shift", "alt", "ctrl", "control", "cmd", "super", "meta")
        vertical = bool(event.key and any(m in event.key for m in mods))
        if vertical:
            lo, hi = ax.get_ylim()
            anchor = event.ydata
        else:
            lo, hi = ax.get_xlim()
            anchor = event.xdata
        if anchor is None:
            return
        # Keep the value under the cursor fixed while the span scales.
        new_lo = anchor - (anchor - lo) * factor
        new_hi = anchor + (hi - anchor) * factor
        if vertical:
            ax.set_ylim(new_lo, new_hi)
        else:
            ax.set_xlim(new_lo, new_hi)
        self.canvas.draw_idle()

    def _reset_zoom(self):
        """Put every panel back to the limits _draw() chose for this object.

        Scroll-zooming can easily wander off the spectrum entirely, and the
        toolbar's Home button restores its own view stack, which a re-fit
        invalidates -- so this is the reliable way back.
        """
        if self._default_lims is None:
            return
        for ax, (xl, yl) in zip(self._panels, self._default_lims):
            ax.set_xlim(xl)
            ax.set_ylim(yl)
        # Don't let a stale zoom be re-applied by the next re-fit.
        self._saved_lims = None
        for ax in self._spec_panels:
            self._on_xlim_changed(ax)
            self._hide_hover(ax)
        self.canvas.draw_idle()
        self._set_status("Zoom reset to default view.", _STATUS_OK)

    def _on_xlim_changed(self, ax):
        """Reveal per-pixel markers once few enough pixels are in view."""
        marks = getattr(ax, "_pix_marks", None)
        if marks is None:
            return
        show = 0 < self._visible_count(ax) <= _PIXEL_MARK_MAX
        if marks.get_visible() != show:
            marks.set_visible(show)
            if not show:
                self._hide_hover(ax)

    def _hide_hover(self, ax):
        hov = getattr(ax, "_hover_mark", None)
        if hov is not None and hov.get_visible():
            hov.set_visible(False)

    def _on_motion(self, event):
        """Highlight the pixel that a right-click would mask, but only when
        zoomed in far enough for the markers to be showing (a redraw per pixel
        crossed would be far too slow with the whole spectrum in view)."""
        ax = event.inaxes
        changed = False
        for other in self._spec_panels:
            if other is not ax and getattr(other, "_hover_mark", None) is not None:
                if other._hover_mark.get_visible():
                    other._hover_mark.set_visible(False)
                    changed = True
        if ax in self._spec_panels and getattr(ax, "_pix_marks", None) is not None \
                and ax._pix_marks.get_visible():
            idx = self._nearest_index(event.xdata)
            if idx is not None and (idx != self._hover_idx
                                    or not ax._hover_mark.get_visible()):
                self._hover_idx = idx
                ax._hover_mark.set_data([self._wave_plot[idx]],
                                        [self._flux_plot[idx]])
                ax._hover_mark.set_visible(True)
                changed = True
        if changed:
            self.canvas.draw_idle()

    # ---- drawing ---------------------------------------------------------
    def _draw(self, res):
        wave, flux = res["wave_arb"], res["flux_arb"]
        errs, mask = res["errs_arb"], res["mask_arb"]
        wave_ica, flux_ica = res["wave_ica"], res["flux_ica"]
        spec_name = res["spec_name"]

        # Snap target for right-click masking and the hover cursor. This is the
        # grid actually on screen; fit_for_gui() re-finds the nearest pixel on
        # the pre-ICA grid, which is the same one unless main_ICA re-binned.
        self._wave_plot, self._flux_plot = wave, flux
        self._hover_idx = None

        # Drop marker references *before* clear(): clearing resets the axis
        # limits, which fires xlim_changed on artists that no longer exist.
        for ax in self._spec_panels:
            ax._pix_marks = None
            ax._hover_mark = None

        for ax in self._panels:
            ax.clear()
        self._connect_xlim_callbacks()   # clear() discarded the old registry
        # ax.clear() already removed the overlay artists; drop the stale
        # references so _draw_mask_overlays() doesn't try to remove them again
        # (that raises NotImplementedError: cannot remove artist).
        for ax in (self.ax_full, self.ax_civ):
            ax._mask_patches = []

        ylow, yhigh = _robust_flux_ylims(flux, wave, errs=errs)
        # flag objects with no flux in the CIV window (e.g. COS coverage < 1400 A)
        self._civ_has_data = bool(
            np.any(np.isfinite(flux[(wave >= 1500) & (wave <= 1600)])))

        # full spectrum
        self.ax_full.plot(wave, flux, "-k", alpha=0.6)
        self.ax_full.plot(wave_ica, flux_ica, "-r")
        self.proc.plot_HST(wave, flux, mask, self.ax_full)
        self.ax_full.set_xlim(max(min(wave), 1250), min(max(wave), max(wave_ica)))
        self.ax_full.set_ylim(ylow, yhigh + 5)
        self.ax_full.set_ylabel("Flux (arb.)")
        self.ax_full.set_title("%s  —  ICA fit" % res["name"])

        # CIV region
        self.ax_civ.plot(wave, flux, "-k", alpha=0.6)
        self.ax_civ.plot(wave_ica, flux_ica, "-r")
        self.proc.plot_HST(wave, flux, mask, self.ax_civ)
        self.ax_civ.set_xlim(1500, 1600)
        # Scale this panel to what is actually inside the C IV window. The
        # shared percentile formula caps the top at the 99th percentile of
        # flux above 1400 A, which clips a sharp narrow peak clean off the
        # plot -- on NGC 4395 the peak was not visible at all, so there was no
        # way to judge the fit. Data and model both count, so the model's peak
        # is never lost either.
        w_civ = (wave >= 1500) & (wave <= 1600) & np.isfinite(flux)
        i_civ = (wave_ica >= 1500) & (wave_ica <= 1600) & np.isfinite(flux_ica)
        if np.any(w_civ) or np.any(i_civ):
            top = max(float(np.nanmax(flux[w_civ])) if np.any(w_civ) else ylow,
                      float(np.nanmax(flux_ica[i_civ])) if np.any(i_civ) else ylow)
            bot = min(float(np.nanmin(flux[w_civ])) if np.any(w_civ) else ylow, 0.0)
            pad = 0.06 * max(top - bot, 1e-6)
            self.ax_civ.set_ylim(bot - pad, top + pad)
        else:
            self.ax_civ.set_ylim(ylow, yhigh + 5)
        self.ax_civ.set_ylabel("Flux (arb.)")

        # errors
        self.ax_err.plot(wave, errs, "-k", alpha=0.6)
        self.ax_err.set_xlim(1500, 1600)
        self.ax_err.set_ylim(*_robust_ylims(errs))
        self.ax_err.set_xlabel("Rest wavelength (Å)")
        self.ax_err.set_ylabel("Error")

        # CIV fit panel (reuse processor helper)
        self.proc.plot_CIV_analysis(self.ax_fit, wave, flux, wave_ica, flux_ica,
                                    res["name"], res["civ_blue"], res["civ_ew"])

        self._draw_mask_overlays()

        # Per-pixel markers + the nearest-pixel hover cursor, on the two panels
        # that accept mask clicks. Both stay hidden until the view is zoomed in
        # (see _on_xlim_changed), so the zoomed-out plot looks exactly as before.
        for ax in self._spec_panels:
            ax._pix_marks = ax.plot(wave, flux, linestyle="none", marker=".",
                                    ms=3.5, color="#1f77b4", alpha=0.8,
                                    zorder=3, visible=False)[0]
            ax._hover_mark = ax.plot([], [], linestyle="none", marker="o",
                                     ms=9, mfc="none", mec="red", mew=1.6,
                                     zorder=5, visible=False)[0]

        # Snapshot the limits this draw chose, *before* any saved zoom is put
        # back -- this is what "Reset zoom" returns to.
        self._default_lims = [(ax.get_xlim(), ax.get_ylim()) for ax in self._panels]

        # restore zoom if we saved it before this refit
        if self._saved_lims is not None:
            for ax, (xl, yl) in zip(self._panels, self._saved_lims):
                ax.set_xlim(xl)
                ax.set_ylim(yl)

        for ax in self._spec_panels:
            self._on_xlim_changed(ax)

        # Force a full synchronous repaint. draw_idle() can be coalesced away
        # when the restored zoom leaves the axis limits unchanged, so the new
        # fit wouldn't show until a toolbar action forced a draw.
        self.canvas.draw()

    def _draw_mask_overlays(self):
        # remove previous overlays, then re-add current ranges (shaded spans)
        # and single pixels (thin vertical lines).
        for ax in (self.ax_full, self.ax_civ):
            for patch in list(getattr(ax, "_mask_patches", [])):
                try:
                    patch.remove()
                except (NotImplementedError, ValueError):
                    pass  # already detached (e.g. by ax.clear())
            patches = [ax.axvspan(lo, hi, color="orange", alpha=0.18, zorder=0)
                       for lo, hi in self.mask_ranges]
            patches += [ax.axvline(wl, color="orange", alpha=0.6, lw=1.0, zorder=0)
                        for wl in self.mask_pixels]
            # unmask overlays in green to distinguish from orange masks
            patches += [ax.axvspan(lo, hi, color="green", alpha=0.15, zorder=0)
                        for lo, hi in self.unmask_ranges]
            patches += [ax.axvline(wl, color="green", alpha=0.7, lw=1.0,
                                   linestyle="--", zorder=0)
                        for wl in self.unmask_pixels]
            ax._mask_patches = patches

    # ---- save ------------------------------------------------------------
    def _save(self):
        entry = {}
        if self.mask_ranges:
            entry["mask_ranges"] = [list(r) for r in self.mask_ranges]
        if self.mask_pixels:
            entry["mask_pixels"] = list(self.mask_pixels)
        if self.unmask_ranges:
            entry["unmask_ranges"] = [list(r) for r in self.unmask_ranges]
        if self.unmask_pixels:
            entry["unmask_pixels"] = list(self.unmask_pixels)
        if self.comps_use and self.comps_use != "auto":
            entry["forced_components"] = self.comps_use
        if entry:
            self.overrides[self.current] = entry
        else:
            self.overrides.pop(self.current, None)  # no-op fix -> don't store
        save_overrides(self.overrides_path, self.overrides)

        meas_note = self._record_measurement()

        note = ""
        if self._override_source == "config":
            # The fix came from MANUAL_FIX_CONFIG and now also exists in the
            # JSON, which takes precedence from here on. Say so, because the
            # config entry silently stops being the operative one.
            note = ("  (was MANUAL_FIX_CONFIG; the JSON entry now takes "
                    "precedence for this object)")
            self._override_source = "json"
        self.status.setText("Saved override for %s → %s%s\n%s"
                            % (self.current, os.path.basename(self.overrides_path),
                               note, meas_note))

    # ---- measurement + plot record ---------------------------------------
    @staticmethod
    def _safe_stem(name):
        """Object name as a filename: keep it recognisable, drop path-hostile
        characters (object names contain spaces, '+' and '.')."""
        return re.sub(r"[^A-Za-z0-9.+_-]+", "_", name).strip("_")

    def _save_plot(self, path):
        """Write the diagnostic PNG at the object's default view.

        Saving whatever zoom happens to be on screen would make the archived
        plots inconsistent with each other, so the default limits are restored
        for the render and the working view is put back afterwards.
        """
        working = [(ax.get_xlim(), ax.get_ylim()) for ax in self._panels]
        for ax in self._spec_panels:
            self._hide_hover(ax)
        try:
            if self._default_lims is not None:
                for ax, (xl, yl) in zip(self._panels, self._default_lims):
                    ax.set_xlim(xl)
                    ax.set_ylim(yl)
            self.fig.savefig(path, dpi=110)
        finally:
            for ax, (xl, yl) in zip(self._panels, working):
                ax.set_xlim(xl)
                ax.set_ylim(yl)
            self.canvas.draw_idle()

    def _record_measurement(self):
        """Append/update this object's row in the measurements CSV, and save the
        diagnostic PNG. Returns a status line.

        Refuses to write when the masks have changed since the last fit: the
        numbers on screen would then describe a different override than the one
        just saved, and a silently-stale row in a file that may feed the paper
        is worse than no row at all.
        """
        if self._last_res is None:
            return "⚠  No fit yet — measurements not recorded."
        if not self._fit_is_current():
            return ("⚠  Masks changed since the last fit — measurements NOT "
                    "recorded. Re-fit, then Save again.")

        res = self._last_res
        plot_name = ""
        if self.save_png_cb.isChecked():
            try:
                os.makedirs(self.plots_dir, exist_ok=True)
                plot_name = self._safe_stem(self.current) + ".png"
                self._save_plot(os.path.join(self.plots_dir, plot_name))
            except Exception as exc:
                plot_name = ""
                traceback.print_exc()
                return "⚠  Measurements recorded, but PNG failed: %s" % exc

        row = {
            "object_name": self.current,
            "spec_name": res.get("spec_name", ""),
            # repr() round-trips a float exactly; fixed-precision formatting
            # would silently lose digits from numbers that feed the paper and
            # would break exact comparison against the batch CSV.
            "redshift": repr(float(res["z"])),
            "CIV_blueshift": repr(float(res["civ_blue"])),
            "CIV_EW": repr(float(res["civ_ew"])),
            "f2500": repr(float(res["f2500"])),
            "components_used": "" if self.comps_use == "auto" else self.comps_use,
            "override_source": self._override_source,
            "override_summary": manual_fix_store.describe(
                resolve_override(self.current, overrides=self.overrides)),
            "plot_file": plot_name,
            "saved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        try:
            n = upsert_measurement(row, path=self.measurements_path)
        except Exception as exc:
            traceback.print_exc()
            return "⚠  Measurement write failed: %s" % exc
        return "✓  Measurements → %s (%d objects)%s" % (
            os.path.basename(self.measurements_path), n,
            "; plot → %s" % plot_name if plot_name else "")


# ---------------------------------------------------------------------------
# Reproducibility check: a batch fit must equal the GUI fit for a saved override
# ---------------------------------------------------------------------------
def verify_reproducible(rebin_dir, names=None, overrides_path=DEFAULT_OVERRIDES_JSON,
                        rtol=0.0, verbose=True):
    """Refit every object with a saved override twice -- once through the GUI
    entry point, once through the batch entry point (process_object) -- and
    report whether CIV blueshift, CIV EW and f2500 agree.

    Both routes now call ICAManualFixProcessor.fit_with_overrides(), so the only
    way this can disagree is if the override *resolution* diverges. That is
    exactly the regression worth guarding: the numbers in the paper come from the
    batch runner, while the decisions are made in the GUI.

    Returns (n_checked, failures); failures is a list of (name, field, gui, batch).
    """
    proc = _make_processor(rebin_dir)
    proc.overrides_path = overrides_path
    proc._overrides = load_overrides(overrides_path)

    targets = names if names is not None else sorted(proc._overrides)
    available = set(proc.list_master_objects())
    skipped = [n for n in targets if n not in available]
    targets = [n for n in targets if n in available]

    failures = []
    for name in targets:
        ov = resolve_override(name, overrides=proc._overrides)
        gui = proc.fit_for_gui(
            name, mask_ranges=ov["mask_ranges"], mask_pixels=ov["mask_pixels"],
            comps_use=ov["comps_use"], unmask_ranges=ov["unmask_ranges"],
            unmask_pixels=ov["unmask_pixels"])
        batch = proc.process_object(name, plot=False, save_plot=False)
        for field, g, b in (("civ_blue", gui["civ_blue"], batch["CIV_blueshift"]),
                            ("civ_ew", gui["civ_ew"], batch["CIV_EW"]),
                            ("f2500", gui["f2500"], batch["f2500"])):
            same = (g == b) if rtol == 0 else bool(np.isclose(g, b, rtol=rtol))
            if not same:
                failures.append((name, field, g, b))
        if verbose:
            mark = "OK " if not any(f[0] == name for f in failures) else "DIFF"
            print("%s %-42s blue=%9.2f ew=%7.3f  [%s]"
                  % (mark, name, gui["civ_blue"], gui["civ_ew"],
                     manual_fix_store.describe(ov)))

    if verbose:
        if skipped:
            print("\nSkipped (no spectrum in %s): %s" % (rebin_dir, ", ".join(skipped)))
        print("\n%d/%d objects reproduce exactly; %d mismatched field(s)."
              % (len(targets) - len({f[0] for f in failures}), len(targets),
                 len(failures)))
    return len(targets), failures


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _make_processor(rebin_dir):
    return GuiFixProcessor(
        rebin_path=rebin_dir, master_mode=True,
        output_path="ICA_Plots_Rebin_master")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebin-dir", default=os.environ.get("HST_PAPER_REBIN_DIR"),
        help="Directory of rebinned FITS files (master mode). "
             "Falls back to HST_PAPER_REBIN_DIR env var.")
    parser.add_argument(
        "--overrides", default=str(DEFAULT_OVERRIDES_JSON),
        help="Path to the override JSON store (read + written by Save).")
    parser.add_argument(
        "--measurements", default=str(DEFAULT_MEASUREMENTS_CSV),
        help="CSV that Save appends this object's CIV measurements to "
             "(one row per object, updated in place on re-save).")
    parser.add_argument(
        "--fix-plots", default=None,
        help="Directory for diagnostic PNGs written by Save. "
             "Defaults to <plot output>/ManualFix.")
    parser.add_argument(
        "--verify-reproducible", action="store_true",
        help="Headless check: refit every object with a saved override through "
             "both the GUI and the batch entry points and confirm the CIV "
             "measurements agree. Exits non-zero on any mismatch.")
    parser.add_argument(
        "--selftest", metavar="OBJECT", default=None,
        help="Headless self-test: build the window offscreen, fit OBJECT, "
             "save a screenshot to manual_fix_gui_selftest.png, exit.")
    args = parser.parse_args(argv)

    if not args.rebin_dir:
        parser.error("--rebin-dir is required (or set HST_PAPER_REBIN_DIR).")

    if args.verify_reproducible:
        # No Qt needed; this is pure numerics on both entry points.
        n, failures = verify_reproducible(args.rebin_dir,
                                          overrides_path=args.overrides)
        for name, field, gui_val, batch_val in failures:
            print("MISMATCH %s %s: gui=%r batch=%r" % (name, field, gui_val, batch_val))
        return 1 if failures else 0

    if args.selftest:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    proc = _make_processor(args.rebin_dir)
    names = proc.list_master_objects()
    if not names:
        parser.error("No FITS files found in %s" % args.rebin_dir)

    win = ManualFixWindow(proc, names, args.overrides,
                          measurements_path=args.measurements,
                          plots_dir=args.fix_plots)

    if args.selftest:
        target = args.selftest
        if target not in names:
            # allow passing a bare stem present in the dir
            print("Self-test object %r not in list; using first (%s)."
                  % (target, names[0]))
            target = names[0]
        win.obj_combo.setCurrentText(target)
        win._load_object(target)
        # _load_object triggers an async fit; run the fit synchronously here so
        # the self-test is deterministic.
        res = proc.fit_for_gui(target, win.mask_ranges, [], win.comps_use)
        win._draw(res)
        out = "manual_fix_gui_selftest.png"
        win.fig.savefig(out, dpi=110)
        print("Self-test OK -> %s (EW=%.2f, blue=%.1f)"
              % (out, res["civ_ew"], res["civ_blue"]))
        return 0

    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
