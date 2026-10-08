"""For each object with more than one spectrum: what went into the fit, and what did not.

The rebinned product and the fit figures show the result of the co-addition but
say nothing about which archive exposures reached it.  Several things quietly
remove data on the way -- CalCOS extractions that came back empty, exposures
carrying no flux, the temporary highest-S/N cap on the monitoring targets, and
x1dsum products deliberately skipped because their constituent exposures are
read individually -- and a reviewer looking at a fit has no way to see any of it.

Every archive exposure is drawn in the rest frame: those the pipeline used in
colour, those it dropped in grey, with the reason listed.  The vertical scale is
set by the exposures that were used, so that the data the fit actually saw is
readable; a dropped exposure at a very different level would then be a flat line
on the floor or off the panel entirely, so its level is written into its legend
entry instead.  That matters because the screening has been seen to drop the
right exposure and keep the wrong one (index 119), and this figure is where that
is caught.

    python figure_included_excluded.py --index 11 19
    python figure_included_excluded.py --multi-only
"""
import argparse
import csv
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/gtr/Work/git/HST-Paper")
from rebinning import read_spec_data as R
from exposure_decisions import (read_exposures as _dec_read, per_file as _dec_per_file,
                                assign_epochs as _dec_epochs, recommend as _dec_recommend,
                                alignment as _dec_align, choose_visit as _dec_choose,
                                _overrides as _dec_OV)

HERE = os.path.dirname(os.path.abspath(__file__))
MAST = os.path.join(HERE, "data_v23", "MAST_v23")
OUT = os.path.join(HERE, "pipeline_output", "final", "data_review")
ICA = (1260.0, 3000.0)
CIV = (1500.0, 1600.0)
MAX_LINES = 40          # drawn per class; the caption always states the true count


def exposures(path, z):
    """(label, rest wavelength, flux) for every archive file, however it was used.

    FOS is stored differently from COS and STIS: the wavelengths are in the _c0f
    file and the fluxes in its _c1f partner, so the pair has to be read together
    rather than looking for a WAVELENGTH column that FOS does not have.
    """
    out = []
    for p0 in sorted(glob.glob(os.path.join(glob.escape(path), "*_c0f.fits"))):
        p1 = p0.replace("_c0f.fits", "_c1f.fits")
        if not os.path.exists(p1):
            continue
        try:
            w = np.atleast_2d(fits.getdata(p0).astype(float))
            f = np.atleast_2d(fits.getdata(p1).astype(float))
        except Exception:
            continue
        for k in range(min(w.shape[0], f.shape[0])):
            g = np.isfinite(w[k]) & np.isfinite(f[k]) & (w[k] > 500) & (f[k] != 0)
            if g.sum() > 20:
                out.append((os.path.basename(p0), w[k][g] / (1 + z), f[k][g]))
    if out:
        return out
    pats = ("*_x1d.fits", "*_x1dsum.fits", "*_sx1.fits")
    for p in sorted({f for q in pats for f in glob.glob(os.path.join(glob.escape(path), q))}):
        try:
            with fits.open(p, memmap=False) as h:
                t = h[1].data
                if t is None or "WAVELENGTH" not in getattr(t, "names", []):
                    continue
                for r in np.atleast_1d(t):
                    w = np.asarray(r["WAVELENGTH"], float)
                    f = np.asarray(r["FLUX"], float)
                    # Show what the pipeline uses, not what the file contains.
                    # Pixels with flux exactly 0.0 are dead ones the readers now
                    # mask; drawing them made kept exposures appear to lie at zero
                    # and sent us looking for a contamination that was not there.
                    g = np.isfinite(w) & np.isfinite(f) & (w > 500) & (f != 0)
                    if g.sum() > 20:
                        out.append((os.path.basename(p), w[g] / (1 + z), f[g]))
        except Exception:
            continue
    return out


def draw(index, name, z, out_path, label="", simple=False):
    base = os.path.join(MAST, name)
    insts = [d for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, d))
             and glob.glob(os.path.join(glob.escape(os.path.join(base, d)), "*.fits"))]
    if not insts:
        return None
    # One x range for the whole object, from the coverage that actually exists:
    # padding every panel out to the nominal 1260-3000 A of the ICA components
    # buries the data in empty axis, and per-panel ranges stop the panels being
    # comparable with each other.
    # "Used" means the visit the selection policy actually fits, not merely every
    # exposure that survived screening.  Showing the screening verdict alone put
    # 38 COS exposures of NGC 3783 in blue when the object is to be fitted on
    # three FOS exposures, and showed FOS visits at quite different flux levels
    # all as used.
    chosen = None
    try:
        per_inst, rows_all, recs_all = {}, {}, []
        for inst in insts:
            raw = _dec_read(os.path.join(base, inst), inst, z)
            recs = _dec_per_file(raw)
            if not recs:
                continue
            for q in raw:
                rows_all.setdefault(q["file"], []).append(q)
            per_inst[inst] = _dec_epochs(recs)
            recs_all += per_inst[inst]
        al = _dec_align(recs_all, rows_all)
        for inst, recs in per_inst.items():
            for r in recs:
                r["align_rms"] = al.get(r["file"], np.nan)
            per_inst[inst] = _dec_recommend(recs)[0]
        cands = _dec_choose(per_inst, rows_all, index=index)
        if cands:
            b = cands[0]
            # Apply the hand overrides, exactly as fit_chosen_visit.py does.
            # Without this the panel showed what the PIPELINE would have chosen
            # rather than what was actually fitted, which inverts the picture on
            # every object carrying an override -- on PG 0953+414 (index 119) it
            # drew the corrupt exposure as kept and the good one it replaced as
            # dropped.  This figure is where a reviewer checks which photons
            # reached the fit, so it has to show the fit that exists.
            _ov = _dec_OV().get(index, {})
            _want = _ov.get("epochs") or (b["epoch"],)
            _force = _ov.get("keep_files", ())
            kept = [r for r in per_inst[b["inst"]]
                    if r["epoch"] in _want
                    and (str(r["action"]).startswith("keep")
                         or any(r["file"].startswith(x) for x in _force))]
            _drop_g = _ov.get("drop_gratings", ())
            if _drop_g:
                kept = [r for r in kept
                        if str(r["grating"]).upper() not in _drop_g]
            _drop_f = _ov.get("drop_files", ())
            if _drop_f:
                kept = [r for r in kept
                        if not any(r["file"].startswith(x) for x in _drop_f)]
            chosen = (b["inst"], b["epoch"], {r["file"] for r in kept}, b,
                      tuple(_force), tuple(_drop_f), tuple(_drop_g))
    except Exception:
        chosen = None

    spans = []
    for inst in insts:
        for _, w, _ in exposures(os.path.join(base, inst), z):
            spans.append((w.min(), w.max()))
    # The axis is set by every exposure drawn, kept and dropped alike -- NOT by
    # the ICA component range.  Clamping to ICA[0] = 1260 A silently dropped the
    # excluded exposures off the left edge whenever they lay blueward of it: on
    # Mrk 1383 (index 88) the panel legended four G140M exposures covering
    # 1100-1196 A and drew none of them, and GTR, looking at the figure to find
    # out why the fit had changed, could not see what had been excluded.  An
    # exposure outside the kept range is exactly the case the figure exists to
    # show.  The floor is physical rather than nominal: HST ultraviolet coverage
    # begins near 900 A and read_spec_data drops anything below 500 A as an
    # unphysical wavelength solution.
    if spans:
        XLO = max(900.0, min(a for a, _ in spans) - 10)
        XHI = max(b for _, b in spans) + 10
    else:
        XLO, XHI = ICA
    fig, axes = plt.subplots(len(insts), 1, figsize=(15.5, 3.4 * len(insts)), squeeze=False)
    summary = []
    for ax, inst in zip(axes[:, 0], insts):
        d = os.path.join(base, inst)
        try:
            R.read_data_flat(name, d, inst, z)
            prov = {p["file"]: p for p in R.PROVENANCE}
        except Exception as exc:
            prov = {}
            ax.text(0.01, 0.9, "reader failed: %s" % exc, transform=ax.transAxes, fontsize=9)
        # The pipeline's own verdict is only half the story: the figure also has
        # to show what is being PROPOSED for exclusion, or there is nothing for
        # GTR to agree or disagree with.  Three classes, not two.
        try:
            recs = _dec_recommend(_dec_epochs(_dec_per_file(_dec_read(d, inst, z))))[0]
            action = {r["file"]: r["action"] for r in recs}
        except Exception:
            action = {}
        used, dropped, proposed, review = [], [], [], []
        for fn, w, f in exposures(d, z):
            p = prov.get(fn)
            act = action.get(fn, "keep")
            # A hand override in spectrum_overrides.csv supersedes the
            # screening's own verdict in BOTH directions, and the panel has to
            # say so rather than repeat the screening: on index 119 the override
            # drops an exposure the screening kept and reinstates one it called
            # "no signal", and labelling either by the screening's reason would
            # describe a decision that was overturned.
            _forced = chosen[4] if chosen is not None and len(chosen) > 4 else ()
            _dropped_by_hand = chosen[5] if chosen is not None and len(chosen) > 5 else ()
            _dropped_grating = chosen[6] if chosen is not None and len(chosen) > 6 else ()
            is_forced = any(fn.startswith(x) for x in _forced)
            if chosen is not None and not (inst == chosen[0] and fn in chosen[2]):
                if any(fn.startswith(x) for x in _dropped_by_hand):
                    why = "dropped by hand (spectrum_overrides)"
                elif (_dropped_grating
                      and str(prov.get(fn, {}).get("grating", "")).upper()
                      in _dropped_grating):
                    why = "grating dropped by hand (spectrum_overrides)"
                else:
                    why = ("not the chosen visit" if p is None or p["kept"]
                           else p["reason"])
                dropped.append((fn, w, f, dict(kept=False, reason=why)))
            elif p is not None and not p["kept"]:
                dropped.append((fn, w, f, p))
            elif act.startswith("drop") and not is_forced:
                proposed.append((fn, w, f, dict(kept=True, reason=act)))
            elif act.startswith("review") and not is_forced:
                review.append((fn, w, f, dict(kept=True, reason=act)))
            else:
                used.append((fn, w, f, p))
        # Scale on what was used -- but a panel whose every exposure was rejected
        # has nothing to scale on, and was coming out empty with the rejected data
        # far off the axis.  Fall back to the data that IS shown, so a panel of
        # rejections is still readable.
        pool = used if used else (dropped + proposed + review)
        # Scale on a smoothed version of each exposure, not the raw pixels. The
        # raw 99.5th percentile is set by noise spikes wherever the spectrum runs
        # out of signal, which on a red-dominated object squashes everything the
        # figure exists to show: GTR could not judge Mrk 1018 (index 56) because
        # "the data_review file is dominated by noise at the red end". A running
        # median leaves real continuum and lines untouched while the spikes
        # average away, so the scale follows the signal rather than the noise.
        # Scale on the C IV neighbourhood, not the whole plotted range. A running
        # median does not help: where COS runs out of signal past about 1900 A the
        # excursions are broad rather than spiky, reaching 70 times the continuum,
        # and they set the 99.5th percentile however much they are smoothed. The
        # decision this figure supports is about C IV and the continuum windows
        # either side of it, so the scale is taken from 1400-1750 A -- covering
        # 1445-1465, 1500-1600 and 1700-1705 -- whenever there are enough pixels
        # there, and from everything otherwise.
        SCALE_LO, SCALE_HI = 1400.0, 1750.0
        near = [f[(w >= SCALE_LO) & (w <= SCALE_HI)] for _, w, f, _ in pool]
        near = [a for a in near if a.size]
        n_near = int(sum(a.size for a in near))
        vals = (np.concatenate(near) if n_near >= 200
                else (np.concatenate([f for _, _, f, _ in pool]) if pool
                      else np.array([np.nan])))
        lo, hi = np.nanpercentile(vals, [1, 99.5]) if np.isfinite(vals).any() else (0, 1)
        pad = 0.15 * (hi - lo) if np.isfinite(hi - lo) and hi > lo else 1.0
        def thin(seq):
            if len(seq) <= MAX_LINES:
                return seq
            step = len(seq) / float(MAX_LINES)
            return [seq[int(k * step)] for k in range(MAX_LINES)]
        # One colour for what was used and one for what was not, as asked: the
        # question this figure answers is which class an exposure fell into, not
        # which exposure it was.
        # Individual exposures under question have to be identifiable: GTR could
        # not answer a question about one of them because the figure gave no way
        # to tell which curve it was.  Anything not simply kept gets its own
        # legend entry; the kept ones stay aggregated, since there can be
        # hundreds of them and they are not what is being decided.
        seen_lbl = set()
        def _lab(fn, reason, f=None):
            root = fn.split("_")[0]
            key = (root, reason)
            if key in seen_lbl:
                return None
            seen_lbl.add(key)
            txt = "%s  %s" % (root, reason)
            # The vertical scale is set by the exposures that were USED, so a
            # dropped exposure at a very different level renders as a flat line
            # on the floor or leaves the panel entirely -- and that is exactly
            # the case the figure exists to show.  On PG 0953+414 (index 119)
            # the screening kept a corrupt exposure 267x brighter than the good
            # one it dropped; scaled on the kept exposure, the good one is a
            # line along the bottom and the panel cannot answer GTR's question,
            # "why were the bad FOS spectra kept over COS?".  Widening the axis
            # to hold both would squash the kept data flat instead, so the level
            # is written into the legend: no curve is drawn off the axis without
            # the reader being told where it actually sits.  Same rule as the
            # x-axis fix above -- never legend a curve the axis cannot show.
            if f is not None and np.isfinite(f).any():
                med = float(np.nanmedian(f))
                span = (hi + pad) - (lo - pad)
                # Outside the axis, or inside it but flattened onto the floor --
                # both are invisible to the reader, and the second is the common
                # case when one exposure is orders of magnitude brighter.
                if np.isfinite(med) and np.isfinite(span) and span > 0:
                    if not (lo - pad <= med <= hi + pad):
                        txt += "  [level %.2e, OFF SCALE]" % med
                    elif abs(med - (lo - pad)) < 0.02 * span:
                        txt += "  [level %.2e, flat at this scale]" % med
            return txt
        # One colour per class, dash pattern to tell individuals apart.  Giving
        # each excluded exposure its own colour from the default cycle made them
        # look like ordinary data: GTR read green and red curves lying at zero as
        # still being included.  Colour now means only in or out.
        DASHES = [(6, 2), (2, 2), (8, 2, 2, 2), (4, 1, 1, 1), (10, 3), (1, 1)]
        MANY = 8   # above this, individuals cannot be told apart anyway

        def _style(n, total):
            """Dashes and a legend entry only while the curves are separable.

            NGC 5548 has 698 excluded COS exposures: dashing and labelling them
            individually produced a thicket nobody can read a single exposure
            out of.  GTR: "there is no point marking all the spectra, I can't
            see them individually."
            """
            if total > MANY:
                return dict(lw=0.5, alpha=0.5), None
            return dict(lw=1.1, alpha=0.95, dashes=DASHES[n % len(DASHES)]), True
        classes = (("#c0392b", dropped, 1, "already dropped"),
                   ("#e08214", proposed, 3, "proposed to drop"),
                   ("#7b3294", review, 4, "to review"))
        if simple:
            classes = (("0.65", dropped + proposed + review, 1, "not used"),)
        for colour, seq, zo, tag in classes:
            nfiles = len({q[0] for q in seq})
            first = True
            for n, (fn, w, f, p) in enumerate(thin(seq)):
                kw, lab_ok = _style(n, nfiles)
                lbl = None
                if lab_ok:
                    lbl = _lab(fn, p["reason"], f)
                elif first:
                    lbl = "%d %s" % (nfiles, tag)
                    first = False
                ax.plot(w, f, color=colour, zorder=zo, label=lbl, **kw)
        first_kept = True
        for fn, w, f, p in thin(used):
            ax.plot(w, f, color="#2c6fbb", lw=0.6, alpha=0.55, zorder=2,
                    label=("%d kept" % len({q[0] for q in used})) if first_kept else None)
            first_kept = False
        h, l = ax.get_legend_handles_labels()
        if h and len(h) <= 26:
            ax.legend(h, l, fontsize=7, ncol=2, loc="upper right", framealpha=0.85)
        ax.axvspan(*CIV, color="#f5d76e", alpha=0.30, zorder=0)
        ax.set_xlim(XLO, XHI)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel("Flux")
        # Count files, not rows: a COS exposure writes one row per detector
        # segment, so counting rows double-counted every dropped COS exposure.
        reasons, _seen = {}, set()
        for fn, _, _, p in dropped:
            if p and fn not in _seen:
                _seen.add(fn)
                reasons[p["reason"]] = reasons.get(p["reason"], 0) + 1
        note = "; ".join("%d %s" % (v, k) for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
        # Count files, not rows: an echelle exposure is one observation however
        # many orders it writes, and that is the unit GTR thinks in.
        n_used_f = len({fn for fn, _, _, _ in used})
        n_drop_f = len({fn for fn, _, _, _ in dropped})
        n_prop_f = len({fn for fn, _, _, _ in proposed})
        prop_reasons = {}
        _seen2 = set()
        for fn, _, _, p in proposed:
            if fn not in _seen2:
                _seen2.add(fn)
                prop_reasons[p["reason"]] = prop_reasons.get(p["reason"], 0) + 1
        pnote = "; ".join("%d %s" % (v, k) for k, v in sorted(prop_reasons.items(),
                                                             key=lambda x: -x[1]))
        n_rev_f = len({fn for fn, _, _, _ in review})
        ax.set_title("%s   %d kept (blue), %d already dropped (red), "
                     "%d proposed to drop (orange), %d to review (purple)%s%s"
                     % (inst, n_used_f, n_drop_f, n_prop_f, n_rev_f,
                        ("   --   " + note) if note else "",
                        ("   --   proposed: " + pnote) if pnote else ""),
                     fontsize=10, loc="left")
        summary.append(dict(index=index, name=name, inst=inst,
                            n_used=n_used_f, n_dropped=n_drop_f, n_proposed=n_prop_f,
                            proposed_reasons=pnote,
                            n_rows_used=len(used), n_rows_dropped=len(dropped),
                            reasons=note))
    axes[-1, 0].set_xlabel(r"Rest wavelength ($\AA$)   (shaded: the C IV window)")
    sub = ""
    if chosen is not None:
        b = chosen[3]
        sub = "   --   fitting %s visit %d: %d exposures, %.0f-%.0f A, S/N ~%.0f" % (
            b["inst"], b["epoch"], b["n_exp"], b["wmin"], b["wmax"], b["snr_combined"])
    fig.suptitle("%s   --   archive exposures behind the fit; scale set by those used%s"
                 % (label, sub), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, nargs="*", default=None)
    ap.add_argument("--multi-only", action="store_true",
                    help="only objects with more than one spectrum")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--simple", action="store_true",
                    help="two colours only -- blue for what is used, grey for what is "
                         "not.  Legible when the accepted set is small, which under the "
                         "single-visit policy it usually is.")
    ap.add_argument("--no-copy-to-fits", dest="copy_to_fits", action="store_false",
                    help="do not also place a copy in each fit_iterations_v23 object folder")
    a = ap.parse_args()

    queue = {int(r["order"]): r for r in csv.DictReader(
        open(os.path.join(HERE, "pipeline_output", "work_queue.csv")))}
    dec = pd.read_csv(os.path.join(HERE, "pipeline_output", "final",
                                   "pass_decisions.csv")).set_index("index")
    ep = pd.read_csv(os.path.join(HERE, "pipeline_output", "final", "epochs.csv"))
    _cat = pd.read_csv("/Users/gtr/Work/git/HST-Paper/Data/master_catalog_v22.csv")
    global ZCAT
    ZCAT = {}
    for _c in ("common_name", "name_mast_key", "name_pub"):
        if _c in _cat.columns and "best_z" in _cat.columns:
            for _k, _v in zip(_cat[_c], _cat["best_z"]):
                if isinstance(_k, str):
                    ZCAT.setdefault(_k.strip(), _v)
    zmap = {int(r["index"]): r for _, r in ep.groupby("index").first().reset_index().iterrows()}

    idxs = a.index if a.index else sorted(queue)
    rows = []
    for i in idxs:
        q = queue[i]
        stems = [s for s in q["stems"].split(";") if s]
        if a.multi_only and len(stems) < 2:
            continue
        name = q["name_mast_key"]
        if not os.path.isdir(os.path.join(MAST, name)):
            continue
        z = np.nan
        if i in dec.index:
            try:
                z = float(dec.loc[i, "z_adopted"])
            except Exception:
                z = np.nan
        if not np.isfinite(z):
            # pass_decisions.csv only carries the objects worked so far, so keying
            # the redshift on it silently skipped every object past index ~174 --
            # the run reported success having drawn 145 of 414.  Same fallback as
            # epochs.py and fit_chosen_visit.py: the catalogue best_z.
            for k in (name, q.get("common_name"), q.get("name_pub")):
                if isinstance(k, str) and k.strip() in ZCAT:
                    try:
                        z = float(ZCAT[k.strip()])
                    except Exception:
                        z = np.nan
                    if np.isfinite(z):
                        break
        if not np.isfinite(z):
            print("%03d %s: no redshift" % (i, name), flush=True)
            continue
        nm = str(dec.loc[i, "name_pub"]) if i in dec.index else name
        rn = dec.loc[i, "row_num"] if i in dec.index else ""
        lab = "%03d   row %s   %s" % (i, rn, nm)
        out = os.path.join(a.out_dir, "%03d_%s.png" % (i, name.replace("/", "_")))
        try:
            s = draw(i, name, z, out, lab, simple=a.simple)
            if s and a.copy_to_fits:
                # GTR compares this against the fit figures, so it belongs in the
                # same folder as them rather than in a directory of its own kind.
                # Copy into EVERY tree that holds a folder for this object, not
                # just fit_iterations_v23: that was the working tree when this
                # was written, but the definitive pass fits under
                # chosen_visit/fits, so the panel was landing beside the
                # superseded fits and never beside the ones being judged.
                for tree in ("chosen_visit/fits", "fit_iterations_v23",
                             "fit_iterations"):
                    for fold in glob.glob(os.path.join(HERE, "pipeline_output",
                                                       *tree.split("/"),
                                                       "%03d_*" % i)):
                        if os.path.isdir(fold):
                            shutil.copy2(out, os.path.join(
                                fold, "DATA_archive_exposures.png"))
            if s:
                rows += s
                print("%-52s %s" % (os.path.basename(out),
                                    " | ".join("%s %d/%d" % (r["inst"], r["n_used"],
                                                             r["n_used"] + r["n_dropped"])
                                               for r in s)), flush=True)
        except Exception as exc:
            print("%03d %-28s FAILED: %s" % (i, name[:28], exc), flush=True)
    if rows:
        df = pd.DataFrame(rows)
        idx_path = os.path.join(a.out_dir, "INDEX.csv")
        df.to_csv(idx_path, index=False)
        print("\n%d object/instrument panels -> %s" % (len(df), a.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
