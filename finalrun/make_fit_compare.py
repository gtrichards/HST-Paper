"""Side-by-side comparison of every adopted fit, before and after the data
changes of 2026-09-20/21.

"Before" is the fit as recorded, made on the students' rebinned master.
"After" is the same record replayed -- identical masks, components and
redshift -- on the regenerated tree: re-retrieved from MAST, every detector
segment read rather than only the first, and the co-adds rebuilt.  Nothing
about the fitting changed, so any difference is the data.

The recorded fits are not modified: the replay wrote them, and the tree was
restored from a snapshot afterwards and verified byte-identical.

Figures are pasted with Pillow rather than redrawn through matplotlib's
imshow, which rendered them upside down.

Filenames sort the set by how far the measurement moved, so a reviewer meets
the large shifts first:  A_ = more than three quanta or ten per cent,
B_ = a smaller shift,  C_ = unchanged.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SCRATCH = ("/private/tmp/claude-502/-Users-gtr-Work-projects-hstica/"
           "d417042d-a1d6-4929-8903-4ecb7e5424bd/scratchpad")
OLD, NEW = os.path.join(SCRATCH, "old_png"), os.path.join(SCRATCH, "new_png")
OUT = "pipeline_output/final/fit_compare"
BAR = 62          # caption strip height


def _font(size):
    """A readable caption font; Pillow's default bitmap font is ~11 px."""
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()
GAP = 12


def pair(old_path, new_path, caption, out_path):
    a, b = Image.open(old_path).convert("RGB"), Image.open(new_path).convert("RGB")
    h = max(a.height, b.height)
    w = a.width + GAP + b.width
    canvas = Image.new("RGB", (w, h + BAR), "white")
    canvas.paste(a, (0, BAR))
    canvas.paste(b, (a.width + GAP, BAR))
    d = ImageDraw.Draw(canvas)
    d.text((12, 8), caption, fill="black", font=_font(26))
    d.text((12, 38), "left: BEFORE (rebinned master)          "
                     "right: AFTER (regenerated tree, all detector segments)",
           fill="#555555", font=_font(19))
    d.line([(a.width + GAP // 2, BAR), (a.width + GAP // 2, h + BAR)], fill="#bbbbbb", width=2)
    canvas.save(out_path)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv("pipeline_output/retrieval_v23/replay_seg.csv")
    d["index"] = d.folder.str.extract(r"^(\d+)_")[0].astype(int)
    dec = pd.read_csv("pipeline_output/final/pass_decisions.csv")[["index", "manmask", "name_pub"]]
    d = d.merge(dec, on="index", how="left")
    d["quanta"] = ((d.blue_after - d.blue_before) / 69.0).round().astype(int)
    d["dew_pct"] = 100 * (d.ew_after - d.ew_before) / d.ew_before.replace(0, np.nan)

    def band(r):
        if r.quanta == 0 and abs(r.dew_pct) < 0.1:
            return "unchanged"
        if abs(r.quanta) > 3 or abs(r.dew_pct) > 10:
            return "large"
        return "small"
    d["band"] = d.apply(band, axis=1)
    pre = {"large": "A", "small": "B", "unchanged": "C"}

    made = 0
    for _, r in d.sort_values("index").iterrows():
        stem = "%s__%s.png" % (r.folder, r.record)
        o, n = os.path.join(OLD, stem), os.path.join(NEW, stem)
        if not (os.path.exists(o) and os.path.exists(n)):
            continue
        name = r.name_pub if isinstance(r.name_pub, str) else r["name"]
        mm = "" if pd.isna(r.manmask) else "manmask %d  " % int(r.manmask)
        cap = ("%03d  %s   %s%s      blueshift %.1f -> %.1f (%+d quanta)      "
               "EW %.1f -> %.1f (%+.1f%%)      [%s]"
               % (r["index"], name, mm, r.record, r.blue_before, r.blue_after,
                  r.quanta, r.ew_before, r.ew_after,
                  0.0 if pd.isna(r.dew_pct) else r.dew_pct, r.band))
        pair(o, n, cap, os.path.join(
            OUT, "%s_%03d_%s.png" % (pre[r.band], r["index"], r["name"].replace("/", "_"))))
        made += 1

    d[["index", "name", "name_pub", "manmask", "record", "blue_before", "blue_after",
       "quanta", "ew_before", "ew_after", "dew_pct", "band"]].sort_values("index").to_csv(
        "pipeline_output/final/fit_compare.csv", index=False)
    print("%d comparison figures -> %s" % (made, OUT))
    print(d.band.value_counts().to_string())


if __name__ == "__main__":
    main()
