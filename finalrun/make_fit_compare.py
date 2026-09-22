"""Before-and-after comparison of every adopted fit, laid out for spotting
small differences.

"Before" is the fit as recorded, made on the students' rebinned master.
"After" is the same record replayed -- identical masks, components and
redshift -- on the regenerated tree.  Nothing about the fitting changed, so
any difference is the data.  The recorded fits are not modified: the replay
wrote them and the tree was restored from a snapshot and checked
byte-identical.

Layout.  The first version pasted the two whole figures side by side, which
put each panel a full figure-width away from its counterpart and made subtle
differences invisible.  Here the source figures are cut into their panels and
the two versions of each panel are stacked one above the other, so the
wavelength axes line up and a change shows as a vertical offset.  Order is
C IV zoom first (the measurement), then the equivalent-width panel, then the
full ICA fit for context.

Panel boundaries are measured as fractions of the source figure, which is
always 1040 px tall and about 1790 wide, with a single vertical gutter at
x/w = 0.68 separating the stacked left panels from the tall right one.

Filenames sort by how far the measurement moved: A_ beyond three quanta or
ten per cent, B_ a smaller shift, C_ unchanged.
"""
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SCRATCH = ("/private/tmp/claude-502/-Users-gtr-Work-projects-hstica/"
           "d417042d-a1d6-4929-8903-4ecb7e5424bd/scratchpad")
OLD, NEW = os.path.join(SCRATCH, "old_png"), os.path.join(SCRATCH, "new_png")
OUT = "pipeline_output/final/fit_compare"

LEFT = (0.006, 0.678)   # the stacked left panels, measured off the source figure
# (title, x-range, y-range) of each panel within the source figure, measured
# from its white gutters.  Laid out as columns so the whole thing is wide and
# short: a laptop screen is wide, and the two earlier attempts wasted it --
# first by putting each panel a figure-width from its counterpart, then by
# stacking everything into a column 2644 px tall and 938 wide.
PANELS = [("C IV  1500-1600 A", LEFT, (0.425, 0.968)),
          ("equivalent width", (0.686, 0.995), (0.400, 0.992)),
          ("full ICA fit", LEFT, (0.008, 0.425))]
TARGET_W = 1800         # fill the width of a laptop screen
MAX_H = 980             # and stay inside its height
PAD, LABEL = 10, 26


def _font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def crop(im, xs, ys):
    w, h = im.size
    return im.crop((int(xs[0] * w), int(ys[0] * h), int(xs[1] * w), int(ys[1] * h)))


def scaled(im, height):
    return im.resize((max(1, int(im.width * height / im.height)), height), Image.LANCZOS)


def build(old_path, new_path, caption, out_path):
    a, b = Image.open(old_path).convert("RGB"), Image.open(new_path).convert("RGB")
    cuts = [(t, crop(a, xs, ys), crop(b, xs, ys)) for t, xs, ys in PANELS]

    # One height for every panel, chosen so the row of them fills TARGET_W.
    aspects = [max(pa.width / pa.height, pb.width / pb.height) for _, pa, pb in cuts]
    gaps = PAD * (len(cuts) + 1)
    h = int((TARGET_W - gaps) / sum(aspects))
    h = min(h, (MAX_H - 2 * LABEL - 3 * PAD - 60) // 2)

    cols = [(t, scaled(pa, h), scaled(pb, h)) for t, pa, pb in cuts]
    width = gaps + sum(max(pa.width, pb.width) for _, pa, pb in cols)
    height = 58 + LABEL + h + PAD + LABEL + h + PAD

    canvas = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 8), caption, fill="black", font=_font(24))
    d.text((PAD, 36), "top row BEFORE (rebinned master)    "
                      "bottom row AFTER (regenerated tree, all detector segments)",
           fill="#666666", font=_font(17))

    x = PAD
    for title, pa, pb in cols:
        y = 58
        d.text((x, y), title, fill="#1f4e79", font=_font(17))
        canvas.paste(pa, (x, y + LABEL))
        y2 = y + LABEL + h + PAD
        d.text((x, y2), title, fill="#7f2704", font=_font(17))
        canvas.paste(pb, (x, y2 + LABEL))
        x += max(pa.width, pb.width) + PAD
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
        mm = "" if pd.isna(r.manmask) else "manmask %d   " % int(r.manmask)
        cap = ("%03d  %s   %sblueshift %.1f -> %.1f (%+d quanta)   EW %.1f -> %.1f (%+.1f%%)"
               % (r["index"], name, mm, r.blue_before, r.blue_after, r.quanta,
                  r.ew_before, r.ew_after, 0.0 if pd.isna(r.dew_pct) else r.dew_pct))
        build(o, n, cap, os.path.join(
            OUT, "%s_%03d_%s.png" % (pre[r.band], r["index"], r["name"].replace("/", "_"))))
        made += 1

    d[["index", "name", "name_pub", "manmask", "record", "blue_before", "blue_after",
       "quanta", "ew_before", "ew_after", "dew_pct", "band"]].sort_values("index").to_csv(
        "pipeline_output/final/fit_compare.csv", index=False)
    print("%d comparison figures -> %s" % (made, OUT))
    print(d.band.value_counts().to_string())


if __name__ == "__main__":
    main()
