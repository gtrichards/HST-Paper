"""Before-and-after figures for the 1-110 re-review.

BEFORE is the fit GTR adopted on the students' rebinned master, taken straight
from pipeline_output/fit_iterations.  AFTER is the fresh automatic pass on the
regenerated tree, from fit_iterations_v23.  Unlike the earlier fit_compare set
this is not the same masks replayed: the new side has its own mask proposal, so
the pair shows what the re-review actually has to choose between.

Panels are columns and before/after are rows, so corresponding panels sit one
above the other on the same axes, and the whole thing is wide and short.
Filenames sort by index; review_compare.csv carries the size of the change.
"""
import os, glob, json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OLD_T = 'pipeline_output/fit_iterations'
NEW_T = 'pipeline_output/fit_iterations_v23'
# Hand fits have no counterpart on the students' master -- they were made during
# this pass, on the v23 tree before the co-add error revert.  For those objects
# the honest comparison is the same hand masks fit on the tree before and after
# the revert, so BEFORE comes from the snapshot taken just before it.
PRE_T = os.environ.get('HSTICA_PREREVERT',
    '/private/tmp/claude-502/-Users-gtr-Work-projects-hstica/'
    'd417042d-a1d6-4929-8903-4ecb7e5424bd/scratchpad/iter_v23_before_revert')
OUT   = 'pipeline_output/final/review_compare'
LEFT  = (0.006, 0.678)
PANELS = [("C IV  1500-1600 A", LEFT, (0.425, 0.968)),
          ("equivalent width", (0.686, 0.995), (0.400, 0.992)),
          ("full ICA fit", LEFT, (0.008, 0.425))]
TARGET_W, MAX_H, PAD, LABEL = 1800, 980, 10, 26


def _font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()


def adopted(tree, folder):
    """(png, record) for the adopted fit in a folder, or (None, None)."""
    for pat in ("BEST_*", "REJECT_ALL_*"):
        js = sorted(glob.glob(os.path.join(tree, folder, "records", pat + ".json")))
        if js:
            png = js[0].replace("/records/", "/")[:-5] + ".png"
            if os.path.exists(png):
                return png, json.load(open(js[0]))
    return None, None


def crop(im, xs, ys):
    w, h = im.size
    return im.crop((int(xs[0]*w), int(ys[0]*h), int(xs[1]*w), int(ys[1]*h)))


def build(old_png, new_png, caption, out_path, subtitle=None):
    a, b = Image.open(old_png).convert("RGB"), Image.open(new_png).convert("RGB")
    cuts = [(t, crop(a, xs, ys), crop(b, xs, ys)) for t, xs, ys in PANELS]
    asp = [max(p.width/p.height, q.width/q.height) for _, p, q in cuts]
    h = min(int((TARGET_W - PAD*(len(cuts)+1)) / sum(asp)), (MAX_H - 2*LABEL - 3*PAD - 60)//2)
    cols = [(t, p.resize((int(p.width*h/p.height), h), Image.LANCZOS),
                q.resize((int(q.width*h/q.height), h), Image.LANCZOS)) for t, p, q in cuts]
    W = PAD*(len(cols)+1) + sum(max(p.width, q.width) for _, p, q in cols)
    canvas = Image.new("RGB", (W, 58 + 2*(LABEL+h) + 2*PAD), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 8), caption, fill="black", font=_font(24))
    d.text((PAD, 36), subtitle or
           "top row BEFORE (adopted on the master)    "
           "bottom row AFTER (fresh automatic pass on the regenerated tree)",
           fill="#666666", font=_font(17))
    x = PAD
    for t, p, q in cols:
        y = 58
        d.text((x, y), t, fill="#1f4e79", font=_font(17)); canvas.paste(p, (x, y+LABEL))
        y2 = y + LABEL + h + PAD
        d.text((x, y2), t, fill="#7f2704", font=_font(17)); canvas.paste(q, (x, y2+LABEL))
        x += max(p.width, q.width) + PAD
    canvas.save(out_path)


def main():
    os.makedirs(OUT, exist_ok=True)
    dec = pd.read_csv('pipeline_output/final/pass_decisions.csv')
    # pass_decisions.csv has no folder column; the decisions file it is built
    # from does, keyed the same way.
    import csv as _csv
    fold = {}
    for d0 in _csv.DictReader(open('pipeline_output/final/decisions.csv')):
        f = d0['folder']
        if f[:3].isdigit():
            fold[int(f[:3])] = f
    rows, made = [], 0
    # The 1-110 re-review, plus the reopened exclusions that came back with a
    # measurement (3C 273 at 111 is the only one above 110).
    REOPENED = {13, 15, 20, 22, 27, 31, 40, 50, 68, 69, 78, 80, 84, 85, 87, 97, 111, 213, 236}
    keep = (dec['index'] <= 110) | dec['index'].isin(REOPENED)
    for _, r in dec[dec.manmask.isin([2, 3]) & keep].sort_values('index').iterrows():
        folds = [os.path.basename(p) for p in glob.glob(os.path.join(NEW_T, '%03d_*' % r['index']))]
        if not folds: continue
        # Use the folder the decision actually names; the glob alone can land on
        # a different instrument of the same object (001 has COMBINED and FOS).
        decided = fold.get(r['index'], '')
        new_fold = decided if decided in folds else folds[0]
        new_png, new_rec = adopted(NEW_T, new_fold)
        if not new_png: continue
        hand = 'gtr_hand' in os.path.basename(new_png)
        sub = None
        if hand:
            old_png, old_rec = adopted(PRE_T, new_fold)
            sub = ("top row BEFORE (your hand fit, tree before the co-add revert)    "
                   "bottom row AFTER (the same hand masks, reverted tree)")
        else:
            old_png, old_rec = adopted(OLD_T, decided)
        if not old_png: continue
        ob, oe = old_rec['result']['civ_blue'], old_rec['result']['civ_ew']
        nb, ne = new_rec['result']['civ_blue'], new_rec['result']['civ_ew']
        dq = int(round((nb-ob)/69.0)); dew = 100*(ne-oe)/oe if oe else np.nan
        s = new_rec['civ_subcontinuum']
        band = 'A' if (abs(dq) > 3 or abs(dew) > 10) else ('C' if dq == 0 and abs(dew) < 0.1 else 'B')
        cap = ("%03d  row %d  %s   manmask %d    blueshift %.1f -> %.1f (%+d quanta)    "
               "EW %.1f -> %.1f (%+.1f%%)    new veto: %s"
               % (r['index'], r.row_num, r.name_pub, int(r.manmask), ob, nb, dq, oe, ne, dew,
                  'pass' if s['ok'] else 'REJECT %d px' % s['n_px']))
        build(old_png, new_png, cap,
              # Named by index alone: GTR walks the list in index order, and a
              # band prefix that changes between rebuilds moves the file.
              os.path.join(OUT, '%03d_%s.png' % (r['index'], str(r.common_name).replace('/', '_'))),
              subtitle=sub)
        rows.append(dict(index=r['index'], row_num=r.row_num, name=r.name_pub, manmask=int(r.manmask),
                         old_blue=ob, new_blue=nb, quanta=dq, old_ew=oe, new_ew=ne, dew_pct=dew,
                         new_veto='pass' if s['ok'] else 'REJECT(%d)' % s['n_px'], band=band,
                         old_rec=os.path.basename(old_png)[:-4], new_rec=os.path.basename(new_png)[:-4]))
        made += 1
    d = pd.DataFrame(rows)
    d.to_csv('pipeline_output/final/review_compare.csv', index=False)
    print('%d comparisons -> %s' % (made, OUT))
    print(d.band.value_counts().to_string())


if __name__ == '__main__':
    main()
