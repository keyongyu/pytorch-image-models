# How to build an open-set product model — end to end

This walks the full pipeline: **product videos → object cutouts → dataset + class-map → train an
ArcFace classifier → build & export the open-set model.** It ties together three tools; see their
own docs for depth:

- cutouts from video — [`extract_object.md`](extract_object.md) (`nptools/extract_object.py`)
- training wrapper — `nptools/train_posm.sh` (wraps `train.py`)
- open-set model + export — [`openset.md`](openset.md) (`nptools/openset.py`)

```
 videos ─► extract_object.py ─► train/<class>/nobg_*.png ─┐
                                                          ├─► train_posm.sh (ArcFace) ─► checkpoint
 class map (class_84.txt) ────────────────────────────────┘                                │
                                                                                           ▼
                                            openset.py (prototypes + per-class thresholds) ─► .pt/.onnx (+meta.json) ─► pnnx ─► .ncnn.param/.bin
```

Conventions below use `<DATA>` for the dataset root (e.g. `posmlv`) and `tf_efficientnet_lite0` as
the backbone (what `train_posm.sh` uses).

---

## 1. Collect one video per product

Shoot a short clip of each product (SKU / standee), slowly orbiting so the object is seen from
several angles. One video = one class. Put them in a folder, e.g. `<DATA>/<class>videos/`.

The video **file stem becomes the class name** in the extracted filenames, so name videos after the
class (or pass `--stem`/rename later). Keep clips per class in their own folder.

---

## 2. Extract object cutouts from the videos

`extract_object.py` turns each frame into a background-removed cutout (keeps thin parts like a
standee foot) and runs QC to drop failed frames. Write straight into the class's train folder:

```bash
uv run python nptools/extract_object.py \
    --video-folder <DATA>/<class>videos \
    --dest-dir     <DATA>/train/<class> \
    --flat-out                                   # flat output, rejects discarded
```

Per kept frame you get an aligned pair in `<DATA>/train/<class>/`:
- `nobg_<video>_<frame>_.png` — background removed (transparent),
- `<video>_<frame>_.jpg` — same crop, **original background kept**.

**Which to keep for training?** For video-sourced classes the frames all share one background, so a
plain photo would leak that background as a shortcut. **Use the `nobg_*.png`** — under `--nobg`
(step 6) they are composited onto random backgrounds (bg-swap), which breaks the leak. If you don't
want the background-kept JPGs trained on, delete them (or keep only the PNGs):

```bash
find <DATA>/train/<class> -name '*.jpg' -not -name 'nobg_*' -delete   # optional: drop bg-kept jpgs
```

Repeat step 2 for every product/class. Tuning (`--margin`, QC thresholds, etc.) is in
[`extract_object.md`](extract_object.md).

---

## 3. Add an `others` / unknown bucket (recommended)

Create `<DATA>/train/others/` with miscellaneous non-target images. **Do NOT list it in the
class-map** — it will be:
- **excluded from training** (the reader drops folders not in the class-map),
- **excluded from prototypes** (openset builds prototypes only for class-map names),
- available as the **negative / unknown set** for calibrating and testing the reject threshold.

(Training `others` as an ArcFace class is a bad idea — it's heterogeneous and fights ArcFace's
compactness objective. Keep it out of the map.)

---

## 4. Write the class-map file

One class name per line; **line order = class index**, and the **non-empty line count =
`num_classes`**. List every real product folder; **omit `others`**:

```
# <DATA>/class_84.txt
posm_16
posm_18
...
```

`train_posm.sh` derives `--num-classes` from this file automatically. The same file drives
`openset.py` (which classes get a prototype, and in what order).

---

## 5. (optional) Sanity-check the augmentation

Before a long train, dump what the augmentation pipeline actually produces — no training, ≤4 epochs,
grouped by class:

```bash
sh nptools/train_posm.sh --test-npaug-dir /tmp/aug_check --test-npaug-class posm_18
```

Eyeball `/tmp/aug_check/posm_18/` — confirm the object (and its foot) survives crop/erase. See
`--test-npaug-*` in `train_posm.sh`.

---

## 6. Train the ArcFace classifier

`train_posm.sh` wraps `train.py` with the posm defaults (`--nobg` bg-swap, `--crop-mode=squash`,
`tf_efficientnet_lite0`, etc.). Add `--arcface` for open-set-friendly embeddings:

```bash
sh nptools/train_posm.sh \
    --data-dir <DATA> \
    --class-map <DATA>/class_84.txt \
    --new \                       # fresh from pretrained backbone (omit to resume last.pth.tar)
    --arcface --per-class-acc     # forwarded through to train.py
```

- `--arcface` trains a normalized-feature / angular-margin head → **compact, well-separated
  clusters** (tune with `--arcface-s`, `--arcface-m`). The saved checkpoint is a plain timm
  `state_dict`; the ArcFace head is training-only.
- `--new` starts from the pretrained backbone; without it the script resumes the latest
  `last.pth.tar` under the output dir.
- Checkpoints land in `<DATA>/output/<timestamp>/`; use `model_best.pth.tar` next.

---

## 7. Build & export the open-set model

`openset.py` loads the checkpoint, builds an L2-normalized **prototype** (mean pre-logits feature)
per class, computes a **per-class reject threshold**, and exports a deployable model.

```bash
uv run python nptools/openset.py \
    --data-dir <DATA> \
    --class-map <DATA>/class_84.txt \
    --ck <DATA>/output/<run>/model_best.pth.tar \
    --quantile 0.97 \             # per-class threshold quantile (default)
    --aug \                       # calibrate thresholds on npaug-augmented distances (default on)
    --no-argmin --export-pt nptools/openset.pt   # stock-ncnn-friendly export (or --export-onnx)
```

What happens:
- **Prototypes** come from clean `train/<class>/` images (class-map only; `others` never read).
- **Thresholds** = the `--quantile` of each class's in-distribution distances; with `--aug` those
  distances are measured on npaug-augmented views (prototype stays clean), so the threshold reflects
  real-world variation instead of the optimistic clean-train spread.
- Export writes `openset.pt` (or `.onnx`) **plus** `openset.pt.meta.json` (class order, per-class
  thresholds, img_size, mean/std). After export it **self-verifies** by predicting through the
  exported model.

Threshold theory, per-class vs global, `others`-as-negatives, and ncnn conversion are all in
[`openset.md`](openset.md).

---

## 8. Convert to ncnn with pnnx

Turn the exported `.pt` (or `.onnx`) into ncnn files for on-device deployment. Install the tools
once, then run pnnx with the model's input shape:

```bash
uv pip install pnnx ncnn                                   # once
cd nptools && pnnx openset.pt inputshape=[1,3,224,224]     # or: pnnx openset.onnx inputshape=[1,3,224,224]
#   → openset.ncnn.param + openset.ncnn.bin   (+ openset.ncnn.py, and openset.pt.meta.json from step 7)
```

- **Export with `--no-argmin` (step 7) for a stock ncnn wheel.** That graph is just
  `Normalize → matmul → subtract` and converts/runs as-is; the client does the `argmin` + reject.
  The default (in-graph argmin) export emits ops a stock wheel can't run and needs a custom layer.
- The TorchScript route (`.pt`) usually yields a leaner ncnn graph than the ONNX route; both give the
  same model. Useful pnnx args: `fp16=1` (default), `optlevel=2`.
- Blob names (input `in0`, outputs `out0`=`dists`/`out1`=`margins`) are in the top of
  `openset.ncnn.param`; carry `openset.pt.meta.json` alongside for class names + thresholds.

Full ncnn client code and the custom-layer route are in [`openset.md`](openset.md) (§4–§6).

---

## 9. Verify / predict

```bash
# predict via the exported model (labels an image, or scans val/ + test/)
uv run python nptools/openset.py --load-onnx nptools/openset.onnx --image sample.jpg
```

Decision at inference: nearest prototype by cosine distance; if `dist > threshold[that class]` →
**unknown / new product**, else the matched class. Sanity-check that real products in `val/`/`test/`
are accepted and `others/` items are rejected; adjust `--quantile` if needed and re-export.

---

## 10. Enroll a new product later (no retraining)

The ArcFace embedding generalizes, so adding a product usually needs **no retrain**:
1. add its cutouts under `train/<newclass>/` (step 2),
2. add its name to the class-map (step 4),
3. rebuild/export prototypes (step 7).

The new class gets a prototype + threshold from a handful of images; retrain the backbone only if
accuracy on the new class is poor.

---

## TL;DR

```bash
# 2. video -> cutouts (per class)
uv run python nptools/extract_object.py --video-folder <DATA>/<class>videos --dest-dir <DATA>/train/<class> --flat-out
# 4. class map: one class per line (omit 'others')
# 6. train ArcFace
sh nptools/train_posm.sh --data-dir <DATA> --class-map <DATA>/class_84.txt --new --arcface
# 7. build + export open-set model
uv run python nptools/openset.py --data-dir <DATA> --class-map <DATA>/class_84.txt \
    --ck <DATA>/output/<run>/model_best.pth.tar --no-argmin --export-pt nptools/openset.pt
# 8. convert to ncnn
cd nptools && pnnx openset.pt inputshape=[1,3,224,224]      # → openset.ncnn.param + .bin
```
