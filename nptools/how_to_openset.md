# How to build an open-set product model — end to end

This walks the full pipeline: **product videos → object cutouts → dataset + class-map → train an
ArcFace classifier → build & export the open-set model.** It ties together three tools; see their
own docs for depth:

- cutouts from video — [`extract_object.md`](extract_object.md) (`nptools/extract_object.py`)
- training wrapper — `nptools/train_posm.sh` (wraps `train.py`)
- open-set model + export — [`openset.md`](openset.md) (`nptools/openset.py`)

```
 per-class videos
      │
      ▼
 extract_object.py   ─►  train/<class>/nobg_*.png (+ .jpg)
      │
      ▼
 reduce_imbalance.py   (balance train/ by symlink-oversampling)
      │
      ▼
 train_posm.sh   (ArcFace; --nobg bg-swap)   ─►  checkpoint   ◄─ needs: class map, bg photos (NOBG_BG_DIR)
      │
      ▼
 openset.py   (prototypes + calibrated threshold)   ─►  .pt / .onnx (+ meta.json)   ◄─ needs: class map
      │
      ▼
 pnnx   ─►  .ncnn.param / .ncnn.bin
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
(step 7) they are composited onto random backgrounds (bg-swap), which breaks the leak. If you don't
want the background-kept JPGs trained on, delete them (or keep only the PNGs):

```bash
find <DATA>/train/<class> -name '*.jpg' -not -name 'nobg_*' -delete   # optional: drop bg-kept jpgs
```

Repeat step 2 for every product/class. Tuning (`--margin`, QC thresholds, etc.) is in
[`extract_object.md`](extract_object.md).

---

## 3. (softmax only) Add an `others` / unknown bucket

**Recommended for a plain softmax classifier, NOT for this ArcFace open-set workflow.**

- **Softmax classification:** add an explicit `others` / background class so the network has a place
  to send unknowns (the classic background-class approach to open-set with softmax). There it earns
  its keep.
- **ArcFace open-set (this pipeline):** **no longer recommended.** Reject is decided by a single
  **cosine-distance threshold** measured against leave-one-class-out negatives drawn from your own
  labelled data (step 8), so unknowns are handled by distance-to-prototype — an `others` bucket is
  not needed for the decision, and calibration does not need one either. You must not train it as a
  class anyway: `others` is heterogeneous and fights ArcFace's compactness objective, so a single
  `others` prototype is meaningless.

If you already have an `others/` folder (e.g. for evaluation), it does no harm as long as you **keep
it out of the class-map** — the reader drops folders not in the map, and `openset.py` builds
prototypes only for class-map names. It can still be handy as a **negative set to spot-check** the
reject threshold (step 10), but it is not part of building the model.

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

## 6. Reduce class imbalance (oversample minority classes)

Classes built from video have very different image counts (a 142-frame clip vs a 51-frame one).
`reduce_imbalance.py` evens the training signal by **symlink-oversampling**: every
class with fewer than `--ratio` × the largest class's image count is topped up to that count with
symlinks to its own images (cycled). Symlinks keep disk usage flat, and because training re-augments
each epoch, a symlinked duplicate behaves like a fresh sample — balancing per-class gradients (and
ArcFace prototypes) without memorization.

```bash
# preview the plan
python -m nptools.reduce_imbalance --data-dir <DATA> --split train --dry-run
# apply (default --ratio 0.5 = top minority classes up to half the largest class's count)
python -m nptools.reduce_imbalance --data-dir <DATA> --split train
```

- **Idempotent:** each run first removes its own previous symlinks (marked `imbaldup_`) and recounts
  the real images — safe to re-run after adding more cutouts; it won't stack duplicates.
- Only touches `train/`, and the symlinks it creates are **skipped when composing the validation
  set**, so eval isn't polluted with duplicates.

---

## 7. Train the ArcFace classifier

`train_posm.sh` wraps `train.py` with the posm defaults (`--nobg` bg-swap, `--crop-mode=squash`,
`tf_efficientnet_lite0`, etc.). It also passes **`--arcface --per-class-acc` by default**, since
the open-set pipeline needs ArcFace embeddings:

**Background folder (required for `--nobg`).** The bg-swap composites each `nobg_*.png` cutout onto a
random photo from a **background-image folder** every epoch — this is what breaks the shared-video
background leak. Point the `NOBG_BG_DIR` env var at a folder of varied scene photos (any
jpg/jpeg/png, searched recursively). `train_posm.sh` sets it near the top:

```sh
export NOBG_BG_DIR=/path/to/bg_photos   # edit this in nptools/train_posm.sh
```

`train.py` **fails fast** if `--nobg` is set but `NOBG_BG_DIR` is unset/empty or contains no images,
so set it before training. Use general background scenes (shelves, streets, rooms) — not more
product images.

```bash
sh nptools/train_posm.sh \
    --data-dir <DATA> \
    --class-map <DATA>/class_84.txt \
    --new                         # fresh from pretrained backbone (omit to resume last.pth.tar)
                                  # --arcface --per-class-acc are added automatically
```

- **`--arcface` is on by default** — a normalized-feature / angular-margin head giving **compact,
  well-separated clusters**, which is what makes the prototypes in step 8 usable (tune with
  `--arcface-s`, `--arcface-m`; those still pass straight through). The saved checkpoint is a plain
  timm `state_dict`; the ArcFace head is training-only. Turn it off with `--no-arcface`.
- **`--per-class-acc` is on by default** too — per-class top-1, which is how look-alike classes get
  spotted. Turn it off with `--no-per-class-acc`.
- `--new` starts from the pretrained backbone; without it the script resumes the latest
  `last.pth.tar` under the output dir.
- Checkpoints land in `<DATA>/output/<timestamp>/`; use `model_best.pth.tar` next.

---

## 8. Build & export the open-set model

`openset.py` loads the checkpoint, builds an L2-normalized **prototype** (mean pre-logits feature)
per class, resolves a **single constant reject threshold** shared by every class, and exports a
deployable model.

```bash
# optional first: inspect the false-reject / false-accept tradeoff without exporting
uv run python nptools/openset.py \
    --data-dir <DATA> --class-map <DATA>/class_84.txt \
    --ck <DATA>/output/<run>/model_best.pth.tar --calibrate

# export (the threshold is calibrated automatically -- there is nothing to pass)
uv run python nptools/openset.py \
    --data-dir <DATA> \
    --class-map <DATA>/class_84.txt \
    --ck <DATA>/output/<run>/model_best.pth.tar \
    --no-argmin --export-pt nptools/openset.pt   # stock-ncnn-friendly export (or --export-onnx)
```

What happens:
- **Prototypes** come from clean `train/<class>/` images (class-map only; `others` never read).
- **One constant threshold** is applied to every class. Per-class thresholds were removed after
  measurement: they overfit 3.8× worse on held-out data, and the old quantile values were tight
  enough to falsely reject ~13–15% of real product images.
- **The threshold is always calibrated** during the run, sharing the feature pass with prototype
  building — it cannot be pinned, so it can never go stale against the checkpoint. Negatives come
  from leave-one-class-out on your own labelled data, so no curated negative set is required. It is
  seeded, so the same checkpoint always exports the same artifact. A floor applies
  (`--min-threshold`, default 0.2); a suggestion below it is clamped up and warns, since that
  indicates a data problem rather than a genuinely tiny threshold.
- Export writes `openset.pt` (or `.onnx`) **plus** `openset.pt.meta.json` (class order, the scalar
  threshold, img_size, mean/std, resize filter, and the measured FRR/FAR that justified the
  value). After export it **self-verifies** by predicting through the exported model.

Why one threshold rather than per-class, how calibration works, the client's preprocessing contract,
and ncnn conversion are all in [`openset.md`](openset.md).

---

## 9. Convert to ncnn with pnnx

Turn the exported `.pt` (or `.onnx`) into ncnn files for on-device deployment. Install the tools
once, then run pnnx with the model's input shape:

```bash
uv pip install pnnx ncnn                                   # once
cd nptools && pnnx openset.pt inputshape=[1,3,224,224]     # or: pnnx openset.onnx inputshape=[1,3,224,224]
#   → openset.ncnn.param + openset.ncnn.bin   (+ openset.ncnn.py, and openset.pt.meta.json from step 8)
```

- **Export with `--no-argmin` (step 8) for a stock ncnn wheel.** That graph is just
  `Normalize → matmul → subtract` and converts/runs as-is; the client does the `argmin` + reject.
  The default (in-graph argmin) export emits ops a stock wheel can't run and needs a custom layer.
- The TorchScript route (`.pt`) usually yields a leaner ncnn graph than the ONNX route; both give the
  same model. Useful pnnx args: `fp16=1` (default), `optlevel=2`.
- Blob names (input `in0`, outputs `out0`=`dists`/`out1`=`margins`) are in the top of
  `openset.ncnn.param`; carry `openset.pt.meta.json` alongside for class names + thresholds.

Full ncnn client code and the custom-layer route are in [`openset.md`](openset.md) (§4–§6).

---

## 10. Verify / predict

```bash
# predict via the exported model (labels an image, or scans val/ + test/)
uv run python nptools/openset.py --load-onnx nptools/openset.onnx --image sample.jpg
```

Decision at inference: nearest prototype by cosine distance; if `dist > threshold` → **unknown /
new product**, else the matched class. Sanity-check that real products in `val/`/`test/` are
accepted; if you kept an `others/` set (step 3), confirm its items are rejected. Re-run
`--calibrate` to inspect the tradeoff; every export recalibrates, so re-exporting is how you pick
up a corrected value after fixing data.

---

## 11. Enroll a new product later (no retraining)

The ArcFace embedding generalizes, so adding a product usually needs **no retrain**:
1. add its cutouts under `train/<newclass>/` (step 2),
2. add its name to the class-map (step 4),
3. rebuild/export prototypes (step 8).

The new class gets a prototype from a handful of images and inherits the shared threshold; retrain
the backbone only if accuracy on the new class is poor. Re-running `--calibrate` after enrolling is
cheap and worthwhile — leave-one-class-out is exactly the "a product I have not enrolled yet"
scenario, so it measures the case this step creates.

---

## TL;DR

```bash
# 2. video -> cutouts (per class)
uv run python nptools/extract_object.py --video-folder <DATA>/<class>videos --dest-dir <DATA>/train/<class> --flat-out
# 4. class map: one class per line (omit 'others')
# 6. reduce class imbalance (symlink-oversample minority classes)
python -m nptools.reduce_imbalance --data-dir <DATA> --split train
# 7. train ArcFace
sh nptools/train_posm.sh --data-dir <DATA> --class-map <DATA>/class_84.txt --new   # arcface on by default
# 8. build + export open-set model (threshold is calibrated automatically)
uv run python nptools/openset.py --data-dir <DATA> --class-map <DATA>/class_84.txt \
    --ck <DATA>/output/<run>/model_best.pth.tar \
    --no-argmin --export-pt nptools/openset.pt
# 9. convert to ncnn
cd nptools && pnnx openset.pt inputshape=[1,3,224,224]      # → openset.ncnn.param + .bin
```
