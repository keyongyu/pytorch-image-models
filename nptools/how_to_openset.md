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
- **`--amp --channels-last` are on by default.** fp16 mixed precision keeps fp32 master weights and
  loss-scales the gradients, so accuracy is unaffected, and the export is untouched (`openset.py`
  rebuilds the model and loads fp32 weights). Disable with `--no-amp`; if the ArcFace margin logits
  ever produce NaN losses, pass `--amp-dtype bfloat16` through.
- **The split is pre-resized before training, always.** `nptools/resize_dataset.py` writes a sibling
  `<data-dir>_<2N>` tree (squashed to `2N × 2N`, where N is `--resize`, default 224) and training
  reads that. This is not a shortcut: `npaug` normalises every sample to a `2*img_size` canvas, so
  nothing above it survives — and decoding full-resolution sources each epoch cost **10.5 ms of the
  14.3 ms** per-sample budget, more than the entire augmentation. Measured: the 16-worker loader goes
  941.8 → 1820.0 img/s and the split shrinks 1705 MB → 879 MB, built in ~8 s.
  Originals are never modified, `imbaldup_*` symlinks from step 6 are relinked (not duplicated), and
  `nobg_*.png` cutouts keep PNG + alpha. Re-runs are mtime-checked, so only changed images are rewritten.
- **`--resize N`** (default 224) is the single size knob: it is the model input size passed to
  `train.py --img-size`, it fixes the pre-resize target at `2N`, and it fills in the sizes in the
  export commands printed at the end of the run. Keep it consistent across training and export.
- **`--workers`** is derived as 2/3 of the **physical** cores, not `nproc`: this augmentation is
  compute-bound, so SMT siblings only contend. Measured on a 24C/48T box, 32 workers was *slower*
  than 16 (418 vs 576 img/s). Override with `WORKERS=N` in the environment.
- `--new` starts from the pretrained backbone; without it the script resumes the latest
  `last.pth.tar` under the output dir.
- Checkpoints land in `<DATA>/output/<timestamp>/`; use `model_best.pth.tar` next. **When training
  finishes the script prints the exact `openset.py` export command and the `pnnx` follow-up**, with
  the run directory, class-map, `--data-dir` and sizes already filled in — steps 8 and 9 below
  explain what those commands do, but you can copy them straight from the run output.

---

## 8. Build & export the open-set model

`openset.py` loads the checkpoint, builds an L2-normalized **prototype** (mean pre-logits feature)
per class, resolves a **single constant reject threshold** shared by every class, and exports a
deployable model.

```bash
# optional first: inspect the false-reject / false-accept tradeoff without writing anything --
# calibration always runs, so simply omit the --export-* flags
uv run python nptools/openset.py \
    --data-dir <DATA> --class-map <DATA>/class_84.txt \
    --ck <DATA>/output/<run>/model_best.pth.tar

# export both formats in ONE run (shared prototype pass, and both artifacts then carry the
# same prototypes and the same calibrated threshold -- a second run would recalibrate)
uv run python nptools/openset.py \
    --data-dir <DATA> \
    --class-map <DATA>/class_84.txt \
    --ck <DATA>/output/<run>/model_best.pth.tar \
    --img-size 224 \
    --no-argmin \
    --export-pt <DATA>/output/<run>/openset.pt \
    --export-onnx <DATA>/output/<run>/openset.onnx
```

`train_posm.sh` prints this command with every path filled in when training ends, so prefer copying
it from there over retyping it.

Two things that are easy to get wrong and do **not** fail loudly:
- **`--data-dir` here is not `train.py`'s.** `openset.py` wants the directory that *contains*
  `train/`; if you pointed training straight at the class-folder root, pass its **parent**. Point it
  at the **originals**, not the pre-resized copy from step 7 — prototypes and the threshold must come
  from the images a deployed client will actually see.
- **`--img-size` must match training.** `openset.py` defaults to 224 independently of the wrapper, so
  training at another size and exporting without it extracts prototypes at one scale while the client
  feeds another. The size is baked into the graph and recorded in the sidecar, so accuracy just
  quietly drops.

What happens:
- **Prototypes** come from clean `train/<class>/` images (class-map only; `others` never read).
- **One prototype per class-map entry.** A class with no usable images gets a **zero** prototype
  rather than being dropped, so class indices always match the class-map file; those columns score a
  constant `dist = 1.0` and can never match. Their names land in `meta.json` as `empty_classes`.
  (Dropping them, as older exports did, silently shifted every index after the first gap.)
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
uv pip install pnnx ncnn                                   # once (not a project dependency)
uv run pnnx <DATA>/output/<run>/openset.pt inputshape=[1,3,224,224]
#   → openset.ncnn.param + openset.ncnn.bin   (+ openset_ncnn.py and the openset.pnnx.* intermediates,
#     all written NEXT TO the .pt, so they land in the run dir beside the checkpoint)
```

`train_posm.sh` prints this line too, with the path and `inputshape` already matching `--resize`.

- **`inputshape` must match the exported `--img-size`.** pnnx traces at that shape and
  constant-folds `Conv2dSame`'s dynamic padding against it, so a wrong value bakes wrong padding
  into the `.param` instead of failing.
- **Export with `--no-argmin` (step 8) for a stock ncnn wheel.** That graph is just
  `Normalize → matmul → subtract` and converts/runs as-is; the client does the `argmin` + reject.
  The default (in-graph argmin) export emits ops a stock wheel can't run and needs a custom layer.
- Verified on a `--no-argmin` export: the resulting `.param` is 76 layers of `Convolution`,
  `ConvolutionDepthWise`, `BinaryOp`, `Split`, `Pooling`, `Reshape`, `Flatten`, **one** `Normalize`
  and **one** `InnerProduct` — zero `ArgMin`/`ArgMax`/`Gather`/`Crop`. ncnn output matches
  TorchScript to ~4e-3 on `dists` (fp32 kernel differences, three orders below the 0.46 threshold),
  with identical decisions on every image tested.
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
accepted; if you kept an `others/` set (step 3), confirm its items are rejected. Re-run without
the `--export-*` flags to inspect the tradeoff; every export recalibrates, so re-exporting is how
you pick up a corrected value after fixing data.

---

## 11. Enroll a new product later (no retraining)

Adding a product needs no retrain, because the decision uses the **embedding**, not the classifier
head: a new class needs a prototype, not a new output neuron.

1. add its cutouts under `train/<newclass>/` (step 2),
2. add its name to the class-map (step 4) — **append**, don't reorder: existing indices must not move,
3. rebuild/export prototypes (step 8), pointing `--checkpoint` at the **same** checkpoint as before.

The new class gets a prototype from a handful of images and inherits the shared threshold. The
class-map may now be longer than the checkpoint's classifier; that is fine, `openset.py` sizes the
head from the checkpoint and prints a note, because the head is a training artifact it never reads.
(Before that fix this step failed outright with `size mismatch for classifier.weight`.)

Then **verify the new class explicitly**, because nothing above measures it:

- do its own images land close to its new prototype? The export log prints
  `<class>: N imgs, mean=…, max=…` — compare that mean against the other classes (on posmlvx the
  trained classes sit at ~0.005–0.013); a visibly larger mean means the embedding does not separate
  this product, and *that* is when to retrain the backbone.
- is it distinct from its look-alikes? A near-duplicate SKU shows up in `outlier/outlier.txt` as a
  `nearest_type` pointing at the new class, or vice versa.

Re-running the calibration report (the same command without `--export-*`) is cheap and worth doing,
since the threshold should be re-measured whenever the class set changes. Note what it does and does
not tell you: its negatives come from **leave-one-class-out**, i.e. scoring a class's images with its
own prototype masked, which measures *rejecting a product you have not enrolled* — the situation this
step removes. It does not measure whether the newly enrolled class is correctly **accepted**; only
the two checks above do.

---

## TL;DR

```bash
# 2. video -> cutouts (per class)
uv run python nptools/extract_object.py --video-folder <DATA>/<class>videos --dest-dir <DATA>/train/<class> --flat-out
# 4. class map: one class per line (omit 'others')
# 6. reduce class imbalance (symlink-oversample minority classes)
python -m nptools.reduce_imbalance --data-dir <DATA> --split train
# 7. train ArcFace (arcface + per-class-acc + amp on by default; split auto-resized to 2*224)
sh nptools/train_posm.sh --data-dir <DATA> --class-map <DATA>/class_84.txt --new
#    ...and copy the two commands it prints when it finishes, which are steps 8 and 9 pre-filled:
# 8. build + export open-set model (threshold is calibrated automatically)
uv run python nptools/openset.py --data-dir <DATA> --class-map <DATA>/class_84.txt \
    --ck <DATA>/output/<run>/model_best.pth.tar --img-size 224 --no-argmin \
    --export-pt <DATA>/output/<run>/openset.pt --export-onnx <DATA>/output/<run>/openset.onnx
# 9. convert to ncnn (outputs land next to the .pt)
uv run pnnx <DATA>/output/<run>/openset.pt inputshape=[1,3,224,224]   # → openset.ncnn.param + .bin
```
