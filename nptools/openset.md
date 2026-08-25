# Open-set recognition (openset.py) → ONNX / ncnn

`nptools/openset.py` turns a normal **closed-set classifier** (timm `tf_efficientnet_lite0`, trained
on N known products) into an **open-set** model that can also say *"this is not any product I
know"* (reject unknown / newly-introduced SKUs). It then exports that model to ONNX and/or
TorchScript for deployment with **ncnn**.

---

## 1. Concept & flow

A plain softmax classifier is **closed-set**: it always forces an input into one of its N trained
classes, even a brand-new product it has never seen. Open-set recognition adds a *reject* option.

How it works here (feature-prototype / metric approach):

```
      ┌──────────── build once, from training data (classes from class-map) ─────┐
      │  for each class: mean of its pre-logits features → prototype (L2-norm)   │
      │  ONE constant reject threshold, always calibrated (floor 0.2)            │
      └──────────────────────────────────────────────────────────────────────────┘

  image ─► backbone ─► pre-logits feature (1280-d) ─► L2 normalize
                                                          │
                                                          ▼
                        cosine distance to every class prototype  →  dists[C]
                                                          │
                             best = argmin(dists)         │
                                                          ▼
                    margin = best_dist − threshold              (one shared constant)
                        margin < 0  →  KNOWN   → class = class_names[best]
                        margin > 0  →  UNKNOWN → reject (new / unseen product)
```

Key idea: a known product lands **close** to its class prototype in feature space; an unknown
product lands **far from all** prototypes → large `best_dist` → positive margin → rejected.
The classifier logits are *not* used for the decision — the **pre-logits feature** is.

**Why ONE constant threshold, not per-class.** Earlier versions gave each class its own threshold,
set to the `--quantile` of that class's own training distances. Measurement retired that:

- **Per-class overfits.** Fitting a threshold per class on half the data and scoring the other half
  gives **3.8× more errors** than a single constant (0.870% vs 0.227%). 15 of 70 classes have fewer
  than 10 images to fit from — two classes have 2. Even fitting *and* scoring on the same data
  (pure overfit, unachievable) beats the constant by only **0.030pp**. There is no headroom.
- **The old quantile values were far too tight.** They had a median of 0.017, which measures out at
  a **~13–15% false-reject rate** on real product images; 13 of 70 classes sat below 0.005. Over
  that same range the false-accept rate barely moved (0.05% → 0.18%), so the tightness bought
  nothing.
- **The cause is the data, not the classes.** Each class comes from one video, so its crops are
  near-duplicate frames. Its distance spread measures *how that clip happened to be shot*, not how
  the product varies — an artifact being baked in as if it were a property of the product.

Geometrically there is nothing for a per-class threshold to exploit: ArcFace drives prototypes to
near-orthogonality (nearest-competitor distance min 0.89, median 0.96), positives sit at ~0.007 and
negatives at ~0.93. Every class needs the same thing — a cut somewhere in that large empty gap.
It is measured on every run, not guessed (§2.2).

---

## 2. openset.py — usage

Prototypes are built from `<data-dir>/train/<class>/*`, **driven by the class-map file** (the
class-map defines which classes to use, in what order, and — via its line count — `num_classes`).

### CLI arguments

| arg | required | meaning |
|---|---|---|
| `--class-map PATH` | **yes** (PyTorch/export path) | class-map file, one class name per line. Its **line count = `num_classes`**, and its **order = class index order**. Not needed for `--load-onnx`. |
| `--checkpoint PATH` / `--ck` | **yes** (PyTorch/export path) | the trained checkpoint to load (`--ck` is a short alias) |
| `--data-dir DIR` | no (default posmlv) | dataset root with `train/`, `val/`, `test/` subfolders |
| `--img-size N` | no (default 224) | square input size. **Must match what was trained** — this default is independent of `train_posm.sh`, and a mismatch does not error: the size is baked into the exported graph and recorded in the sidecar, so prototypes get extracted at one scale while the client feeds another and accuracy quietly drops. `train_posm.sh` fills this in for you in the export command it prints. |
| `--min-threshold M` | no (default 0.2) | floor on the calibrated threshold. A suggestion below it is clamped up and warned about, since a tiny threshold rejects genuine products (0.02 costs ~10% false-reject on posmlv) while barely reducing false-accepts. |
| `--aug` / `--no-aug` | no (default `--aug`) | during calibration, probe with **npaug-augmented** views so the in-distribution spread reflects real variation instead of near-duplicate video frames (prototypes stay clean); `--no-aug` probes the clean crops with leave-one-image-out instead |
| `--aug-views N` | no (default 3) | augmented probe views per image during calibration |
| `--aug-max-per-class N` | no (default 200) | cap on probe images per class during calibration; `0` = use all |
| `--cpu-aug` | no (default off) | use CPU npaug (albumentations) instead of GPU torchvision for the aug probe — faithful to training augmentation, much slower |
| `--eval-csv [PATH]` | no | score the **training crops** per class twice -- over everything (`all`) and again with the outlier crops dropped (`inliers`) -- writing `(class type, recall rate, precision rate, eval type)` to PATH (default `<checkpoint dir>/eval_rounds.csv`). Same prototypes and threshold in both rounds, so the delta is what those crops cost, per class. Reuses the export's feature pass (~0.2s). **Fit, not generalization**: the scored crops are the ones that built the prototypes, so recall sits near 100% by construction and the signal is *which* classes move between the rounds. |
| `--copy-inliers` | no (default off) | while building prototypes, also copy the **good inlier** crops (cosine dist ≤ 0.5 to their class prototype) into `<data-dir>/inlier/<class>/` — a cleaned training set (see §2.1) |
| `--gpu-predict` | no (default **CPU**) | run **prediction/verification** on GPU (ONNX `CUDAExecutionProvider`, TorchScript + in-memory model on `cuda`). Default is CPU so verification mirrors the ncnn deployment target. **Prototype building always uses the GPU** regardless of this flag; it only affects the predict/verify pass. |
| `--export-onnx [PATH]` | no | export ONNX (+ `<PATH>.meta.json`); default `<checkpoint>_openset.onnx` |
| `--export-pt [PATH]` | no | export TorchScript (+ `<PATH>.meta.json`); default `<checkpoint>_openset.pt` |
| `--image PATH` | no | classify a single image (else scans `val/` + `test/`) |
| `--load-onnx PATH` | no | predict via an already-exported ONNX; class names and output format (`no_argmin`) are read from `<PATH>.meta.json` — the CLI `--no-argmin` flag is ignored on this path |
| `--no-argmin` | no | export the **client-side-argmin** variant: the graph emits per-class `dists`/`margins` vectors (no `argmin`/`gather` in the model) and the argmin + reject move to the client. Converts and runs on a **stock ncnn wheel**. Default keeps the **in-graph argmin** scalar outputs — the graph runs `torch.argmin` + a dynamic-index `Crop` (gather), which a stock ncnn wheel **can't** run (ncnn has **no `ArgMin` layer** and no `Gather`), so that path needs a custom layer. |

```bash
# Classify val/ + test/ images with the PyTorch prototypes (no export).
# Still calibrates first — that happens on every run that builds prototypes.
uv run python nptools/openset.py \
    --data-dir posmlv --class-map posmlv/class_84.txt --checkpoint <run>/model_best.pth.tar

# Export the open-set model (then it self-verifies by predicting THROUGH the exported model)
uv run python nptools/openset.py \
    --data-dir posmlv --class-map posmlv/class_84.txt --checkpoint <run>/model_best.pth.tar \
    --export-onnx model.onnx        # or --export-pt model.pt

# Predict through an already-exported ONNX model (CPU), no prototype rebuild
uv run python nptools/openset.py --load-onnx model.onnx --image x.jpg
```

Notes:
- **Class-map drives prototype building, one prototype per entry.** Only classes listed in the
  class-map are used, in that order, and the exported model has **exactly as many columns as the
  class-map has entries** — so a class index means the same thing in the class-map file, the
  exported matrix, and the client. A class with **< 10 images** is warned (prototype may be
  unreliable) but still used. Folders **not** in the class-map are ignored.
- **Classes with no usable images get a zero prototype**, not a dropped column. "No usable images"
  means the folder is missing, the folder is empty, or every image was an outlier; each case is
  warned about, and the names are listed in `meta.json` as `empty_classes`.
  Zero is the *correct* sentinel here, not just a convenient one: the pre-logits feature is
  post-ReLU6 and therefore non-negative, so cosine similarity lies in [0, 1] and cosine distance
  in [0, 1] (measured max 1.0000 over 1800 images). A zero row scores `sims = 0`, i.e.
  `dist == 1.0` — the **maximum attainable distance** — so it can never strictly win the argmin
  against a class that has data, and if it ever tied at 1.0 the margin (`1.0 − threshold`) is
  positive, so the verdict is `unknown` anyway. It survives PNNX → ncnn as exactly `1.0`, because a
  zero weight row makes the dot product identically zero regardless of kernel path.
  Dropping such classes instead (as this used to do) shifted every index after the first gap, which
  a client cannot recover from — it has only the class-map file to index with.
- **The threshold is one constant, and it is always calibrated.** There is no way to pin a value:
  a hand-set threshold goes stale silently the moment the checkpoint or the class set changes, and
  nothing in the artifact would reveal it. The clean feature pass is shared between calibration and
  prototype building, so this costs one extra augmented pass rather than two full ones. The chosen
  value and its measured FRR/FAR land in the sidecar `meta.json`.
- **A floor applies** (`--min-threshold`, default 0.2). Small thresholds reject genuine products —
  measured on posmlv, 0.02 costs 10% false-reject and 0.007 costs 50%, while false-accept stays
  under 0.2% across that range. A suggestion below the floor is clamped up and warns loudly: it
  means the positives and negatives are not well separated (mislabelled crops, or near-duplicate
  classes), which is a data problem rather than a threshold to honour.
- **Calibration is seeded** (`torch.manual_seed`, plus python/numpy in the DataLoader workers), so
  export is reproducible: the same checkpoint produces the same artifact every run.
- Prototype building is **parallelized** (DataLoader workers + batched GPU inference) and prints
  its **elapsed time**.
- **Prediction source:** with `--image` it's that one image; otherwise it scans **only** the `val/`
  and `test/` folders under `--data-dir` (the `outlier/`/`inlier/` folders are **not** predicted).
- **Prediction report:** each image is expected to live under a class subfolder (`val/<class>/…`),
  so for every image it prints the **expected** class (that subfolder name), the **detected** type
  (the open-set label — a known class or `unknown`), and the **nearest** known class with its cosine
  distance; a `✗` marks a mismatch. A final line reports the match rate against the folder labels:
  ```
  test/Other Drinks/CBKZ_20260806_770_0013.jpg
    expected: Other Drinks   detected: unknown   nearest: Pepsi (dist=0.5210)  ✗

  matched 812/900 (90.2%) against the folder label
  ```
- **Prediction device & speed:** prediction/verification runs on **CPU by default** so it mirrors the
  ncnn deployment target (the ONNX session uses `CPUExecutionProvider`; the TorchScript/in-memory
  model runs on CPU). Pass **`--gpu-predict`** to run it on the GPU instead (ONNX
  `CUDAExecutionProvider` with CPU fallback, TorchScript/in-memory model on `cuda`) — much faster for
  large `val/`+`test/` sweeps. Image decode is **batched via DataLoader workers** either way; note
  that after an export, predictions run **through the exported model**, which for the CPU default is
  intentionally the slow-but-faithful path. Prototype building itself always uses the GPU.
- **Self-verification:** if you export (`--export-onnx`/`--export-pt`), the subsequent predictions
  run **through the exported model** (ONNX / TorchScript), not the PyTorch prototypes — so any
  conversion discrepancy surfaces immediately. With no export, predictions use the in-memory
  PyTorch prototypes.
- Both exports write **`<path>.meta.json`**:
  ```json
  {"class_names": [...], "threshold": 0.46, "img_size": 224,
   "mean": [0.5,0.5,0.5], "std": [0.5,0.5,0.5], "resize": "area", "no_argmin": true,
   "threshold_source": "calibrated", "threshold_frr": 0.0011, "threshold_far": 0.0007,
   "threshold_plateau": [0.166, 0.7637], "threshold_floor": 0.2,
   "empty_classes": ["posm_14", "posm_44"]}
  ```
  `class_names` is the **full class-map order**, one entry per exported column; `empty_classes` (only
  present when there are any) names the ones with no training images, whose column is a zero
  prototype that always scores `dist = 1.0` and therefore never matches — report those as "no data"
  rather than as a product.
  `threshold` is a **scalar** (it used to be a per-class list — see the breaking-change note in §5);
  `resize` names the downscale filter the client should use; `no_argmin` records which output format
  was baked in. The `threshold_*` provenance fields are always present, since every export
  calibrates.

---

## 2.1 Outlier & inlier folders — training-data hygiene

Building prototypes is also a **label-quality audit** of your training crops, for free. For each
class the mean feature is a prototype, and every crop's cosine distance to *its own* prototype tells
you how well it fits its label. A fixed gate `OUTLIER_THR = 0.5` (cosine dist > 0.5 ≈ nearly
orthogonal; same-class crops should sit well under 0.2) splits each class into two groups:

- **outlier** — dist **> 0.5**: the crop looks nothing like the rest of its class. Usually a
  **mislabel**, a **bad crop** (occluded, blurred, wrong object, background), or a genuinely odd
  sample. These are **excluded from the prototype mean** (so one bad crop can't drag the prototype
  off), and — because they'd pollute a classifier — you generally want them **out of training** too.
- **inlier** — dist **≤ 0.5**: the crop is consistent with its class. These form the robust
  prototype, and they are the crops worth **training on**.

### Outlier folder (always written when outliers are found)

Outliers are **copied** to `<data-dir>/outlier/<class>/` (originals in `train/` are untouched) and
each is logged with two distances:

```
CocaCola    : WARNING — 2/153 outlier(s) excluded (dist>0.5) → .../outlier/CocaCola
               dist=0.6120  nearest_other=Pepsi (0.0450)   .../train/CocaCola/img_0007.jpg
               dist=0.5340  nearest_other=OtherDrinks (0.4900)  .../train/CocaCola/img_0088.jpg
```

- `dist` — distance to the crop's **own** (labeled) class prototype. Large ⇒ it doesn't belong.
- `nearest_other=<class> (<dist>)` — the **closest other** class prototype and its distance. This
  disambiguates *why* it's an outlier:
  - nearest-other **small** (e.g. `Pepsi (0.045)`) → the crop actually looks like that class →
    likely a **mislabel**; move it there.
  - nearest-other **also large** (e.g. `0.49`) → far from everything → a **genuine unknown / bad
    crop**; drop it.

Alongside the copies, a machine-readable summary is written to **`<data-dir>/outlier/outlier.txt`**
(CSV, one row per outlier, header included) so you can sort/filter the audit instead of scanning the
log:

```csv
classtype/file       , bad_dist, nearest_type, nearest_dist
CocaCola/img_0007.jpg, 0.6120  , Pepsi       , 0.0450
CocaCola/img_0088.jpg, 0.5340  , OtherDrinks , 0.4900
```

Columns are comma-separated but **space-padded** so the file also reads as aligned columns; any CSV
reader that strips surrounding whitespace parses it unchanged.

| column | meaning |
|---|---|
| `classtype/file` | `<labeled class>/<filename>` — matches the copied path under `outlier/` |
| `bad_dist` | cosine distance to its **own** class prototype (the reason it was flagged) |
| `nearest_type` | closest **other** class prototype (blank if only one class) |
| `nearest_dist` | cosine distance to `nearest_type` |

Tip: sort by `nearest_dist` ascending to surface the likely **mislabels** first (small
`nearest_dist` ⇒ the crop belongs to `nearest_type`); rows where `nearest_dist` is also large are
**genuine unknowns / bad crops**. The file is written whenever any outlier exists (independent of
`--copy-inliers`).

Review `outlier/` by eye (or sort `outlier.txt`), then fix labels or delete the bad crops in your
source dataset.

### Inlier folder (opt-in, `--copy-inliers`)

With `--copy-inliers`, the good inliers (dist ≤ 0.5) are **copied** to `<data-dir>/inlier/<class>/`,
preserving the class-folder layout. The result is a **cleaned mirror of `train/`** with the
noisy/mislabeled crops removed — point a fresh classifier training run at `inlier/` to learn from
clean data. A summary line reports the total: `inliers copied: <N> images → <inlier_root>`.

A companion report **`<data-dir>/inlier/inlier.txt`** lists every copied inlier with its distance to
its own class prototype (same space-padded, comma-separated format as `outlier.txt`):

```csv
classtype/file                , distance
CocaCola/img_0007.jpg         , 0.0450
Other Drinks/CBKZ_770_0013.jpg, 0.1832
```

| column | meaning |
|---|---|
| `classtype/file` | `<class>/<filename>` — matches the copied path under `inlier/` |
| `distance` | cosine distance to its **own** class prototype (smaller = tighter fit) |

Sort by `distance` **descending** to see the weakest-but-still-accepted crops (borderline cases near
the 0.5 gate); the tightest crops sit at the top when sorted ascending.

```bash
# Audit + emit a cleaned training set in one pass (no export needed)
uv run python nptools/openset.py \
    --data-dir posmlv --class-map posmlv/class_84.txt \
    --checkpoint <run>/model_best.pth.tar --copy-inliers
#   → posmlv/outlier/<class>/…   (crops to review / drop)
#   → posmlv/outlier/outlier.txt (CSV audit: classtype/file, bad_dist, nearest_type, nearest_dist)
#   → posmlv/inlier/inlier.txt   (CSV: classtype/file, distance)
#   → posmlv/inlier/<class>/…    (clean crops → train the next classifier here)
```

Notes:
- Both folders are **copies** (`shutil.copy2`); `train/` is never modified, so it's safe to re-run.
  Re-running overwrites same-named files rather than clearing the folder — delete `inlier/` first if
  you change the threshold and want a fresh set.
- Inlier/outlier membership uses the **clean** prototype distance (the same gate used for the robust
  mean) — it reflects "close to its own class on clean preprocessing," independent of the
  reject threshold.
  Outliers are excluded from calibration probes too, not just from the prototype: they are
  mislabelled crops, so scoring them would charge the threshold for a false reject *and* a false
  accept when the model is behaving correctly.
- This is a **bootstrap loop**: train a first classifier → run `--copy-inliers` to clean the data →
  retrain on `inlier/` for a stronger model → optionally repeat.

---

## 2.2 Choosing the threshold — always measured, never passed

The threshold is one number, so it is measured rather than guessed, and there is no flag to pin or
to skip it: every run that builds prototypes sweeps candidates, prints what each one costs, and
bakes the chosen value into the export. To read the curve **without writing artifacts, just omit
`--export-pt`/`--export-onnx`**:

```bash
uv run python nptools/openset.py \
    --data-dir posmlv --class-map posmlv/class_84.txt \
    --ck <run>/model_best.pth.tar
```

**No curated negative set is needed** — which matters, because a folder of assorted non-target
images is rarely trustworthy enough to calibrate against. Both error rates come from your own
labelled `train/` data, via two leave-one-out estimators computed from a single feature pass:

- **False accept** (an un-enrolled product wrongly accepted) — *leave-one-CLASS-out*. For class `c`,
  mask out its own prototype and score its images against the rest; by construction `c` is now a
  product that was never enrolled. Since each prototype is built only from its own class's images,
  dropping `c` leaves the others untouched, so this is a column mask, not a rebuild.
- **False reject** (a known product wrongly rejected) — the probe must not sit inside the prototype
  it is scored against. With `--aug` the probe is an augmented view, which was never part of the
  clean prototype, so the plain distance is already unbiased. With `--no-aug` the probe *is* one of
  the prototype's own images, so a *leave-one-IMAGE-out* mean `normalize(Σf − f_j)` removes it first.

Sorting the distances once turns every candidate threshold into a `searchsorted`, so the sweep is
effectively free after the feature pass.

```
     thr    FRR (known rejected)   FAR (unenrolled accepted)
  0.0069                  50.00%                       0.00%
  0.0205                  10.00%                       0.00%
  0.0523                   2.00%                       0.02%
  0.1171                   0.50%                       0.04%
  0.4065                   0.11%                       0.07%
  0.8450                   0.05%                       0.50%
  0.8836                   0.05%                       5.01%
  0.9147                   0.05%                      25.01%

safe plateau: [0.1660 .. 0.7637] — anywhere in here performs within 0.2pp
suggested   : 0.4600  (FRR=0.11%, FAR=0.07%)   [floor 0.2]
```

Reading it: the two distributions are far apart, so a wide band of thresholds performs identically.
The tool reports that **plateau** and suggests its midpoint — the value that tolerates the most
drift in either direction before falling off an edge. An equal-error point would be arbitrary here,
since both curves are flat across the gap.

Three caveats it prints with the table:

- LOCO negatives are other posm standees — same domain, same photographic style — so they sit closer
  to the prototypes than a random shelf photo would. **FAR is a pessimistic bound.**
- With no `val/`/`test/` split, every number is train-domain.
- The midpoint weights false-reject and false-accept equally, and there is no flag to bias it.

Every export runs this same sweep automatically and bakes the suggestion in, recording the
provenance in `meta.json`. Omitting the `--export-*` flags stops before writing anything, so you
can read the curve without producing an artifact.
`--min-threshold` can only raise the floor (more permissive); there is currently no way to bias the
choice toward a stricter value.

---

## 3. The open-set model — input & output

Both exports wrap the backbone in the **same** `_OpenSetWrapper` (ONNX and TorchScript are the
identical graph), so ncnn computes exactly what PyTorch does. The prototypes and the **scalar
threshold** are **baked into the graph as constants**, so at runtime the client only feeds the
image.

**Input** (both modes)

| name | shape | dtype | notes |
|---|---|---|---|
| `image` | `[1, 3, 224, 224]` | float32 | RGB, **squash**-resized to 224×224, normalized `mean=std=0.5` |

**Output** — depends on `--no-argmin`:

**(A) default — in-graph argmin (scalar outputs)**

| name | shape | dtype | meaning |
|---|---|---|---|
| `best_class_idx` | scalar | int64 | index of nearest prototype → `class_names[idx]` |
| `best_dist` | scalar | float32 | cosine distance to that prototype |
| `margin` | scalar | float32 | `best_dist − threshold`; **< 0 = known**, **> 0 = unknown** |

The graph does the `argmin` + gather itself, so the client just reads the scalars. **But** a
**stock pip ncnn wheel can't run it**: PNNX emits a `torch.argmin` node plus a dynamic-index `Crop`
(the gather), and ncnn has **no `ArgMin` layer at all** and no `Gather`
(`layer torch.argmin not exists or registered` → `network graph not ready`). Use this only with a
**custom decision layer** (see §6). (For reference: ncnn *does* ship an `ArgMax` layer, but it is
`OFF` by default; `ArgMin` does not exist at all.)

**(B) `--no-argmin` — client-side argmin (per-class vectors)**

| name | shape | dtype | meaning |
|---|---|---|---|
| `dists` | `[1, C]` | float32 | cosine distance to **every** class prototype |
| `margins` | `[1, C]` | float32 | `dists[c] − threshold` (same constant for every c) |

**C is the class-map line count**, not the number of classes that had images — data-less classes
occupy their column with a zero prototype and score a constant `dist = 1.0` (see §2 notes). So the
client indexes straight into the class-map order, and `meta['empty_classes']` names the columns that
can never match.

The graph stops at the vectors; the client does `best = argmin(dists)`, `unknown = margins[best] > 0`.
Only conv, L2-normalize, matmul and subtract remain — all universally supported — so it **converts
and runs on a stock ncnn wheel**. Recommended unless you specifically need the model to emit the
decision itself.

---

## 4. Generating ncnn files

Two routes — both start from the identical `_OpenSetWrapper`. Install tools once:

```bash
uv pip install pnnx ncnn        # (onnx already present for the ONNX route)
```

> **Which export for ncnn?** For a **stock pip ncnn wheel**, export with **`--no-argmin`** (§3B) —
> the graph is `Normalize → matmul → subtract` and runs as-is. The **default** in-graph-argmin
> export emits `torch.argmin` + a gather that a stock wheel can't run (§3A); use it only if you add
> the custom decision layer from **§6**.

### Route A — direct: TorchScript → ncnn  (leaner graph, recommended)

```bash
uv run python nptools/openset.py --export-pt nptools/openset.pt
uv run pnnx nptools/openset.pt inputshape=[1,3,224,224]
#   → openset.ncnn.param + openset.ncnn.bin, written NEXT TO the .pt (no cd needed), plus
#     openset_ncnn.py and the openset.pnnx.* intermediates, and openset.pt.meta.json from the export
```

PNNX's native TorchScript path skips the ONNX intermediate, so it usually emits **fewer ncnn
layers** (no ONNX shape-tracking `Gather/Unsqueeze/Reshape` clutter around the normalize/matmul
head). `inputshape` must match the exported `--img-size`: pnnx traces at that shape and
constant-folds `Conv2dSame`'s dynamic padding against it, so a wrong value bakes wrong padding into
the `.param` rather than failing.

Measured on a `--no-argmin` export of `tf_efficientnet_lite0` (83 classes): 76 layers — 33
`Convolution`, 16 `ConvolutionDepthWise`, 11 `BinaryOp`, 10 `Split`, and one each of `Pooling`,
`Reshape`, `Flatten`, `Normalize`, `InnerProduct`, `Input` — with **zero**
`ArgMin`/`ArgMax`/`Gather`/`Crop`, 494 MFLOPs. ncnn matches TorchScript to ~4e-3 on `dists` (fp32
kernel differences; the decision boundary sits three orders of magnitude away), with identical
labels on every image tested.

### Route B — via ONNX: PyTorch → ONNX → ncnn

```bash
uv run python nptools/openset.py --export-onnx nptools/openset.onnx
uv run pnnx nptools/openset.onnx inputshape=[1,3,224,224]     # pnnx's ONNX frontend
#   (or the older: onnx2ncnn openset.onnx openset.ncnn.param openset.ncnn.bin)
```

Same resulting model; the graph may carry a few extra layers from the ONNX export. Use this if
you prefer reusing the existing ONNX export.

**Useful pnnx args:** `fp16=1` (default, smaller/faster), `optlevel=2` (max fusion),
`inputshape2=[1,3,256,256]` (enable dynamic H/W). Output blob names are written in
`openset.ncnn.param` (top lines) and the generated `openset.ncnn.py`.

---

## 5. How the client uses the output

This shows the **`--no-argmin`** export (mode B) — the recommended, stock-ncnn-compatible one. The
client does: **preprocess → run → argmin over the 2 output vectors**. (For the default scalar
export, the client instead reads `best_class_idx`/`best_dist`/`margin` directly — but that graph
needs a custom decision layer on ncnn, see §3A / §6.)

```python
import numpy as np, ncnn, json, cv2

meta = json.load(open('nptools/openset.pt.meta.json'))   # class_names, threshold, img_size, mean/std, resize
net = ncnn.Net()
net.load_param('nptools/openset.ncnn.param')
net.load_model('nptools/openset.ncnn.bin')

# --- preprocess: MUST match training/export ---
bgr = cv2.imread('x.jpg', cv2.IMREAD_COLOR)
# Prefer INTER_AREA: it is the correct filter for shrinking and matches how the prototypes
# were built. cv2's default is INTER_LINEAR, which does not anti-alias -- see the note below.
bgr = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)      # squash (not letterbox)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
mat = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, 224, 224)
# mean=std=0.5 on 0..1  ==  (x/255 - 0.5)/0.5  ==  x/127.5 - 1  on 0..255 pixels:
mat.substract_mean_normalize([127.5, 127.5, 127.5], [1/127.5, 1/127.5, 1/127.5])

ex = net.create_extractor()
ex.input('in0', mat)                        # input/output blob names: see openset.ncnn.param
dists   = np.array(ex.extract('out0')[1])   # [C] cosine distance to each prototype
margins = np.array(ex.extract('out1')[1])   # [C] dists[c] - threshold

best_idx = int(dists.argmin())              # nearest prototype (argmin done on the CLIENT)
if margins[best_idx] > 0:                    # margin > 0 → unknown
    result = 'unknown / new product'
else:
    result = meta['class_names'][best_idx]  # known product
print(result, 'dist=', float(dists[best_idx]))
```

> Blob names: PNNX names the input `in0` and the outputs `out0` (`dists`) / `out1` (`margins`) —
> confirm against the top of `openset.ncnn.param` / the generated `*_ncnn.py`.

### The resize filter — recorded, but not critical

`meta.json` carries **`"resize": "area"`** alongside `mean`/`std`/`img_size`, so the client has one
authoritative place to read it.

**Measured impact at current class separation: negligible.** Calibrating end-to-end under each
filter gives the *same* threshold, and shipping PIL-built prototypes to a client using any of them
costs nothing detectable:

```
                    calibrated thr    FRR     FAR      (client vs PIL-built prototypes, at 0.47)
PIL bicubic              0.47        0.01%   0.02%
cv2 INTER_AREA           0.47        0.02%   0.02%
cv2 INTER_LINEAR         0.47        0.02%   0.01%
```

That is because positives sit at ~0.007 and negatives at ~0.93, so a filter-induced shift of median
2e-4 (max 0.05) cannot move anything across a 0.47 threshold. Unlike a wrong `mean`/`std`, which is
catastrophic, a wrong resize filter is currently harmless.

Still worth pinning `INTER_AREA`: it is the correct filter for downscaling, it matches how the
prototypes are built, and it is free insurance if class separation ever tightens as more SKUs are
enrolled. The trap is that **`cv2.resize(img, (224,224))` with no `interpolation=` argument gives
`INTER_LINEAR`**, and nobody writing that line thinks they are choosing a filter. Measured at a 16×
reduction, by how many source pixels actually influence the output:

| filter | source pixels read | anti-aliased? |
|---|---|---|
| `cv2.INTER_AREA` | 100% | **yes — prefer this** |
| PIL `BICUBIC` (what the exporter uses) | 92.6% | yes |
| `cv2.INTER_CUBIC` | 3.1% | no |
| `cv2.INTER_LINEAR` ← **cv2 default** | 1.6% | no |
| `cv2.INTER_NEAREST` | 0.4% | no |

OpenCV's `INTER_CUBIC`/`INTER_LINEAR` use a *fixed* 4×4 / 2×2 kernel regardless of scale — they are
built for upscaling. PIL's `BICUBIC` scales its kernel support with the reduction factor, which is
why it lands with `INTER_AREA` and not with cv2's similarly-named cubic.

Feature divergence against the exporter's preprocessing (cosine distance, 2000 crops). Severity
scales with the reduction factor, so the largest source images differ most — but as the table above
shows, none of it reaches the threshold:

```
PIL-bicubic ↔ cv2 INTER_AREA     median 0.00003   p95 0.00019   max 0.043   ← effectively identical
PIL-bicubic ↔ cv2 INTER_LINEAR   median 0.00019   p95 0.00213   max 0.050   ← the cv2 default
PIL-bicubic ↔ cv2 INTER_CUBIC    median 0.00027   p95 0.00270   max 0.053
```

Client rules:
1. **Preprocess identically** to export — squash-resize to 224×224 (because training used
   `--crop-mode=squash`) with **`INTER_AREA`**, RGB, `substract_mean_normalize([127.5]*3,
   [1/127.5]*3)`. This is the #1 source of "correct in PyTorch, wrong in ncnn."
2. **Decision:** `best = argmin(dists)`; `margins[best] > 0` (equivalently `dists[best] >
   meta['threshold']`) ⇒ **reject as unknown**; otherwise the product is `class_names[best]`. The
   threshold is already baked into `margins`, so the client just checks the sign at the nearest class.
3. **Index → name** via `meta.json`'s `class_names`, which is the full class-map order and matches
   the exported prototype matrix column for column. A name listed in `meta['empty_classes']` has no
   training data behind it and can only appear as a rejection, so surface it as "no data" if it ever
   comes back as `best`.
4. **Retuning the threshold without re-export:** it is baked into `margins`, so to adjust without
   re-exporting, ignore `margins` and apply your own rule on `dists`, e.g.
   `dists[best] > my_threshold`.

> **Breaking change:** `meta.json` used to carry a per-class **`thresholds`** list (one entry per
> class). It now carries a single scalar **`threshold`**, because calibration showed per-class
> thresholds overfit badly (3.8× worse on held-out data). Clients reading `meta['thresholds'][i]`
> must switch to `meta['threshold']`. The graph outputs are unchanged.
>
> **Breaking change:** the output vectors now have **one column per class-map entry**, where they
> used to have one per class that had images (on posmlvx: 83 instead of 71). Index with the full
> class-map order and treat `meta['empty_classes']` as never-matching. Models exported before this
> change are misaligned against the current class map and must be re-exported — the shift starts at
> the first data-less class, so early indices look correct, which makes it easy to miss.

---

## 6. Default (in-graph argmin) on ncnn — custom decision layer

If you want the model itself to emit the decision (the default, non-`--no-argmin` export) on ncnn,
you must supply the missing op. ncnn has no `ArgMin` and no `Gather`, so don't try to add those two
separately — implement the **whole head as one custom layer** that takes `dists` in and outputs
`(best_idx, best_dist, unknown)`:

```cpp
// nptools/openset_decide.cpp  — register before load_param()
#include "layer.h"
using namespace ncnn;
class OpenSetDecide : public Layer {
public:
    OpenSetDecide() { one_blob_only = true; }
    virtual int load_param(const ParamDict& pd) { thr = pd.get(0, Mat()); return 0; }   // -23300=C,thr0,...
    virtual int forward(const Mat& dists, Mat& top, const Option&) const {
        const float* d = dists; int best = 0; float bd = d[0];
        for (int c = 1; c < dists.w; c++) if (d[c] < bd) { bd = d[c]; best = c; }        // argmin
        const float* t = thr;
        top.create(3); top[0] = (float)best; top[1] = bd; top[2] = (bd > t[best]) ? 1.f : 0.f;
        return 0;
    }
private: Mat thr;
};
DEFINE_LAYER_CREATOR(OpenSetDecide)
```
```cpp
net.register_custom_layer("OpenSetDecide", OpenSetDecide_layer_creator);
net.load_param("openset.ncnn.param");   // hand-edit the argmin/Crop tail → one OpenSetDecide line
net.load_model("openset.ncnn.bin");
```

This is C++ only (the pip wheel can't register a pure-Python custom layer). In practice the
**`--no-argmin` route is simpler and portable** — same result with no custom layer — so prefer it
unless your app framework must have the model emit the final decision.

---

## TL;DR (recommended: `--no-argmin`, runs on stock ncnn)

```bash
# 0. (optional) see what each threshold costs, on your own data, no negatives needed:
#    same command as below minus the --export-* flags, so nothing is written
uv run python nptools/openset.py \
    --data-dir posmlv --class-map posmlv/class_84.txt \
    --ck <run>/model_best.pth.tar

# 1. build open-set model from checkpoint + training data (classes from class-map), export it
#    the threshold is always calibrated automatically -- there is nothing to pass
uv run python nptools/openset.py \
    --data-dir posmlv --class-map posmlv/class_84.txt \
    --checkpoint <run>/model_best.pth.tar \
    --no-argmin --export-pt nptools/openset.pt  # (or --export-onnx)

# 2. convert to ncnn
uv run pnnx nptools/openset.pt inputshape=[1,3,224,224]                       # → openset.ncnn.*

# 3. client: preprocess (squash 224 + mean/std 0.5) → run → best=argmin(dists) →
#    margins[best]>0 ⇒ unknown ; else class_names[best]  (names/threshold in openset.pt.meta.json)
```

(Drop `--no-argmin` to bake the decision into the graph instead — but then deploy with the §6
custom layer.)
