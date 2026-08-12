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
      │  PER-CLASS threshold[c] = --quantile of its own distances (default 0.97) │
      └──────────────────────────────────────────────────────────────────────────┘

  image ─► backbone ─► pre-logits feature (1280-d) ─► L2 normalize
                                                          │
                                                          ▼
                        cosine distance to every class prototype  →  dists[C]
                                                          │
                             best = argmin(dists)         │
                                                          ▼
                    margin = best_dist − threshold[best]        (per-class threshold)
                        margin < 0  →  KNOWN   → class = class_names[best]
                        margin > 0  →  UNKNOWN → reject (new / unseen product)
```

Key idea: a known product lands **close** to its class prototype in feature space; an unknown
product lands **far from all** prototypes → large `best_dist` → positive margin → rejected.
The classifier logits are *not* used for the decision — the **pre-logits feature** is.

**Why per-class (not one global) threshold:** classes have very different spreads — a tight class
may sit at mean-dist 0.02, a diffuse catch-all (`others`) at 0.19 with max 0.47. A single global
threshold is dominated by the diffuse classes and becomes too loose for the tight ones (letting
genuine unknowns pass as a known product). Each class is judged against **its own** distance
distribution instead.

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
| `--img-size N` | no (default 224) | square input size |
| `--quantile Q` | no (default 0.97) | per-class reject threshold = this quantile of the class's in-distribution cosine distances (higher ⇒ fewer knowns rejected, more unknowns accepted) |
| `--aug` / `--no-aug` | no (default `--aug`) | calibrate thresholds from **npaug-augmented** distances (prototype stays clean), widening the spread to real-world variation; `--no-aug` uses clean distances |
| `--aug-views N` | no (default 2) | number of augmented passes over train used for threshold calibration |
| `--copy-inliers` | no (default off) | while building prototypes, also copy the **good inlier** crops (cosine dist ≤ 0.5 to their class prototype) into `<data-dir>/inlier/<class>/` — a cleaned training set (see §2.1) |
| `--gpu-predict` | no (default **CPU**) | run **prediction/verification** on GPU (ONNX `CUDAExecutionProvider`, TorchScript + in-memory model on `cuda`). Default is CPU so verification mirrors the ncnn deployment target. **Prototype building always uses the GPU** regardless of this flag; it only affects the predict/verify pass. |
| `--export-onnx [PATH]` | no | export ONNX (+ `<PATH>.meta.json`); default `<checkpoint>_openset.onnx` |
| `--export-pt [PATH]` | no | export TorchScript (+ `<PATH>.meta.json`); default `<checkpoint>_openset.pt` |
| `--image PATH` | no | classify a single image (else scans `val/` + `test/`) |
| `--load-onnx PATH` | no | predict via an already-exported ONNX; class names and output format (`no_argmin`) are read from `<PATH>.meta.json` — the CLI `--no-argmin` flag is ignored on this path |
| `--no-argmin` | no | export the **client-side-argmin** variant: the graph emits per-class `dists`/`margins` vectors (no `argmin`/`gather` in the model) and the argmin + reject move to the client. Converts and runs on a **stock ncnn wheel**. Default keeps the **in-graph argmin** scalar outputs — the graph runs `torch.argmin` + a dynamic-index `Crop` (gather), which a stock ncnn wheel **can't** run (ncnn has **no `ArgMin` layer** and no `Gather`), so that path needs a custom layer. |

```bash
# Classify val/ + test/ images with the PyTorch prototypes (no export)
uv run --no-sync python nptools/openset.py \
    --data-dir posmlv --class-map posmlv/class_84.txt --checkpoint <run>/model_best.pth.tar

# Export the open-set model (then it self-verifies by predicting THROUGH the exported model)
uv run --no-sync python nptools/openset.py \
    --data-dir posmlv --class-map posmlv/class_84.txt --checkpoint <run>/model_best.pth.tar \
    --export-onnx model.onnx        # or --export-pt model.pt

# Predict through an already-exported ONNX model (CPU), no prototype rebuild
uv run --no-sync python nptools/openset.py --load-onnx model.onnx --image x.jpg
```

Notes:
- **Class-map drives prototype building.** Only classes listed in the class-map are used, in that
  order. A class-map entry with **no folder** or an **empty folder** is warned about and skipped;
  a class with **< 10 images** is warned (prototype/threshold may be unreliable) but still used.
  Folders **not** in the class-map are ignored. So the exported model may have **fewer** classes
  than the class-map (skipped ones); `meta.json` records the exact final list.
- **Per-class thresholds** are computed automatically — the `--quantile` (default 0.97) of each
  class's own distance spread; there is no single global value, each class is judged against its own
  spread. With `--aug` (default on) those distances come from **npaug-augmented** views (the
  prototype is still built from clean images), so the threshold reflects real-world variation rather
  than the optimistic clean-train spread; `--no-aug` uses clean distances. Only class-map folders are
  read, so `others`/unknown is never augmented.
- Prototype building is **parallelized** (DataLoader workers + batched GPU inference) and prints
  its **elapsed time**.
- **Prediction source:** with `--image` it's that one image; otherwise it scans **both** `val/`
  and `test/` under `--data-dir`.
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
- Both exports write **`<path>.meta.json`** = `{class_names, thresholds, img_size, mean, std, no_argmin}` —
  `thresholds` is a per-class list aligned with `class_names`; `no_argmin` records which output
  format was baked in. The client (and `--load-onnx`) reads this to map an output index to a class
  name and to know whether to expect scalar or per-class-vector outputs.

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
uv run --no-sync python nptools/openset.py \
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
  mean), not the augmented threshold distances — it reflects "close to its own class on clean
  preprocessing," independent of `--quantile` / `--aug`.
- This is a **bootstrap loop**: train a first classifier → run `--copy-inliers` to clean the data →
  retrain on `inlier/` for a stronger model → optionally repeat.

---

## 3. The open-set model — input & output

Both exports wrap the backbone in the **same** `_OpenSetWrapper` (ONNX and TorchScript are the
identical graph), so ncnn computes exactly what PyTorch does. Prototypes and the **per-class
threshold vector** are **baked into the graph as constants**, so at runtime the client only feeds
the image.

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
| `margin` | scalar | float32 | `best_dist − threshold[best_class]`; **< 0 = known**, **> 0 = unknown** |

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
| `margins` | `[1, C]` | float32 | `dists[c] − threshold[c]` (per-class) |

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
uv run --no-sync python nptools/openset.py --export-pt nptools/openset.pt
cd nptools && pnnx openset.pt inputshape=[1,3,224,224]
#   → openset.ncnn.param + openset.ncnn.bin   (plus openset.pt.meta.json from the export)
```

PNNX's native TorchScript path skips the ONNX intermediate, so it usually emits **fewer ncnn
layers** (no ONNX shape-tracking `Gather/Unsqueeze/Reshape` clutter around the normalize/matmul
head).

### Route B — via ONNX: PyTorch → ONNX → ncnn

```bash
uv run --no-sync python nptools/openset.py --export-onnx nptools/openset.onnx
cd nptools && pnnx openset.onnx inputshape=[1,3,224,224]     # pnnx's ONNX frontend
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
import numpy as np, ncnn, json
from PIL import Image

meta = json.load(open('nptools/openset.pt.meta.json'))   # class_names, thresholds[], img_size, mean/std
net = ncnn.Net()
net.load_param('nptools/openset.ncnn.param')
net.load_model('nptools/openset.ncnn.bin')

# --- preprocess: MUST match training/export ---
img = Image.open('x.jpg').convert('RGB').resize((224, 224))          # squash (not letterbox)
mat = ncnn.Mat.from_pixels(np.array(img), ncnn.Mat.PixelType.PIXEL_RGB, 224, 224)
# mean=std=0.5 on 0..1  ==  (x/255 - 0.5)/0.5  ==  x/127.5 - 1  on 0..255 pixels:
mat.substract_mean_normalize([127.5, 127.5, 127.5], [1/127.5, 1/127.5, 1/127.5])

ex = net.create_extractor()
ex.input('in0', mat)                        # input/output blob names: see openset.ncnn.param
dists   = np.array(ex.extract('out0')[1])   # [C] cosine distance to each prototype
margins = np.array(ex.extract('out1')[1])   # [C] dists[c] - threshold[c]

best_idx = int(dists.argmin())              # nearest prototype (argmin done on the CLIENT)
if margins[best_idx] > 0:                    # margin > 0 → unknown
    result = 'unknown / new product'
else:
    result = meta['class_names'][best_idx]  # known product
print(result, 'dist=', float(dists[best_idx]))
```

> Blob names: PNNX names the input `in0` and the outputs `out0` (`dists`) / `out1` (`margins`) —
> confirm against the top of `openset.ncnn.param` / the generated `*_ncnn.py`.

Client rules:
1. **Preprocess identically** to export — squash-resize to 224×224 (because training used
   `--crop-mode=squash`), RGB, `substract_mean_normalize([127.5]*3, [1/127.5]*3)`. This is the #1
   source of "correct in PyTorch, wrong in ncnn."
2. **Decision:** `best = argmin(dists)`; `margins[best] > 0` (equivalently `dists[best] >
   thresholds[best]`) ⇒ **reject as unknown**; otherwise the product is `class_names[best]`. The
   per-class threshold is already baked into `margins`, so the client just checks the sign at the
   nearest class.
3. **Index → name** via `meta.json`'s `class_names` (order matches the exported prototype matrix
   and the `thresholds` list).
4. **Retuning thresholds without re-export:** the per-class thresholds are baked into `margins`. To
   adjust without re-exporting, ignore `margins` and apply your own rule on `dists` directly, e.g.
   `dists > np.array(meta['thresholds'])` (or scale them) in the client.

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
# 1. build open-set model from checkpoint + training data (classes from class-map), export it
uv run --no-sync python nptools/openset.py \
    --data-dir posmlv --class-map posmlv/class_84.txt \
    --checkpoint <run>/model_best.pth.tar --no-argmin --export-pt nptools/openset.pt  # (or --export-onnx)

# 2. convert to ncnn
cd nptools && pnnx openset.pt inputshape=[1,3,224,224]                       # → openset.ncnn.*

# 3. client: preprocess (squash 224 + mean/std 0.5) → run → best=argmin(dists) →
#    margins[best]>0 ⇒ unknown ; else class_names[best]  (names/thresholds in openset.pt.meta.json)
```

(Drop `--no-argmin` to bake the decision into the graph instead — but then deploy with the §6
custom layer.)
