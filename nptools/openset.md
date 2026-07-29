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
      ┌──────────── build once, from training data (classes from class-map) ──────┐
      │  for each class: mean of its pre-logits features → prototype (L2-norm)   │
      │  PER-CLASS threshold[c] = 95th percentile of THAT class's own distances  │
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
| `--checkpoint PATH` | **yes** (PyTorch/export path) | the trained checkpoint to load |
| `--data-dir DIR` | no (default posmlv) | dataset root with `train/`, `val/`, `test/` subfolders |
| `--img-size N` | no (default 224) | square input size |
| `--export-onnx [PATH]` | no | export ONNX (+ `<PATH>.meta.json`); default `<checkpoint>_openset.onnx` |
| `--export-pt [PATH]` | no | export TorchScript (+ `<PATH>.meta.json`); default `<checkpoint>_openset.pt` |
| `--image PATH` | no | classify a single image (else scans `val/` + `test/`) |
| `--load-onnx PATH` | no | predict via an already-exported ONNX; class names read from `<PATH>.meta.json` |

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
- **Per-class thresholds** are computed automatically (95th percentile of each class's own
  distances) — there is no single scalar override; each class is judged against its own spread.
- Prototype building is **parallelized** (DataLoader workers + batched GPU inference) and prints
  its **elapsed time**.
- **Prediction source:** with `--image` it's that one image; otherwise it scans **both** `val/`
  and `test/` under `--data-dir`.
- **Self-verification:** if you export (`--export-onnx`/`--export-pt`), the subsequent predictions
  run **through the exported model** (ONNX / TorchScript), not the PyTorch prototypes — so any
  conversion discrepancy surfaces immediately. With no export, predictions use the in-memory
  PyTorch prototypes.
- Both exports write **`<path>.meta.json`** = `{class_names, thresholds, img_size, mean, std}` —
  `thresholds` is a per-class list aligned with `class_names`. The client (and `--load-onnx`) reads
  this to map an output index to a class name.

---

## 3. The open-set model — input & output

Both exports wrap the backbone in the **same** `_OpenSetWrapper` (ONNX and TorchScript are the
identical graph), so ncnn computes exactly what PyTorch does.

**Input**

| name | shape | dtype | notes |
|---|---|---|---|
| `image` | `[1, 3, 224, 224]` | float32 | RGB, **squash**-resized to 224×224, normalized `mean=std=0.5` |

**Output** (3 tensors)

| name | shape | dtype | meaning |
|---|---|---|---|
| `best_class_idx` | scalar | int64 | index of nearest prototype → `class_names[idx]` |
| `best_dist` | scalar | float32 | cosine distance to that prototype (0 = identical, higher = less similar) |
| `margin` | scalar | float32 | `best_dist − threshold[best_class]` (per-class); **< 0 = known**, **> 0 = unknown (reject)** |

Prototypes and the **per-class threshold vector** are **baked into the graph as constants**, so at
runtime the client only feeds the image — and the reject decision is simply `margin > 0`.

---

## 4. Generating ncnn files

Two routes — both start from the identical `_OpenSetWrapper`. Install tools once:

```bash
uv pip install pnnx ncnn        # (onnx already present for the ONNX route)
```

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

The client (ncnn app) does: **preprocess → run → interpret the 3 outputs**.

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
ex.input('image', mat)                      # input/output blob names: see openset.ncnn.param
_, idx    = ex.extract('best_class_idx')
_, dist   = ex.extract('best_dist')
_, margin = ex.extract('margin')

best_idx = int(np.array(idx)[0])
if float(np.array(margin)[0]) > 0:          # margin > 0 → unknown
    result = 'unknown / new product'
else:
    result = meta['class_names'][best_idx]  # known product
print(result, 'dist=', float(np.array(dist)[0]))
```

Client rules:
1. **Preprocess identically** to export — squash-resize to 224×224 (because training used
   `--crop-mode=squash`), RGB, `substract_mean_normalize([127.5]*3, [1/127.5]*3)`. This is the #1
   source of "correct in PyTorch, wrong in ncnn."
2. **Decision:** `margin > 0` (equivalently `best_dist > thresholds[best_class_idx]`) ⇒ **reject as
   unknown**; otherwise the product is `class_names[best_class_idx]`. The per-class threshold is
   already baked into `margin`, so the client just checks its sign.
3. **Index → name** via `meta.json`'s `class_names` (order matches the exported prototype matrix
   and the `thresholds` list).
4. **Retuning thresholds without re-export:** the per-class thresholds are fixed in the graph at
   export time. To adjust, re-export (e.g. change `threshold_quantile` in `build_prototypes`), or
   export a distance-vector variant and apply `dists > meta['thresholds']` in the client.

---

## TL;DR

```bash
# 1. build open-set model from checkpoint + training data (classes from class-map), export it
uv run --no-sync python nptools/openset.py \
    --data-dir posmlv --class-map posmlv/class_84.txt \
    --checkpoint <run>/model_best.pth.tar --export-pt nptools/openset.pt   # (or --export-onnx)

# 2. convert to ncnn
cd nptools && pnnx openset.pt inputshape=[1,3,224,224]                       # → openset.ncnn.*

# 3. client: preprocess (squash 224 + mean/std 0.5) → run → 
#    margin>0 ⇒ unknown ; else class_names[best_class_idx]  (names/thresholds in openset.pt.meta.json)
```
