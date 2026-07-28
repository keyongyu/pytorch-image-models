# Open-set recognition (openset.py) → ONNX / ncnn

`kytest/openset.py` turns a normal **closed-set classifier** (timm `tf_efficientnet_lite0`, trained
on N known products) into an **open-set** model that can also say *"this is not any product I
know"* (reject unknown / newly-introduced SKUs). It then exports that model to ONNX and/or
TorchScript for deployment with **ncnn**.

---

## 1. Concept & flow

A plain softmax classifier is **closed-set**: it always forces an input into one of its N trained
classes, even a brand-new product it has never seen. Open-set recognition adds a *reject* option.

How it works here (feature-prototype / metric approach):

```
                    ┌─────────────── build once, from training data ────────────────┐
                    │  for each class: mean of its pre-logits features → prototype  │
                    │  prototypes are L2-normalized                                 │
                    │  threshold = 95th percentile of in-distribution distances     │
                    └───────────────────────────────────────────────────────────────┘

  image ─► backbone ─► pre-logits feature (1280-d) ─► L2 normalize
                                                          │
                                                          ▼
                        cosine distance to every class prototype  →  dists[C]
                                                          │
                             best = argmin(dists)         │
                                                          ▼
                    margin = best_dist − threshold
                        margin < 0  →  KNOWN   → class = class_names[best]
                        margin > 0  →  UNKNOWN → reject (new / unseen product)
```

Key idea: a known product lands **close** to its class prototype in feature space; an unknown
product lands **far from all** prototypes → large `best_dist` → positive margin → rejected.
The classifier logits are *not* used for the decision — the **pre-logits feature** is.

---

## 2. openset.py — usage

Runs against the posmlv model by default (`MODEL_NAME`, `NUM_CLASSES`, `DATA_DIR`, `OUTPUT_DIR`
constants at the top of the file). Prototypes are built from `<DATA_DIR>/train/<class>/*`.

```bash
# Build prototypes from training data and classify one image (PyTorch, prints top-1 + distance)
uv run --no-sync python kytest/openset.py --image path/to/img.jpg

# Override the auto-picked checkpoint / auto threshold
uv run --no-sync python kytest/openset.py --checkpoint <run>/model_best.pth.tar --threshold 0.23

# Export the open-set model:
uv run --no-sync python kytest/openset.py --export-onnx model.onnx   # ONNX
uv run --no-sync python kytest/openset.py --export-pt   model.pt     # TorchScript (+ meta.json)

# Run inference through an exported ONNX model (CPU), skipping prototype rebuild
uv run --no-sync python kytest/openset.py --load-onnx model.onnx --class-names a,b,c --image x.jpg
```

Notes:
- Classes with **no training images are skipped** (a prototype needs ≥1 example), so the number
  of prototypes may be < N. `class_names` is returned aligned to the prototype matrix.
- `--export-pt` also writes **`<path>.meta.json`** = `{class_names, threshold, img_size, mean, std}`
  — the client needs this to map an output index to a name and to know the threshold.

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
| `margin` | scalar | float32 | `best_dist − threshold`; **< 0 = known**, **> 0 = unknown (reject)** |

Prototypes and the threshold are **baked into the graph as constants**, so at runtime the client
only feeds the image.

---

## 4. Generating ncnn files

Two routes — both start from the identical `_OpenSetWrapper`. Install tools once:

```bash
uv pip install pnnx ncnn        # (onnx already present for the ONNX route)
```

### Route A — direct: TorchScript → ncnn  (leaner graph, recommended)

```bash
uv run --no-sync python kytest/openset.py --export-pt kytest/openset.pt
cd kytest && pnnx openset.pt inputshape=[1,3,224,224]
#   → openset.ncnn.param + openset.ncnn.bin   (plus openset.pt.meta.json from the export)
```

PNNX's native TorchScript path skips the ONNX intermediate, so it usually emits **fewer ncnn
layers** (no ONNX shape-tracking `Gather/Unsqueeze/Reshape` clutter around the normalize/matmul
head).

### Route B — via ONNX: PyTorch → ONNX → ncnn

```bash
uv run --no-sync python kytest/openset.py --export-onnx kytest/openset.onnx
cd kytest && pnnx openset.onnx inputshape=[1,3,224,224]     # pnnx's ONNX frontend
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

meta = json.load(open('kytest/openset.pt.meta.json'))   # class_names, threshold, img_size, mean/std
net = ncnn.Net()
net.load_param('kytest/openset.ncnn.param')
net.load_model('kytest/openset.ncnn.bin')

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
2. **Decision:** `margin > 0` (equivalently `best_dist > threshold`) ⇒ **reject as unknown**;
   otherwise the product is `class_names[best_class_idx]`.
3. **Index → name** via `meta.json`'s `class_names` (order matches the exported prototype matrix).
4. **Tuning the threshold without re-export:** if you'd rather keep the threshold adjustable at
   runtime, export a distance-vector variant instead (drop the baked `argmin`/threshold) and do
   `argmin` + compare in the client — but with the current model the threshold is fixed at export
   and read from `meta.json` for reference.

---

## TL;DR

```bash
# 1. build open-set model from the trained checkpoint + training data, export it
uv run --no-sync python kytest/openset.py --export-pt kytest/openset.pt      # (or --export-onnx)

# 2. convert to ncnn
cd kytest && pnnx openset.pt inputshape=[1,3,224,224]                        # → openset.ncnn.*

# 3. client: preprocess (squash 224 + mean/std 0.5) → run → 
#    margin>0 ⇒ unknown ; else class_names[best_class_idx]  (names/threshold in openset.pt.meta.json)
```
