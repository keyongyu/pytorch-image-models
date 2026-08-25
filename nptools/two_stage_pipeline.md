# Two-stage pipeline: SSD box detector → crop → open-set classifier

Working notes for running `NPBox_openset.npnn` end to end: detect boxes in a photo, crop each box,
classify the crop with the open-set model. Covers what is actually inside the container, the
client-side geometry the detector section does **not** do for you, and an evaluation of whether the
two stages can be fused into one network.

Reference run throughout: `posmlvx/output/20260825-075721-tf_efficientnet_lite0_in1k-128/`.

---

## 1. What is in the container

`NPBox_openset.npnn` (8,418,267 B) is a valid two-section npnn, built by `nptools/make_npnn.py`
(format: [`npnn_openset_format.txt`](npnn_openset_format.txt)). Verified by splitting it and
comparing against the sources — **both payloads round-trip byte-exact**:

| section | offset | param | bin | identical to |
|---|---|---|---|---|
| §1 detector | 302 | 18,621 B (119 layers, 136 blobs) | 5,215,812 B | `pbfiles/NPBox_20260816.pb` payload (5,234,433 B) |
| §2 open-set | 5,234,812 | 8,202 B (76 layers, 86 blobs) | 3,175,244 B | `openset.ncnn.param` / `.bin` |

The `Model: OpenSet` key sits at 5,234,752 and carries `SkuNum: 83`, `InputWidth: 128`,
`InputHeight: 128`. That agrees with `openset.onnx.meta.json` (83 `class_names`) and with
`posmlvx/class_84.txt`, which despite its name has **83** lines.

Detector header, kept verbatim from the `.pb`:

```
Version: MBV2_1/FORMAT4
StepScale: 0.0267 0.0534 0.1068 0.2136 0.4272
AspectRatio: 0.25 0.5 0.75 1.0 1.333 2.0 4.0
SkuNum: 1
InferNetType: NCNN
AspectRatioThresholdMin: 0.6
AspectRatioThresholdMax: 1.6
OverlapRatio: 0.5
InputWidth: 600
InputHeight: 800
Net: mobilenet_v2
TrainDate: 2026-08-16:04:17:14
Content: OpenSet
```

**Not in the container:** class names (only the count), the detector's score/NMS thresholds, and the
prior-box definition. The class map has to ship alongside — a mismatch silently shifts every label.

---

## 2. Stage 1 — the detector is raw heads only

Loaded §1 in ncnn and ran it. There is **no `PriorBox` and no `DetectionOutput` layer**; the graph
ends at the reshaped heads:

```
input.1  (1, 3, 800, 600)  ->  np_loc   [48260, 4]   raw offsets
                           ->  np_score [48260, 2]   softmax already applied in-graph, col 1 = box
```

Layer census: 46 Convolution, 27 ConvolutionDepthWise, 14 Split, 10 BinaryOp, 10 Reshape,
8 Permute, 2 Concat, 1 Input, 1 Softmax.

So decode, prior generation and NMS are all **client-side work**.

### 2.1 Priors

Four heads, taken from `base.13` (stride 16), `base.17` (32), `extras.0` (64), `extras.1` (128).
At 800×600 that is feature maps `(50,38) (25,19) (13,10) (7,5)` = 2540 cells, and every loc head
emits 76 channels = **19 anchors per cell**. 19 × 2540 = **48260**, matching the tensor ncnn returns.

`Version: MBV2_1/FORMAT4` selects the generator:
**`/home/keyong/DMTrain/lib/layers/functions/prior_box.py` → `PriorBox.forward_format4()` (line 478).**

```
aspect  = W / H = 0.75
steps[k] = (16 * 2^k / H, 16 * 2^k / W)          # "mobilenet_v2", i.e. NOT the mobilenet8x8 branch
offset[k] = half of steps[k]
s_k = StepScale[k],  s_k_y = s_k * aspect
```

Per cell, for each aspect ratio `ar`:
- `0.6 < ar < 1.6` → one anchor at the cell centre.
- `ar <= 0.6` (wide) → **tiled horizontally** across the cell, stepping `(1 − OverlapRatio) * anchor_w`.
- `ar >= 1.6` (tall) → **tiled vertically**, stepping `(1 − OverlapRatio) * anchor_h`.

then two extras per cell: the `sqrt(s_k · s_{k+1})` box, and the FORMAT4 corner box
`[(j+1)·step_w, (i+1)·step_h, step_w, step_h]`. Finally `clamp(0, 1)`.

Hand-counting layer 0 gives `0.25→4, 0.5→3, 0.75→1, 1.0→1, 1.333→1, 2.0→3, 4.0→4` = 17, plus the
two extras = **19**. Scales and steps both double per level, so the count is 19 at every level.

> Not yet done: a numerical diff of a standalone reimplementation against `forward_format4()`
> itself. The count matches, the formula is transcribed, but the arrays have not been compared
> element-wise. Worth doing before trusting it.

`StepScale` has 5 entries for 4 heads because head `k` needs `s_{k+1}`. Note the last head's grid
runs slightly past 1.0 (`cy` up to 1.04) before the clamp.

### 2.2 Decode

`/home/keyong/DMTrain/lib/utils/box_utils.py:387`, with `VARIANCE = [0.1, 0.2]`
(`config_parse.py:161`):

```
cx = prior_cx + loc[0] * var[0] * prior_w
cy = prior_cy + loc[1] * var[0] * prior_h
w  = prior_w * exp(loc[2] * var[1])
h  = prior_h * exp(loc[3] * var[1])
```

This is **exactly** ncnn's `DetectionOutput` (`detectionoutput.cpp:177-185`), whose default
variances `(0.1, 0.1, 0.2, 0.2)` are the same numbers split per-coordinate. So if you ever want the
decode in-graph, `DetectionOutput` + the 48,260 priors baked in as a constant `MemoryData` blob is a
drop-in (~1.5 MB added to the `.bin`). `PriorBox` cannot generate FORMAT4 anchors — bake them.

Clamp decoded boxes to the image and drop degenerate ones before cropping.

---

## 3. The two preprocessing contracts

Different for each stage. None of it fails loudly when wrong.

| | stage 1 detector | stage 2 open-set |
|---|---|---|
| input blob | `input.1` | `in0` |
| size | 600 × 800, squash | 128 × 128, squash |
| channel order | BGR | BGR |
| normalize | subtract `(103.94, 116.78, 123.68)`, **no scale** | `(x − 127.5) / 127.5` |
| resize filter | INTER_LINEAR / INTER_CUBIC | **INTER_AREA** |
| outputs | `np_loc`, `np_score` | `out0` dists, `out1` margins |

Detector values from `data_augment.py:497` (`preproc_for_test`) and `config_parse.py:206`
(`PIXEL_MEANS`, BGR-ordered). **Caveat:** that is the DMTrain default config; the specific yml used
for the 20260816 build was not located. Confirm before shipping.

Because the resize is a **squash** and not a letterbox, mapping normalized detector output back to
photo pixels is just `x_photo = x_norm · W_photo`, `y_photo = y_norm · H_photo`. No padding offset,
no aspect correction.

Stage 2 decision: `best = argmin(dists)`, `unknown = margins[best] > 0`. The threshold (0.11705, a
fixed **28°** angular margin) is already baked into `margins`, so the client only checks the sign.
12 of the 83 classes have no training data (`empty_classes`).

---

## 4. Can the two be fused into one network?

Explored at length. Summary: **the crop boundary is the blocker, and it is a runtime property, not
an authoring one.** You can express the whole flow in PyTorch or ONNX; whether it runs depends
entirely on the target.

### ncnn — no

- `ROIAlign` takes **one** ROI: `roialign.cpp:74` reads `bottom_blobs[1]` as a single box and
  returns a single `pooled_w × pooled_h × C` output. No batch-of-ROIs form.
- `Mat` is CHW with **no batch dimension**, so `[N, 3, 128, 128]` has nowhere to live.
- No standalone NMS op (only welded inside `DetectionOutput`), and **no `ArgMin`** — which is the
  entire reason `openset.py` has the `--no-argmin` export mode.

Merging the two `.param` files into one `Net` is mechanically trivial — **0 blob-name collisions**,
and the only 10 layer-name collisions are generated `splitncnn_*` — but buys nothing at runtime.

### LibTorch — yes, fully

`torch.jit.script` handles the data-dependent control flow; `torchvision.ops.nms` and `roi_align`
are both scriptable. `_OpenSetWrapper` (`openset.py:640`) already batches on the `no_argmin=True`
path — `feat [N,D]`, `[N,D]@[D,C]` → `[N,C]`, no batch-1 assumptions. The **other** branch does not:
`feat.squeeze(0)` at line 691 hardcodes batch 1.

### ONNX — expressible; portability is the problem

All ops exist. `roi_align` with `aligned=True` needs **opset ≥ 16**
(`coordinate_transformation_mode="half_pixel"`); below that torchvision emulates it by shifting ROIs,
which gives a different result. Export at opset 17.

The real limit is that **`NonMaxSuppression` has a data-dependent output shape**, so everything
downstream carries an unresolvable dynamic N. ORT CPU/CUDA is fine; the TensorRT EP partitions and
falls back to CPU mid-graph; mobile NPU compilers reject it. Fix: `TopK(K)` → NMS with
`max_output_boxes_per_class=K` → pad to exactly K, and emit a validity mask. Static shapes
throughout.

### MNN — the one edge runtime that could do it

- Real batch dimension; change it at runtime with `resizeTensor()` then `resizeSession()`.
- **`ROIAlign` takes `[N,4]`/`[N,5]` → `[N, C, pooledH, pooledW]`** (`CPUROIAlign.cpp`,
  `numROI = roiTensor->batch()`). Its `RoiParameters` carries `samplingRatio` and `aligned`, so it
  maps cleanly from `torchvision.ops.roi_align`.
- **`CropAndResize`** → `[N, cropH, cropW, depth]` (`CPUCropAndResize.cpp`), but **bilinear/nearest
  only** — no area mode, so prefer `ROIAlign`.
- `OpType` also has `NonMaxSuppression`/`V2`, `DetectionOutput`, `PriorBox`, `TopKV2`, `Gather`,
  `Interp`, and **`ArgMin`** (so `--no-argmin` becomes unnecessary).
- Caveats: `PriorBox` is Caffe-style and cannot produce FORMAT4 anchors (bake them); every change in
  N costs a `resizeSession` reallocation, so pin K.

### A trick if you ever do fuse

ROIAlign is a weighted average with weights summing to 1, so a per-channel offset **commutes** with
it: `ROIAlign(x − m) == ROIAlign(x) − m`. Feed the *detector-normalized* photo once, share that same
tensor with ROIAlign, and recover the classifier's normalization on the small `[N,3,128,128]` tensor
with a single per-channel affine:

```
scale = 1/127.5   (all channels)
bias  = (m_c − 127.5)/127.5   ->   B −0.18478,  G −0.08408,  R −0.02996
```

Exact, not an approximation. One image buffer instead of two, and the normalize runs on ~50K
elements per crop instead of the full frame.

---

## 5. Resolution: crops vs. what the prototypes were built on

Sampled 400 of the 542 images in `posmlvx/train/`:

```
width   min 120 / median 133 / max 156
height  min 255 / median 256 / max 256
82% have min(w,h) >= 128
```

So the prototypes were built from roughly **133 × 256** crops squashed to 128×128 — about 1:1 in
width and a clean **2× downscale** in height.

If input photos are all 600×800 and detected boxes come out near 130×250, the deployed crops match
the training distribution and interpolation choice barely matters. If boxes come out much smaller
(say 40×90), you are **upscaling** into 128×128 and feeding the classifier less detail than the
prototypes were built from — a train/test resolution gap that no `samplingRatio` fixes.

**Measure the box-size distribution the detector actually produces on real 600×800 photos and
compare it to 133×256.** That single histogram says more about whether this pipeline holds up than
anything else here.

Related: the 28° threshold is a *fixed angular margin*, not a fitted quantile, so any systematic
distance shift eats straight into the accept/reject boundary. If you change the resize path, re-score
against `eval_rounds.csv` (which was produced through the cv2 `INTER_AREA` path) rather than
inheriting the threshold. Footnote: `cv2.INTER_AREA` degenerates to nearest-neighbour when
*upsampling*, so if most crops end up upscaled, the "match INTER_AREA" goal inverts.

---

## 6. Where this got to, and what is next

**Current lean: keep the two stages split, crop box-by-box, classify one at a time.** It is the exact
path the prototypes and threshold were calibrated on (cv2 crop + INTER_AREA squash), it keeps the
stages decoupled — which matters because `export_ncnn.sh` rebuilds prototypes and recalibrates the
threshold on every training run — and it runs on ncnn today with `predict_ncnn.py` / `.cpp` already
correct for stage 2.

**The open question is throughput at ~100 boxes per photo.**

Host numbers were taken (`bench_ncnn`, 100 crops, 128×128, area resize, ncnn 20260526, AVX512,
OpenMP on) but are from an **AMD Threadripper 3960X and are not representative of the target**:

```
threads=1   forward 200.99 ms (2.010 each)   resize 14.82 (0.148)   Mat+norm 1.64 (0.016)   TOTAL 217.52
threads=2   forward 249.21                                                                   TOTAL 280.11
threads=4   forward 201.03                                                                   TOTAL 230.28
threads=8   forward 212.12                                                                   TOTAL 242.58
threads=16  forward 217.37                                                                   TOTAL 247.44
```

Two things there might transfer and are cheap to re-check on device: **forward dominates** (~92% of
per-crop cost, so crop/resize strategy is irrelevant to speed), and **more threads bought nothing**
(the model is too small per-layer for OpenMP to pay off). PyTorch on 1 thread was *worse* batched
than sequential at this size (batch=1 6.80 ms; batch=100 1551 ms, 0.44×), which if it holds would
undercut the whole batching premise — but again, wrong silicon.

### Next step — Android device benchmark

Target is **Android**. Plan agreed but not yet implemented:

1. **Extend `nptools/predict/bench_ncnn.cpp`** to load *both* sections straight out of the `.npnn`
   (the split logic is ~20 lines: find `Content:`, then walk the param line count to find where the
   `.bin` starts — mirror `make_npnn.py:_split_npnn` / `_param_text_len`) and report a **stage-1 vs
   stage-2 split**. If the 600×800 SSD forward costs more than 100 crops do, the batching question
   is moot.
2. **Add an NDK toolchain path to `nptools/predict/CMakeLists.txt`.** It currently hard-requires
   OpenCV 4 (`find_package(OpenCV 4 REQUIRED)`), which is heavy for an `adb push`-able binary.
   Suggested: drop OpenCV on the device build in favour of `stb_image` (ncnn bundles it) plus a
   hand-rolled box-filter resize, so the result is one self-contained static ARM binary.
3. **Sweep `ncnn::set_cpu_powersave(0|1|2)`** (all / little / big) — on a phone, big.LITTLE
   placement is likely the largest single factor, far more than `num_threads`.
4. Feed it **realistic crop sizes taken from actual detections**, not training images.

`opts_ncnn.cpp` already exists to sweep ncnn's optimisation flags for both speed and numerical drift,
and its header notes it is meant to be run on the deployment device since the fp16 flags are ARM-only
and inert on x86. Run it there too.
