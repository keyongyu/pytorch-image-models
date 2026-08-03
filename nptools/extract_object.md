# Object extraction from videos (extract_object.py)

`nptools/extract_object.py` is a one-shot pipeline: **videos in → clean object cutouts out**. For
every frame it runs **extract → two-stage cutout → QC** and writes the survivors as two aligned
files (a background-removed PNG and the same crop with its original background).

It was built for the posm_18 standee footage, where the discriminating feature is a **thin metal
foot** that ordinary background removal (run on a tight crop) deletes. The two-stage approach keeps
that foot; the QC pass drops the frames where segmentation fails.

---

## 1. Usage

```bash
uv run python nptools/extract_object.py --video-folder <videos> --dest-dir <out> [options]
```

### CLI arguments

| arg | default | meaning |
|---|---|---|
| `--video-folder` | **required** | folder of videos (searched recursively; mp4/mov/avi/mkv/m4v/webm) |
| `--dest-dir` | **required** | output root for the cutout PNGs and background JPGs |
| `--model` | `birefnet-massive` | rembg model. `birefnet-massive` best preserves the thin foot |
| `--pad` | `80` | white border added before removal so edge-touching parts aren't clipped |
| `--margin` | `0.10` | expand the object bbox by this fraction **on each side** before cropping. bbox share of the png = `1/(1+2·margin)²` (0.10 → ~70%; 0.12 → ~65%) |
| `--close` | `21` | morphological-close kernel (px) that keeps the foot attached to the board |
| `--step` | `1` | process every Nth frame (1 = all) |
| `--max-partial` | `0.20` | QC: reject if `partial_ratio` (ghosting) exceeds this |
| `--min-solid` | `0.28` | QC: reject if `solid_frac` (opaque object area) is below this |
| `--flat-out` | off | write kept pairs **flat** into `--dest-dir` and **discard** rejects |
| `--overwrite` | off | reprocess even if the output already exists |

```bash
# per-video subfolders, rejects quarantined under _rejected/
uv run python nptools/extract_object.py --video-folder posmlv/posm_18videos --dest-dir posmlv/objects

# flat output, keep only clean cutouts (rejects discarded)
uv run python nptools/extract_object.py --video-folder posmlv/posm_18videos --dest-dir posmlv/objects --flat-out
```

---

## 2. The pipeline (per frame)

```
video ─► [0] GPU setup ─► [1] extract frame ─► [2] two-stage cutout ─► [3] QC ─► [4] write pair
```

### [0] GPU setup — `_ensure_cuda_libs()`
Runs at import: puts the pip NVIDIA libs on `LD_LIBRARY_PATH` and re-execs once, so onnxruntime
finds cuDNN and uses the GPU (~0.5 s/pass) instead of silently falling back to CPU (~12 s/pass).

### [1] Extract — `process_video()`
Opens each video with `cv2.VideoCapture` and reads frames **sequentially** (accurate; `POS_FRAMES`
seeking lands on the nearest keyframe and is unreliable). Every `--step`-th frame is converted
BGR→RGB. Frames are processed **in memory** — nothing is dumped to disk.

### [2] Two-stage cutout — `two_stage()`
This is the core, and why it exists: rembg is salient-object detection, so on a **tight crop** it
drops low-salience parts like the foot. Running on the full frame keeps the foot but downscales the
object; running twice gets both.

- **Stage 1 (locate):** `rembg_padded()` on the *full frame* → coarse mask; `largest_blob()` → the
  object's bounding box; crop the original frame to `bbox + margin` → `crop1` (RGB, **background
  still intact**).
- **Stage 2 (refine):** rembg again on `crop1`. The object now fills the model's ~1024 px input
  instead of being downscaled inside a 1080×1920 frame, so **thin structures (the foot) survive**.
  `largest_blob()` again → zero the alpha of every other blob (drops separated clutter) → crop tight
  to `bbox + margin`.
- Returns two **pixel-aligned** images of the same crop:
  - `cutout_rgba` — background removed (transparent),
  - `cutout_jpg` — `crop1` cropped the same way, **background kept**.

`largest_blob()` first applies a morphological **close** (`--close`) so the thin foot stays joined to
the board, then selects the maximum-area component. This is why *separated* small items get dropped
but items *touching* the board do not (they form one blob).

### [3] QC — `qc_is_bad()`
Looks only at the cutout's **alpha channel** (content-agnostic):

- `partial_ratio` = semi-transparent px / present px → high ⇒ **ghosted** object.
- `solid_frac`    = opaque px / whole image     → low  ⇒ **wrong / empty** object.
- **Bad** if `partial_ratio > --max-partial` **OR** `solid_frac < --min-solid`.

Thresholds are calibrated to separate good (`partial≈0.04`, `solid≈0.43`) from bad
(`partial>0.6`, `solid<0.18`) cutouts, with a wide gap between — so the exact cutoff isn't fragile.
QC catches **ghosted / wrong-object** failures; it does **not** catch clutter that touches the
object (see limitation below).

### [4] Write
For each frame that yielded an object:

| mode | kept → | rejected → |
|---|---|---|
| default | `--dest-dir/<video>/` | `--dest-dir/_rejected/<video>/` |
| `--flat-out` | `--dest-dir/` (flat) | **discarded** |

Filenames: `nobg_<video>_<frame>_.png` (cutout) + `<video>_<frame>_.jpg` (same crop, background
kept). The stem carries the video name, so flat output never collides across videos.

---

## 3. Output

Per kept frame you get an **aligned pair** — the transparent cutout and the original-background crop
of the identical region:

```
posmlv/objects/
  posm_18_Blanc Stand Board_1/
    nobg_posm_18_Blanc Stand Board_1_0_.png     # background removed
    posm_18_Blanc Stand Board_1_0_.jpg          # same crop, background intact
    ...
  _rejected/                                    # (omitted with --flat-out)
    posm_18_Blanc Stand Board_1/ ...
```

`main()` prints per-video and total `kept / rejected / no-object` counts and the elapsed time.

---

## 4. Known limitation (by design)

Clutter that physically **touches** the object (a shelf packet against the board, a dispenser stand
whose legs bridge to it) fuses into the **same mask blob**, so neither `largest_blob` nor QC removes
it — the cutout's alpha is fine, it just contains an extra thing. Separating touching objects needs
instance segmentation (SAM-style), which is unreliable on this footage. QC only removes ghosted /
wrong-object failures.

---

## TL;DR

```bash
# extract every frame of every video → clean board+foot cutouts (+ background jpgs), rejects quarantined
uv run python nptools/extract_object.py --video-folder posmlv/posm_18videos --dest-dir posmlv/objects
# add --flat-out for a single flat folder with rejects discarded
```

Pipeline: **extract frame → two-stage rembg (keeps the foot) → alpha-based QC → aligned PNG+JPG
pair.** GPU is automatic; ~0.5 s/pass on an RTX 4090.
