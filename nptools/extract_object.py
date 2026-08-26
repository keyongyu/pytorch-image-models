"""End-to-end object extraction from videos: frame extraction -> two-stage cutout -> QC.

For every video under ``--video-folder`` this reads frames, cuts out the main object with a
two-stage rembg pass (keeps low-salience parts like a standee's foot), runs a quality-control
check on the resulting mask, and for every frame that PASSES QC writes two aligned files into
``--dest-dir/<video_stem>/``:

  - ``nobg_<video>_<frame>_.png`` -- the RGBA cutout (background removed, transparent).
  - ``<video>_<frame>_.jpg``      -- the SAME crop region taken from the ORIGINAL frame, i.e. the
                                     object WITH its real background still intact.

Both files share the exact same crop box and size, so they are pixel-aligned (one masked, one not).
Frames that FAIL QC (ghosted object / wrong object) are written under ``--dest-dir/_rejected/...``
instead, so the main output contains only clean cutouts.

Pipeline detail:
  1. extract   -- decode frames sequentially (``--step`` to subsample).
  2. two-stage -- rembg on the full frame -> object bbox; crop original to bbox+margin; rembg again
                  on the crop (object fills the model input -> cleaner thin parts); keep the largest
                  blob (a morphological close keeps the foot attached); crop tight to it + margin.
  3. QC        -- from the cutout alpha: reject if ``partial_ratio`` (ghosting) is high OR
                  ``solid_frac`` (opaque object area) is low.

GPU is used automatically once the pip NVIDIA libs are on LD_LIBRARY_PATH (handled below).

Run:
    uv run python nptools/extract_object.py --video-folder posmlv/posm_18videos --dest-dir posmlv/objects
"""
import os
import sys
os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')


def _ensure_cuda_libs():
    """Put pip NVIDIA CUDA libs on LD_LIBRARY_PATH so onnxruntime-gpu loads cuDNN (else CPU).

    LD_LIBRARY_PATH is read by the loader only at process start, so if the libs aren't already on
    it we set the path and re-exec this process once.
    """
    import glob as _glob
    import sysconfig
    site = sysconfig.get_paths()['purelib']
    lib_dirs = sorted(_glob.glob(os.path.join(site, 'nvidia', '*', 'lib')))
    if not lib_dirs:
        return
    current = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    if all(d in current for d in lib_dirs):
        return  # already configured (this is the re-exec'd process)
    os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_dirs + [p for p in current if p])
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_cuda_libs()

import argparse
import time
import warnings

try:
    from numba.core.errors import NumbaWarning
    warnings.filterwarnings('ignore', category=NumbaWarning)
except ImportError:
    pass

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}
PREFIX = 'nobg_'
REJECT_DIRNAME = '_rejected'


# ── stages ───────────────────────────────────────────────────────────────────────

def rembg_padded(img, session, pad):
    """rembg on a white-padded image so edge-touching parts aren't truncated; crop pad back off."""
    w, h = img.size
    if pad > 0:
        canvas = Image.new('RGB', (w + 2 * pad, h + 2 * pad), (255, 255, 255))
        canvas.paste(img, (pad, pad))
        return remove(canvas, session=session).crop((pad, pad, pad + w, pad + h))
    return remove(img, session=session)


def largest_blob(alpha, close):
    """Return (keep_mask, (x, y, w, h)) for the biggest connected foreground blob, or (None, None).

    A morphological close bridges thin gaps first so a slightly-detached foot stays with the board.
    """
    m = (alpha > 10).astype(np.uint8)
    if not m.any():
        return None, None
    if close:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    box = (int(stats[biggest, cv2.CC_STAT_LEFT]), int(stats[biggest, cv2.CC_STAT_TOP]),
           int(stats[biggest, cv2.CC_STAT_WIDTH]), int(stats[biggest, cv2.CC_STAT_HEIGHT]))
    return labels == biggest, box


def _expand(box, margin, w, h):
    """Expand an (x, y, bw, bh) box by `margin` fraction, clamped to (w, h); return PIL crop tuple."""
    x, y, bw, bh = box
    mx, my = int(bw * margin), int(bh * margin)
    return (max(0, x - mx), max(0, y - my), min(w, x + bw + mx), min(h, y + bh + my))


def two_stage(img, session, pad, margin, close):
    """Two-stage cutout of the main object.

    Returns (cutout_rgba, cutout_jpg_rgb) where both are the SAME tight crop region -- the PNG is
    background-removed, the JPG keeps the original background. Returns (None, None) if no object.
    """
    W, H = img.size
    coarse = rembg_padded(img, session, pad)
    _, box = largest_blob(np.array(coarse)[:, :, 3], close)
    if box is None:
        return None, None
    crop_box1 = _expand(box, margin, W, H)
    crop1 = img.crop(crop_box1)                      # RGB with original background

    fine = rembg_padded(crop1, session, pad)
    arr = np.array(fine)
    keep, box = largest_blob(arr[:, :, 3], close)
    if box is None:
        return None, None
    arr[:, :, 3] = np.where(keep, arr[:, :, 3], 0)   # drop separate small items
    Wc, Hc = crop1.size
    crop_box2 = _expand(box, margin, Wc, Hc)
    cutout_rgba = Image.fromarray(arr).crop(crop_box2)
    cutout_jpg = crop1.crop(crop_box2)               # same region, background intact
    return cutout_rgba, cutout_jpg


def qc_is_bad(cutout_rgba, max_partial, min_solid):
    """QC on the cutout alpha: True (bad) if ghosted (high partial_ratio) or too little solid object."""
    a = np.array(cutout_rgba)[:, :, 3]
    present = a >= 10
    npre = int(present.sum())
    partial_ratio = (int((present & (a < 200)).sum()) / npre) if npre else 1.0
    solid_frac = float((a >= 200).mean())
    return partial_ratio > max_partial or solid_frac < min_solid


# ── driver ───────────────────────────────────────────────────────────────────────

def find_videos(folder):
    vids = []
    for dirpath, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                vids.append(os.path.join(dirpath, f))
    return sorted(vids)


def process_video(video_path, dest_dir, session, args):
    """Extract, cut out, QC every (step-th) frame of one video; write kept/rejected pairs."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'  WARN could not open {video_path}')
        return 0, 0, 0
    jpg_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    idx = kept = rejected = noobj = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % args.step == 0:
            img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            cutout_rgba, cutout_jpg = two_stage(img, session, args.pad, args.margin, args.close)
            if cutout_rgba is None:
                noobj += 1
            else:
                bad = qc_is_bad(cutout_rgba, args.max_partial, args.min_solid)
                rejected += bad
                kept += (not bad)
                # --flat-out: write kept pairs flat into dest_dir and discard rejects entirely.
                # otherwise: kept -> dest_dir/<video>/, rejected -> dest_dir/_rejected/<video>/.
                if bad and args.flat_out:
                    sub = None  # discard
                elif args.flat_out:
                    sub = dest_dir
                elif bad:
                    sub = os.path.join(dest_dir, REJECT_DIRNAME, stem)
                else:
                    sub = os.path.join(dest_dir, stem)
                if sub is not None:
                    os.makedirs(sub, exist_ok=True)
                    png_path = os.path.join(sub, f'{PREFIX}{stem}_{idx}_.png')
                    jpg_path = os.path.join(sub, f'{stem}_{idx}_.jpg')
                    if args.overwrite or not os.path.exists(png_path):
                        cutout_rgba.save(png_path)
                        cutout_jpg.convert('RGB').save(jpg_path, quality=95)
        idx += 1
    cap.release()
    print(f'  {stem}: {idx} frames -> {kept} kept, {rejected} rejected, {noobj} no-object')
    return kept, rejected, noobj


def main():
    p = argparse.ArgumentParser(description='Video -> object cutouts (extract + two-stage rembg + QC)')
    p.add_argument('--video-folder', required=True, help='folder of videos (searched recursively)')
    p.add_argument('--dest-dir', required=True, help='output root for cutouts and jpg crops')
    p.add_argument('--model', default='birefnet-massive', help='rembg model (default: birefnet-massive)')
    p.add_argument('--pad', type=int, default=80, help='white pad before removal, px (default: 80)')
    p.add_argument('--margin', type=float, default=0.02,
                   help='bbox expansion fraction per side (default: 0.02 -> object fills 1/1.04^2 ~= 92%% of the png)')
    p.add_argument('--close', type=int, default=21, help='morphological close kernel px (default: 21)')
    p.add_argument('--step', type=int, default=1, help='process every Nth frame (default: 1 = all)')
    p.add_argument('--max-partial', type=float, default=0.20,
                   help='QC: reject if partial_ratio (ghosting) exceeds this (default: 0.20)')
    p.add_argument('--min-solid', type=float, default=0.28,
                   help='QC: reject if solid_frac (opaque object area) is below this (default: 0.28)')
    p.add_argument('--overwrite', action='store_true', help='reprocess even if the output exists')
    p.add_argument('--flat-out', action='store_true',
                   help='write kept pairs flat into --dest-dir (no per-video subfolder) and DISCARD '
                        'rejected frames instead of quarantining them')
    args = p.parse_args()

    if not os.path.isdir(args.video_folder):
        raise NotADirectoryError(args.video_folder)
    videos = find_videos(args.video_folder)
    if not videos:
        print(f'No videos found under {args.video_folder}')
        return
    os.makedirs(args.dest_dir, exist_ok=True)

    session = new_session(args.model)
    print(f'{len(videos)} videos | model={args.model} step={args.step} margin={args.margin} '
          f'| QC: partial>{args.max_partial} or solid<{args.min_solid}')
    print(f'dest: {args.dest_dir}\n', flush=True)

    t0 = time.time()
    tot_kept = tot_rej = tot_noobj = 0
    for v in videos:
        k, r, n = process_video(v, args.dest_dir, session, args)
        tot_kept += k
        tot_rej += r
        tot_noobj += n
    print(f'\nFINISHED: {tot_kept} kept, {tot_rej} rejected, {tot_noobj} no-object | '
          f'{(time.time() - t0) / 60:.1f} min')
    if args.flat_out:
        print(f'clean cutouts written flat in {args.dest_dir}/ ; {tot_rej} rejected discarded')
    else:
        print(f'clean cutouts in {args.dest_dir}/<video>/, rejects in {args.dest_dir}/{REJECT_DIRNAME}/')


if __name__ == '__main__':
    main()
