"""Sanity-check the exact data fed to training and validation.

Rebuilds the train/val transforms the same way train4.sh does (--nobg --heavy-aug,
--crop-mode=border, native input_img_mode) and:
  * saves a labeled grid of real samples for visual inspection, and
  * verifies each sample's label matches its source folder, and
  * reports per-class counts and jpg-vs-png composition (dispatch inputs).

Run:
    NOBG_BG_DIR=/home/keyong/cls6/bg_photos \
      uv run --no-sync python kytest/test_data_pipeline.py
"""
import os
import random
from collections import Counter

import numpy as np
import torch
from PIL import Image, ImageDraw

from timm.data.dataset import ImageDataset
from timm.data.transforms_factory import create_transform

DATA_DIR = '/home/keyong/cls2/code/pepsi_4types/base'
CLASS_MAP = '/home/keyong/cls2/code/pepsi_4types/class.txt'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_check_out')
IMG = 224
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def build(split, is_training):
    """Transform + dataset matching train4.sh (normalize=False so outputs are viewable uint8)."""
    tf = create_transform(
        input_size=(3, IMG, IMG),
        is_training=is_training,
        nobg=True,
        heavy_aug=is_training,          # heavy aug only for training (matches create_transform)
        crop_pct=1.0,
        train_crop_mode='squash',       # squash: stretch to square (train)
        crop_mode='squash',             # squash: stretch to square (eval)
        mean=MEAN, std=STD,
        normalize=False,                # uint8 output for easy visualization
    )
    ds = ImageDataset(
        os.path.join(DATA_DIR, split),
        class_map=CLASS_MAP,
        input_img_mode=None,            # native mode: keep RGBA + filename for the nobg dispatch
        transform=tf,
    )
    return ds


def _idx_to_name(ds):
    try:
        return {v: k for k, v in ds.reader.class_to_idx.items()}
    except Exception:
        return {}


def _path_of(ds, i):
    try:
        return ds.reader.filename(i, absolute=True)
    except Exception:
        return ''


def _to_hwc_uint8(x):
    if x.dtype == torch.uint8:
        return x.permute(1, 2, 0).numpy().astype(np.uint8)
    arr = x.permute(1, 2, 0).numpy()
    arr = (arr * np.array(STD) + np.array(MEAN)) * 255.0
    return arr.clip(0, 255).astype(np.uint8)


def dump_grid(split, is_training, n=16, cols=4):
    ds = build(split, is_training)
    idx2name = _idx_to_name(ds)
    os.makedirs(OUT_DIR, exist_ok=True)

    idxs = random.sample(range(len(ds)), min(n, len(ds)))
    rows = (len(idxs) + cols - 1) // cols
    canvas = Image.new('RGB', (IMG * cols, IMG * rows), (255, 255, 255))
    mismatches = 0
    for j, i in enumerate(idxs):
        x, y = ds[i]
        name = idx2name.get(int(y), str(int(y)))
        path = _path_of(ds, i)
        ok = name in path                      # label must match source folder
        mismatches += 0 if ok else 1
        im = Image.fromarray(_to_hwc_uint8(x))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, IMG, 16], fill=(0, 0, 0))
        d.text((2, 3), f'{name}  {"OK" if ok else "MISLABEL!"}',
               fill=(0, 255, 0) if ok else (255, 0, 0))
        canvas.paste(im, ((j % cols) * IMG, (j // cols) * IMG))
    out = os.path.join(OUT_DIR, f'{split}_grid.png')
    canvas.save(out)
    print(f'[{split}] dumped {len(idxs)} samples -> {out}  (label mismatches: {mismatches})')
    return ds


def report(ds, split):
    """Per-class counts and jpg/png composition from the reader's sample list."""
    idx2name = _idx_to_name(ds)
    try:
        samples = ds.reader.samples  # list of (path, class_idx)
    except Exception:
        samples = [(_path_of(ds, i), None) for i in range(len(ds))]
    by_class = Counter()
    by_ext = Counter()
    label_ok = 0
    for path, lbl in samples:
        ext = os.path.splitext(path)[1].lower()
        by_ext[ext] += 1
        if lbl is not None:
            name = idx2name.get(int(lbl), str(int(lbl)))
            by_class[name] += 1
            if name in path:
                label_ok += 1
    print(f'\n[{split}] total={len(samples)}  label-folder matches={label_ok}/{len(samples)}')
    print(f'[{split}] per-class: {dict(by_class)}')
    print(f'[{split}] file types: {dict(by_ext)}  '
          f'(png => routed to bg-swap, jpg => standard/heavy)')


def test_labels_match_folders():
    """Every sample's mapped class name must appear in its file path (no label scrambling)."""
    for split in ('train', 'val'):
        ds = build(split, is_training=(split == 'train'))
        idx2name = _idx_to_name(ds)
        samples = ds.reader.samples
        for path, lbl in samples:
            assert idx2name.get(int(lbl), '') in path, f'label/path mismatch: {lbl} vs {path}'


if __name__ == '__main__':
    for split, train in (('train', True), ('val', False)):
        ds = dump_grid(split, train)
        report(ds, split)
