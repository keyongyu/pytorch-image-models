"""Visual test for _HeavyAugTrainTransform.

For each input image, runs the heavy augmentation pipeline N times (default 20, random each
time) and saves the augmented RGB outputs to kytest/heavy_aug_out/<stem>/ for visual inspection.

Run:
    uv run --no-sync python kytest/test_heavy_aug.py                 # default input(s)
    uv run --no-sync python kytest/test_heavy_aug.py img1.jpg img2.jpg ...
"""
import os
import sys
import glob

import numpy as np
import torch
from PIL import Image

from timm.data.npaug import _HeavyAugTrainTransform

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'heavy_aug_out')
IMG_SIZE = 224
N_PER_IMAGE = 20


def _default_inputs():
    """One sample per product class from pepsi_4types (jpg originals)."""
    base = '/home/keyong/cls2/code/pepsi_4types/base/train'
    inputs = []
    for cls in sorted(glob.glob(os.path.join(base, '*'))):
        jpgs = sorted(glob.glob(os.path.join(cls, '*.jpg')))
        if jpgs:
            inputs.append(jpgs[0])
    return inputs


def make_samples(img_paths, out_dir=OUT_DIR, n=N_PER_IMAGE):
    # normalize=False -> _finalize_tensor returns uint8 CHW, easy to visualize
    tf = _HeavyAugTrainTransform(IMG_SIZE, mean=(0.5,) * 3, std=(0.5,) * 3,
                                 normalize=False, use_prefetcher=False)
    for img_path in img_paths:
        if not os.path.isfile(img_path):
            print(f'skip (not found): {img_path}')
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0]
        sub = os.path.join(out_dir, stem)
        os.makedirs(sub, exist_ok=True)
        img = Image.open(img_path)
        for i in range(n):
            t = tf(img)                                              # uint8 CHW RGB
            arr = t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)   # HWC RGB
            Image.fromarray(arr).save(os.path.join(sub, f'{stem}_aug_{i:02d}.png'))
        print(f'{img_path} -> {n} files in {sub}')
    print(f'\ndone. outputs under {out_dir}')


def test_heavy_aug_outputs():
    """Smoke test: correct shape/dtype/range."""
    img = Image.open(_default_inputs()[0])
    tf = _HeavyAugTrainTransform(IMG_SIZE, mean=(0.5,) * 3, std=(0.5,) * 3,
                                 normalize=False, use_prefetcher=False)
    t = tf(img)
    assert t.shape == (3, IMG_SIZE, IMG_SIZE)
    assert t.dtype == torch.uint8
    assert int(t.min()) >= 0 and int(t.max()) <= 255


if __name__ == '__main__':
    paths = sys.argv[1:] or _default_inputs()
    make_samples(paths)
