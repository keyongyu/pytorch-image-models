"""Visualize the effect of build_aug_pipeline (the full heavy geometric + photometric pipeline).

Applies build_aug_pipeline directly to input image(s) N times and saves the augmented outputs
to kytest/build_aug_out/<stem>/ so you can see what this pipeline does (it distorts geometry:
perspective/elastic/grid/rotation + aspect-changing resize/crop, plus photometric/noise/blur).

A fixed SEED makes the augmentations reproducible, so a specific bad sample (e.g. aug_11) can be
reduplicated exactly for debugging. Override with the SEED env var.

Run:
    uv run --no-sync python kytest/test_build_aug.py                 # default sample per class
    uv run --no-sync python kytest/test_build_aug.py img1.jpg ...    # your own inputs
    SEED=123 uv run --no-sync python kytest/test_build_aug.py        # reproducible run
"""
import os
import sys
import glob
import random

import numpy as np
from PIL import Image

from timm.data.npaug import build_aug_pipeline

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_aug_out')
IMG_SIZE = 224
N_PER_IMAGE = 20
SEED = int(os.environ.get('SEED', '0'))  # fixed seed => identical outputs every run


def _default_inputs():
    base = '/home/keyong/cls2/code/pepsi_4types/base/train'
    inputs = []
    for cls in sorted(glob.glob(os.path.join(base, '*'))):
        jpgs = sorted(glob.glob(os.path.join(cls, '*.jpg')))
        if jpgs:
            inputs.append(jpgs[0])
    return inputs


def make_samples(img_paths, out_dir=OUT_DIR, n=N_PER_IMAGE, img_size=IMG_SIZE, seed=SEED):
    # seed everything for reproducibility (python / numpy / albumentations)
    random.seed(seed)
    np.random.seed(seed)
    aug = build_aug_pipeline(img_size=img_size)   # albumentations Compose, outputs img_size square
    aug.set_random_seed(seed)                     # deterministic albumentations RNG
    print(f'seed = {seed}')
    for img_path in img_paths:
        if not os.path.isfile(img_path):
            print(f'skip (not found): {img_path}')
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0]
        sub = os.path.join(out_dir, stem)
        os.makedirs(sub, exist_ok=True)
        rgb = np.array(Image.open(img_path).convert('RGB'))
        for i in range(n):
            out = aug(image=rgb)['image']          # HWC uint8 RGB, img_size x img_size
            Image.fromarray(out).save(os.path.join(sub, f'{stem}_aug_{i:02d}.png'))
        print(f'{img_path} -> {n} files in {sub}')
    print(f'\ndone. outputs under {out_dir}')


def test_build_aug_output():
    """Smoke test: build_aug_pipeline outputs an img_size x img_size RGB uint8 image."""
    rgb = np.array(Image.open(_default_inputs()[0]).convert('RGB'))
    out = build_aug_pipeline(img_size=IMG_SIZE)(image=rgb)['image']
    assert out.shape == (IMG_SIZE, IMG_SIZE, 3), out.shape
    assert out.dtype == np.uint8


if __name__ == '__main__':
    paths = sys.argv[1:] or _default_inputs()
    make_samples(paths)
