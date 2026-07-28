"""Visual + smoke test for _NoBgTrainTransform.

Runs the transform several times (augmentation is random) on a background-removed RGBA PNG
and saves the augmented RGB outputs to kytest/nobg_aug_out/ for visual inspection.

Run:
    uv run --no-sync python kytest/test_nobg_transform.py
    uv run --no-sync python kytest/test_nobg_transform.py /path/to/nobg_x.png
"""
import os
import sys
import glob

import numpy as np
import torch
from PIL import Image

from timm.data.npaug import _NoBgTrainTransform

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nobg_aug_out')
IMG_SIZE = 224
N_SAMPLES = 8


def _find_input():
    """Prefer a real background-removed PNG from the datasets."""
    roots = [
        '/home/keyong/cls2/code/pepsi_4types',
        '/home/keyong/cls2/code/pepsi_merge',
    ]
    for r in roots:
        hits = glob.glob(os.path.join(r, '**', 'nobg_*.png'), recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def _synth_rgba(size=(200, 320)):
    """Fallback input: a bottle-ish colored shape on a transparent background."""
    w, h = size
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    y0, y1 = int(h * 0.10), int(h * 0.90)
    rgba[y0:y1, x0:x1, 2] = 200          # blue body
    rgba[y0:y1, x0:x1, 3] = 255          # opaque
    rgba[int(h * 0.45):int(h * 0.55), x0:x1, 0] = 220  # red stripe
    rgba[int(h * 0.45):int(h * 0.55), x0:x1, 2] = 40
    return Image.fromarray(rgba, mode='RGBA')


def _load_input(img_path=None):
    img_path = img_path or _find_input()
    if img_path and os.path.isfile(img_path):
        return Image.open(img_path).convert('RGBA'), img_path
    return _synth_rgba(), 'synthetic RGBA'


def make_samples(img_path=None, out_dir=OUT_DIR, n=N_SAMPLES):
    os.makedirs(out_dir, exist_ok=True)
    img, src = _load_input(img_path)
    print(f'input: {src} | mode {img.mode} | size {img.size}')

    # normalize=False -> _finalize_tensor returns a uint8 CHW tensor, easy to visualize
    tf = _NoBgTrainTransform(IMG_SIZE, mean=(0.5,) * 3, std=(0.5,) * 3,
                             normalize=False, use_prefetcher=False)

    for i in range(n):
        t = tf(img)                                             # uint8 CHW RGB
        arr = t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # HWC RGB
        Image.fromarray(arr).save(os.path.join(out_dir, f'aug_{i:02d}.png'))
    print(f'saved {n} augmented samples to {out_dir}')
    return out_dir


def test_nobg_train_transform_outputs():
    """Smoke test: correct shape/dtype/range for both uint8 and normalized paths."""
    img = _synth_rgba()

    tf_u8 = _NoBgTrainTransform(IMG_SIZE, mean=(0.5,) * 3, std=(0.5,) * 3,
                                normalize=False, use_prefetcher=False)
    t = tf_u8(img)
    assert t.shape == (3, IMG_SIZE, IMG_SIZE)
    assert t.dtype == torch.uint8
    assert int(t.min()) >= 0 and int(t.max()) <= 255

    tf_norm = _NoBgTrainTransform(IMG_SIZE, mean=(0.5,) * 3, std=(0.5,) * 3,
                                  normalize=True, use_prefetcher=False)
    t2 = tf_norm(img)
    assert t2.shape == (3, IMG_SIZE, IMG_SIZE)
    assert t2.dtype == torch.float32


if __name__ == '__main__':
    make_samples(sys.argv[1] if len(sys.argv) > 1 else None)
