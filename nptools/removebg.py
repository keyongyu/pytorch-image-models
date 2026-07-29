"""Remove image backgrounds recursively using rembg.

Given a split folder `<dataset_root>/<split>` (e.g. pepsi_merge/train) it recursively finds
all images under it and writes each cutout as a transparent PNG (alpha preserved) to
`<dataset_root>/rembg/<split>/<subpath>`, preserving the class subfolders. E.g.:
    pepsi_merge/train/CRM060109/0_4_.jpg -> pepsi_merge/rembg/train/CRM060109/nobg_0_4_.png

Uses `birefnet-general` by default, the highest-quality general-purpose segmentation
network in rembg, which handles difficult objects (glass, transparent bottles) and fine
edges better than isnet/u2net. It is larger (~1GB) and slower; pass --model isnet-general-use
for a faster, lighter option. Runs on GPU automatically if onnxruntime-gpu + CUDA libs present.

Run with (rembg lives in the venv but not the project lockfile):
    uv run --no-sync python kytest/removebg.py pepsi_merge/train [--pad 40] [--model birefnet-general]
"""
import os
import sys
# avoid numba's TBB threading-layer warning (older system TBB); use a safe fallback layer.
os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')


def _ensure_cuda_libs():
    """Put the pip-installed NVIDIA CUDA libs on LD_LIBRARY_PATH so onnxruntime-gpu can load them.

    LD_LIBRARY_PATH is read by the dynamic loader only at process start, so if the libs aren't
    already on it we set the path and re-exec this process once. Without this, onnxruntime fails
    to load libcudnn/libcublas and silently falls back to (very slow) CPU.
    """
    import glob
    import sysconfig
    site = sysconfig.get_paths()['purelib']
    lib_dirs = sorted(glob.glob(os.path.join(site, 'nvidia', '*', 'lib')))
    if not lib_dirs:
        return
    current = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    if all(d in current for d in lib_dirs):
        return  # already configured (this is the re-exec'd process)
    os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_dirs + [p for p in current if p])
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_cuda_libs()

import argparse
import warnings

try:
    from numba.core.errors import NumbaWarning
    warnings.filterwarnings('ignore', category=NumbaWarning)
except ImportError:
    pass

from PIL import Image
from rembg import new_session, remove

import re

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
PREFIX = 'nobg_'


def _natural_key(name: str):
    """Sort key that orders embedded numbers numerically (0_2_ < 0_10_ < 0_100_)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def iter_images(root: str):
    """Recursively yield image paths under root, relative to root, in natural sort order."""
    rel_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort(key=_natural_key)  # deterministic descent
        for name in filenames:
            if name.startswith(PREFIX):
                continue  # skip already-processed outputs
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                rel_paths.append(os.path.relpath(os.path.join(dirpath, name), root))
    rel_paths.sort(key=_natural_key)
    yield from rel_paths


def resolve_output_root(folder: str) -> str:
    """Map <dataset_root>/<split> -> <dataset_root>/rembg/<split>."""
    folder = os.path.abspath(folder)
    split_name = os.path.basename(folder)              # e.g. train
    dataset_root = os.path.dirname(folder)             # e.g. <dataset_root>
    return os.path.join(dataset_root, 'rembg', split_name)


def remove_bg_image(img: Image.Image, session, pad: int, fill=None) -> Image.Image:
    """Remove background from a single RGB image.

    Returns an RGBA image with a transparent background (saved as PNG to keep the alpha).
    If `fill` is given, the cutout is instead flattened onto that solid color (opaque RGB).

    If pad > 0, a white border is added before segmentation so objects touching the image
    edge (e.g. a bottle cap at the top) aren't truncated, then the result is cropped back to
    the original size.
    """
    w, h = img.size
    src = img
    if pad > 0:
        # pad with white so the border region reads as background to the model
        src = Image.new('RGB', (w + 2 * pad, h + 2 * pad), (255, 255, 255))
        src.paste(img, (pad, pad))

    cutout = remove(src, session=session)          # RGBA, background is transparent
    if pad > 0:
        cutout = cutout.crop((pad, pad, pad + w, pad + h))  # crop back to original size

    if fill is not None:
        # flatten onto a solid opaque background (drops alpha)
        bg = Image.new('RGB', cutout.size, tuple(fill))
        bg.paste(cutout, mask=cutout.split()[3])   # use alpha channel as paste mask
        return bg
    return cutout                                  # RGBA, transparent background


def remove_bg_folder(folder: str, model: str, bg_color, overwrite: bool, pad: int, inplace: bool):
    session = new_session(model)
    # bg_color=None -> transparent PNG (keep alpha); otherwise flatten onto this solid color
    fill = tuple(bg_color) if bg_color is not None else None

    folder = os.path.abspath(folder)
    rel_paths = list(iter_images(folder))
    if not rel_paths:
        print(f'No images found under {folder}')
        return

    # inplace: write nobg_ files next to each source image; otherwise use the rembg/ output tree
    out_root = folder if inplace else resolve_output_root(folder)

    fill_desc = fill if fill is not None else 'transparent (alpha)'
    print(f'Model: {model} | background: {fill_desc} | pad: {pad}px | inplace: {inplace} | {len(rel_paths)} images')
    print(f'Input root:  {folder}')
    print(f'Output root: {out_root}\n')
    for i, rel in enumerate(rel_paths, 1):
        src = os.path.join(folder, rel)
        rel_dir, name = os.path.split(rel)
        stem = os.path.splitext(name)[0]
        out_dir = os.path.join(out_root, rel_dir)          # preserve subfolder (e.g. class)
        # PNG output so the transparent alpha channel is preserved (JPEG can't hold alpha)
        dst = os.path.join(out_dir, f'{PREFIX}{stem}.png')

        if os.path.exists(dst) and not overwrite:
            print(f'[{i}/{len(rel_paths)}] skip (exists): {rel}')
            continue

        os.makedirs(out_dir, exist_ok=True)
        img = Image.open(src).convert('RGB')
        result = remove_bg_image(img, session, pad, fill)
        result.save(dst)
        print(f'[{i}/{len(rel_paths)}] {rel} -> {os.path.join(rel_dir, os.path.basename(dst))}')


def main():
    parser = argparse.ArgumentParser(description='Remove image backgrounds recursively (rembg)')
    parser.add_argument('folder', help='split folder to process recursively, e.g. pepsi_merge/train')
    parser.add_argument('--model', default='birefnet-general',
                        help='rembg model name (default: birefnet-general, highest quality)')
    parser.add_argument('--bg-color', type=int, nargs=3, default=None, metavar=('R', 'G', 'B'),
                        help='if set, flatten the cutout onto this solid opaque color (drops alpha); '
                             'by default the output is a transparent PNG that keeps the alpha channel')
    parser.add_argument('--overwrite', action='store_true',
                        help='reprocess even if the nobg_ output already exists')
    parser.add_argument('--pad', type=int, default=0, metavar='PX',
                        help='pad each image with a white border of PX pixels before removal so '
                             'edge-touching objects (e.g. a bottle cap) are not truncated; the '
                             'result is cropped back to the original size (default: 0, off)')
    parser.add_argument('--inplace', action='store_true',
                        help='save each nobg_ output next to its source image (same folder) instead '
                             'of writing to the separate rembg/ output tree')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise NotADirectoryError(args.folder)

    remove_bg_folder(args.folder, args.model, args.bg_color, args.overwrite, args.pad, args.inplace)


if __name__ == '__main__':
    main()
