"""Pre-resize a training set to the size the augmentation pipeline actually consumes.

Sources here run from 0.01 MP to 10 MP while nothing above ``2 * img_size`` survives
``nptools/npaug.build_aug_pipeline`` (it normalises to a ``2 * img_size`` square before cropping
back). Decoding a 10 MP JPEG and immediately throwing 98% of it away cost more per sample than the
entire augmentation -- measured 10.5 ms of the 14.3 ms per-sample budget. ``PIL.draft()`` in npaug
recovers part of that, but only for images whose BOTH axes exceed the budget: JPEG DCT scaling is
uniform, so an elongated 1737x371 crop (typical of shelf photos) cannot be reduced at all. Resizing
once, offline, is the only fix that is axis-independent.

Default is a 448x448 squash, which matches ``--crop-mode=squash`` and the pipeline's own canvas, so
nothing is lost relative to what training consumes -- and ``_fit_rgb`` becomes a no-op.

Usage:
    python -m nptools.resize_dataset --src posmlvx/train --out posmlvx/train_448
    python -m nptools.resize_dataset --src posmlvx/train --out posmlvx/train_896 --size 896
    python -m nptools.resize_dataset --src posmlvx/train --out posmlvx/train_fit --mode longest

Idempotent: a destination newer than its source is skipped, so re-running (e.g. from
nptools/train_posm.sh on every training launch) only touches images that changed.

``imbaldup_*`` symlinks (see nptools/reduce_imbalance.py) are NOT resized as images -- each is
recreated as a relative symlink to the resized target, so the oversampling ratio survives and the
resized tree stays as small as the originals.

RGBA ``nobg_*.png`` cutouts keep PNG + alpha: the npaug bg-swap path detects them by name and
needs the alpha channel. Everything else is written as JPEG at ``--quality``.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import cv2

from nptools.reduce_imbalance import _IMBAL_PREFIX, _IMG_EXTS


def _fit_size(h: int, w: int, size: int, mode: str) -> tuple:
    """Target (h, w) for one image.

    Args:
        mode: ``squash`` stretches to a ``size`` square (matches --crop-mode=squash, and is the
            only mode that bounds BOTH axes for an elongated crop); ``longest`` caps the longest
            side at ``size`` and keeps the aspect ratio.
    """
    if mode == 'squash':
        return size, size
    scale = size / max(h, w)
    if scale >= 1.0:
        return h, w                                     # already within the cap; never upscale
    return max(1, round(h * scale)), max(1, round(w * scale))


def _resize_one(job: tuple) -> tuple:
    """Resize one image. Returns (status, src_bytes, dst_bytes) with status in {done, skip, fail}."""
    src, dst, size, mode, quality = job
    try:
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            return 'skip', 0, 0
        # IMREAD_UNCHANGED keeps the alpha channel of nobg_ cutouts; cv2.resize handles 4 channels.
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if img is None:
            return 'fail', 0, 0
        h, w = img.shape[:2]
        th, tw = _fit_size(h, w, size, mode)
        if (th, tw) != (h, w):
            # INTER_AREA is the correct (area-averaging) filter for downscaling and is the filter
            # the deployment contract declares (RESIZE_FILTER in nptools/openset.py); INTER_LINEAR
            # only for the rare upscale, where AREA degenerates to nearest.
            interp = cv2.INTER_AREA if (th < h or tw < w) else cv2.INTER_LINEAR
            img = cv2.resize(img, (tw, th), interpolation=interp)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if dst.lower().endswith('.png'):
            ok = cv2.imwrite(dst, img)                   # lossless; keeps alpha
        else:
            ok = cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return 'fail', 0, 0
        return 'done', os.path.getsize(src), os.path.getsize(dst)
    except Exception:                                    # one bad file must not kill the pool
        return 'fail', 0, 0


def _plan(src_root: str, out_root: str, size: int, mode: str, quality: int) -> tuple:
    """Walk the class folders once, returning (image jobs, symlink jobs, class count)."""
    jobs, links, classes = [], [], 0
    for cls in sorted(os.listdir(src_root)):
        cls_dir = os.path.join(src_root, cls)
        if not os.path.isdir(cls_dir) or cls == 'output':
            continue                                     # 'output' is train.py's checkpoint dir
        classes += 1
        for name in sorted(os.listdir(cls_dir)):
            if not name.lower().endswith(_IMG_EXTS):
                continue
            src, dst = os.path.join(cls_dir, name), os.path.join(out_root, cls, name)
            if os.path.islink(src):
                # An oversampling duplicate: point at the RESIZED target, do not resize it again.
                links.append((os.path.realpath(src), src, dst))
            else:
                jobs.append((src, dst, size, mode, quality))
    return jobs, links, classes


def _link_all(links: list, src_root: str, out_root: str) -> tuple:
    """Recreate the oversampling symlinks in the resized tree. Returns (made, dangling)."""
    made = dangling = 0
    for real_src, link_src, dst in links:
        # Map the link's TARGET through the same src->out relocation, then link relatively so the
        # resized tree can be moved or copied without breaking.
        rel_target = os.path.relpath(real_src, os.path.abspath(src_root))
        target = os.path.join(os.path.abspath(out_root), rel_target)
        if not os.path.exists(target):
            dangling += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.islink(dst) or os.path.exists(dst):
            os.unlink(dst)
        os.symlink(os.path.relpath(target, os.path.dirname(os.path.abspath(dst))), dst)
        made += 1
    return made, dangling


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', required=True, help='source split root (class folders inside)')
    parser.add_argument('--out', required=True, help='destination root; created if absent')
    parser.add_argument('--size', type=int, default=448,
                        help='target size; 2*img_size is what build_aug_pipeline consumes (default: 448)')
    parser.add_argument('--mode', choices=('squash', 'longest'), default='squash',
                        help='squash to a square (matches --crop-mode=squash) or cap the longest side')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality for non-PNG output')
    parser.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 8),
                        help='parallel resize processes')
    parser.add_argument('--dry-run', action='store_true', help='report what would happen, write nothing')
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f'--src is not a directory: {args.src}')
    if os.path.abspath(args.src) == os.path.abspath(args.out):
        sys.exit('--src and --out must differ; this script never overwrites the originals')

    jobs, links, classes = _plan(args.src, args.out, args.size, args.mode, args.quality)
    print(f'resize_dataset: {classes} classes, {len(jobs)} images, {len(links)} '
          f'{_IMBAL_PREFIX}* symlinks -> {args.out} ({args.mode} {args.size})')
    if args.dry_run:
        return

    # cv2 is internally threaded; with a process pool that oversubscribes, so pin it per worker.
    cv2.setNumThreads(1)
    done = skipped = failed = 0
    src_bytes = dst_bytes = 0
    with Pool(args.workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        for status, sb, db in pool.imap_unordered(_resize_one, jobs, chunksize=32):
            done += status == 'done'
            skipped += status == 'skip'
            failed += status == 'fail'
            src_bytes += sb
            dst_bytes += db

    made, dangling = _link_all(links, args.src, args.out)

    print(f'  images : {done} written, {skipped} up-to-date, {failed} failed')
    print(f'  links  : {made} recreated' + (f', {dangling} skipped (target missing)' if dangling else ''))
    if done:
        print(f'  bytes  : {src_bytes / 1e6:.0f} MB -> {dst_bytes / 1e6:.0f} MB '
              f'({dst_bytes / max(src_bytes, 1) * 100:.0f}%) for the images written')
    if failed:
        sys.exit(f'{failed} image(s) failed to resize')


if __name__ == '__main__':
    main()
