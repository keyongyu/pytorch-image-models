"""Reduce class imbalance in a training set by symlink-oversampling the minority classes.

The class with the most images sets the reference; every class with fewer than HALF that many
images is topped up to that half-count by creating symlinks to its own images (cycling through
them). Symlinks (not copies) keep disk usage flat, and because train.py applies fresh random
augmentation each epoch, a symlinked duplicate behaves like a distinct sample rather than an exact
copy — so this is oversampling-with-augmentation, which balances the per-class training signal
(and, for ArcFace, the per-class prototype gradients) without memorization.

Assumes the timm/ImageFolder layout used elsewhere in this repo: ``<data_dir>/<split>/<class>/*``
(``split`` defaults to ``train``, matching ``nptools/openset.py``).

Idempotent: every symlink it creates is named with the ``_IMBAL_PREFIX`` marker, and each run first
removes any leftover marked symlinks before recounting originals — so re-running does not stack
duplicates, and the balancing is recomputed from the real images each time.

CLI:
    python -m nptools.reduce_imbalance --data-dir /path/to/dataset [--split train] [--dry-run]
    python -m nptools.reduce_imbalance --data-dir /path/to/dataset --clean   # remove the symlinks
"""
import os
import glob
import argparse

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# marker prefix for symlinks this script creates, so re-runs can find and clear its own duplicates
# without touching the original images.
_IMBAL_PREFIX = 'imbaldup_'


def _is_image(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(_IMG_EXTS)


def _clear_previous_dups(cls_dir: str) -> int:
    """Remove symlinks previously created by this script in ``cls_dir``; return the count removed."""
    removed = 0
    for name in os.listdir(cls_dir):
        if name.startswith(_IMBAL_PREFIX):
            p = os.path.join(cls_dir, name)
            if os.path.islink(p):
                os.unlink(p)
                removed += 1
    return removed


def _list_originals(cls_dir: str) -> list:
    """Real (non-symlink) image files in a class folder, sorted for deterministic cycling."""
    return sorted(
        os.path.join(cls_dir, name)
        for name in os.listdir(cls_dir)
        if not name.startswith(_IMBAL_PREFIX)
        and not os.path.islink(os.path.join(cls_dir, name))
        and _is_image(os.path.join(cls_dir, name))
    )


def clean_dups(data_dir: str, split: str = 'train', dry_run: bool = False):
    """Remove every ``_IMBAL_PREFIX`` symlink this script previously created, restoring the split
    to its original (real-image-only) state. Does not touch originals.

    Args:
        data_dir: dataset root containing the split folder.
        split: split subfolder holding per-class image folders (default ``train``).
        dry_run: report what would be removed without deleting anything.
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise SystemExit(f'{split_dir} not found (expected <data_dir>/{split}/<class>/*)')

    class_dirs = sorted(
        d for d in glob.glob(os.path.join(split_dir, '*')) if os.path.isdir(d)
    )
    if not class_dirs:
        raise SystemExit(f'No class subfolders under {split_dir}')

    print(f'Split: {split_dir}')
    total_removed = 0
    for cls_dir in class_dirs:
        cls = os.path.basename(cls_dir)
        if dry_run:
            n = sum(
                1 for name in os.listdir(cls_dir)
                if name.startswith(_IMBAL_PREFIX) and os.path.islink(os.path.join(cls_dir, name))
            )
        else:
            n = _clear_previous_dups(cls_dir)
        if n:
            print(f'{cls:20s}: {n:5d} {_IMBAL_PREFIX}* symlinks')
        total_removed += n

    action = 'would remove' if dry_run else 'removed'
    print(f'\nDone: {action} {total_removed} {_IMBAL_PREFIX}* symlinks across {len(class_dirs)} classes.')
    if dry_run:
        print('(dry-run — nothing deleted; re-run without --dry-run to apply)')


def reduce_imbalance(data_dir: str, split: str = 'train', ratio: float = 0.5, dry_run: bool = False):
    """Symlink-oversample minority classes up to ``ratio`` × the largest class's image count.

    Args:
        data_dir: dataset root containing the split folder.
        split: split subfolder holding per-class image folders (default ``train``).
        ratio: target fraction of the max class size that each class is topped up to (default 0.5,
            i.e. half of the most-populated class).
        dry_run: report the plan without creating any symlinks.
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise SystemExit(f'{split_dir} not found (expected <data_dir>/{split}/<class>/*)')

    class_dirs = sorted(
        d for d in glob.glob(os.path.join(split_dir, '*')) if os.path.isdir(d)
    )
    if not class_dirs:
        raise SystemExit(f'No class subfolders under {split_dir}')

    # First pass: clear our own previous symlinks, then count real images per class.
    counts = {}
    for cls_dir in class_dirs:
        _clear_previous_dups(cls_dir)
        counts[cls_dir] = len(_list_originals(cls_dir))

    max_count = max(counts.values())
    max_cls = os.path.basename(max(counts, key=counts.get))
    target = int(max_count * ratio)
    print(f'Split: {split_dir}')
    print(f'Most-populated class: {max_cls} ({max_count} images)')
    print(f'Target per minority class: {target} (= {ratio:g} × {max_count})\n')

    total_created = 0
    for cls_dir in class_dirs:
        cls = os.path.basename(cls_dir)
        originals = _list_originals(cls_dir)
        n = len(originals)
        if n == 0:
            print(f'{cls:20s}: WARNING — no images, skipped')
            continue
        if n >= target:
            print(f'{cls:20s}: {n:5d} imgs  (>= target, left as-is)')
            continue

        need = target - n
        created = 0
        for k in range(need):
            src = originals[k % n]                                   # cycle through the originals
            base = os.path.basename(src)
            link_name = f'{_IMBAL_PREFIX}{k}_{base}'
            link_path = os.path.join(cls_dir, link_name)
            if not dry_run:
                # relative symlink to the sibling original -> portable if the folder is moved
                os.symlink(base, link_path)
            created += 1
        total_created += created
        print(f'{cls:20s}: {n:5d} imgs  + {created:5d} symlinks -> {n + created}')

    action = 'would create' if dry_run else 'created'
    print(f'\nDone: {action} {total_created} symlinks across {len(class_dirs)} classes.')
    if dry_run:
        print('(dry-run — nothing written; re-run without --dry-run to apply)')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', required=True, type=str,
                        help='dataset root containing the split folder (<data_dir>/<split>/<class>/*)')
    parser.add_argument('--split', default='train', type=str,
                        help='split subfolder with per-class image folders (default: train)')
    parser.add_argument('--ratio', default=0.5, type=float,
                        help='target fraction of the largest class size to top minority classes up to '
                             '(default: 0.5 = half of the most-populated class)')
    parser.add_argument('--dry-run', action='store_true',
                        help='report the plan without creating symlinks')
    parser.add_argument('--clean', action='store_true',
                        help=f'remove all {_IMBAL_PREFIX}* symlinks created by this script and exit '
                             '(no oversampling)')
    args = parser.parse_args()
    if args.clean:
        clean_dups(args.data_dir, split=args.split, dry_run=args.dry_run)
    else:
        reduce_imbalance(args.data_dir, split=args.split, ratio=args.ratio, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
