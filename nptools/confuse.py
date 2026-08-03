"""Build a confusion dump for an ArcFace-trained posm checkpoint.

Reproduces ``train.py``'s ArcFace eval scoring (no-margin cosine logits against the saved head
weight) over an image-folder split, then writes both:

  * ``<out-dir>/confuse.txt`` — every misclassification as a TSV with header
    ``src_file<TAB>src_class<TAB>pred_class`` (``src_file`` is the class-relative path, e.g.
    ``posm_2/foo.jpeg``), sorted by true class then predicted class.
  * ``<out-dir>/<class>/*`` — the real images of every class involved in a confusion (union of the
    true and predicted classes), so the confused clusters can be eyeballed side by side.

``imbaldup_*`` imbalance-duplicate symlinks are skipped everywhere, so only real files are listed
and copied.

Since ArcFace leaves the backbone's own classifier untrained, scoring must use the margin head
saved under ``task_state['arcface']`` — a plain ``model(input)`` (as in validate.py) would be
meaningless. See ``nptools/arcfaceloss.py``.

Usage:
    uv run python nptools/confuse.py \
        --checkpoint posmlv/output/<run>/model_best.pth.tar \
        --data-dir ./posmlv --split train --out-dir nptools/confuse --device cuda
"""
import argparse
import os
import shutil
import sys

import torch

# Allow `from nptools...` / `import timm` regardless of the cwd the script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timm.models import create_model
from timm.data import resolve_data_config, create_dataset, create_transform

from nptools.arcfaceloss import ArcFaceLoss


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, help='ArcFace checkpoint (has task_state[arcface])')
    p.add_argument('--data-dir', default='./posmlv', help='dataset root')
    p.add_argument('--split', default='train', help='image-folder subdir to evaluate')
    p.add_argument('--out-dir', default='nptools/confuse', help='where confuse.txt and class dirs are written')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--no-copy', action='store_true', help='write confuse.txt only, skip copying images')
    return p.parse_args()


def load_model_and_head(ckpt, device):
    """Rebuild the backbone + ArcFace head from a checkpoint, matching train.py's eval setup."""
    saved = vars(ckpt['args']) if not isinstance(ckpt['args'], dict) else ckpt['args']
    num_classes = saved['num_classes']

    model = create_model(saved['model'], num_classes=num_classes)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).eval()

    head = ArcFaceLoss(
        in_features=model.num_features,
        num_classes=num_classes,
        s=saved['arcface_s'],
        m=saved['arcface_m'],
    )
    head.load_state_dict(ckpt['task_state']['arcface'])
    head = head.to(device).eval()
    return model, head, saved


def build_loader_and_paths(model, saved, data_dir, split, batch_size, workers, pin):
    """Eval dataset/transform identical to train.py, plus the per-index absolute path list."""
    data_config = resolve_data_config(
        {'crop_pct': saved.get('crop_pct'), 'crop_mode': saved.get('crop_mode')},
        model=model,
        verbose=False,
    )
    transform = create_transform(
        input_size=data_config['input_size'],
        is_training=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        crop_pct=data_config['crop_pct'],
        crop_mode=data_config['crop_mode'],
        nobg=saved.get('nobg', False),
    )
    dataset = create_dataset(
        '',
        root=data_dir,
        split=split,
        class_map=saved['class_map'],
        input_img_mode=None,  # native mode so the nobg dispatch keeps alpha + filename
        transform=transform,
    )
    idx_to_class = {v: k for k, v in dataset.reader.class_to_idx.items()}
    paths = [dataset.filename(i, absolute=True) for i in range(len(dataset))]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # keep order aligned with `paths`
        num_workers=workers,
        pin_memory=pin,
    )
    return loader, paths, idx_to_class


def collect_misclassifications(model, head, loader, paths, idx_to_class, device):
    """Return (rows, skipped) where rows = [(rel_src, src_class, pred_class), ...] for real files."""
    rows = []
    skipped_symlink = 0
    gidx = 0
    with torch.inference_mode():
        for input, target in loader:
            input = input.to(device)
            feat = model.forward_head(model.forward_features(input), pre_logits=True)
            pred = head.logits(feat).argmax(dim=1).cpu()
            for p, t in zip(pred.tolist(), target.tolist()):
                src = paths[gidx]
                gidx += 1
                if p == t:
                    continue
                if os.path.islink(src):  # skip imbaldup_* duplicate symlinks
                    skipped_symlink += 1
                    continue
                src_class = idx_to_class[t]
                # class-relative path, e.g. posm_2/foo.jpeg
                rel = os.path.join(src_class, os.path.basename(src))
                rows.append((rel, src_class, idx_to_class[p]))
    rows.sort(key=lambda r: (r[1], r[2], r[0]))
    return rows, skipped_symlink


def write_confuse_txt(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("src_file\tsrc_class\tpred_class\n")
        for rel, sc, pc in rows:
            f.write(f"{rel}\t{sc}\t{pc}\n")


def copy_misclassified(rows, split_root, out_dir):
    """Copy only the misclassified images into out_dir/<src_class>/, keeping the true-class folder."""
    per_class = {}
    total = 0
    for rel, src_class, _pred in rows:
        src = os.path.join(split_root, rel)  # rel == <src_class>/<basename>
        dst_dir = os.path.join(out_dir, src_class)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(rel)))
        per_class[src_class] = per_class.get(src_class, 0) + 1
        total += 1
    for c in sorted(per_class):
        print(f"  {c:<12s} -> {per_class[c]:4d} wrong-predict files")
    print(f"copied {total} misclassified images across {len(per_class)} classes into {out_dir}/")


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model, head, saved = load_model_and_head(ckpt, device)
    print(f"checkpoint epoch={ckpt.get('epoch')} arch={ckpt.get('arch')} num_classes={saved['num_classes']}")

    loader, paths, idx_to_class = build_loader_and_paths(
        model, saved, args.data_dir, args.split, args.batch_size, args.workers, device.type == 'cuda',
    )
    rows, skipped = collect_misclassifications(model, head, loader, paths, idx_to_class, device)

    out_txt = os.path.join(args.out_dir, 'confuse.txt')
    write_confuse_txt(rows, out_txt)
    print(f"wrote {len(rows)} misclassified rows to {out_txt} (skipped {skipped} symlink dups)")

    if not args.no_copy:
        # resolve the actual eval root (handles split-synonym fallback) for the copy source
        from timm.data.dataset_factory import _search_split
        split_root = _search_split(args.data_dir, args.split)
        copy_misclassified(rows, split_root, args.out_dir)


if __name__ == '__main__':
    main()
