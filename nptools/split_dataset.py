"""Split a flat class-folder dataset into train/val subsets.

Usage:
    python pepsico/split_dataset.py \\
        --src  pepsico/dataset/all \\
        --out  pepsico/split \\
        --val-ratio 0.1 \\
        --seed 42

Output layout:
    <out>/
        train/<class_id>/*.jpg
        val/<class_id>/*.jpg
        class_map.txt          # copied from src parent if present
"""
import argparse
import random
import shutil
from pathlib import Path


def split(src: Path, out: Path, val_ratio: float, seed: int) -> None:
    class_dirs = sorted(p for p in src.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {src}")

    train_root = out / "train"
    val_root   = out / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    total_train = total_val = 0

    for cls_dir in class_dirs:
        images = sorted(cls_dir.glob("*.jpg"))
        if not images:
            continue

        rng.shuffle(images)
        n_val = max(1, round(len(images) * val_ratio))
        val_imgs   = images[:n_val]
        train_imgs = images[n_val:]

        for split_root, imgs in [(train_root, train_imgs), (val_root, val_imgs)]:
            dst_dir = split_root / cls_dir.name
            dst_dir.mkdir(exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dst_dir / img.name)

        total_train += len(train_imgs)
        total_val   += len(val_imgs)
        print(f"  {cls_dir.name:40s}  train={len(train_imgs):4d}  val={len(val_imgs):3d}")

    # Copy class_map.txt from src parent if present
    src_map = src.parent / "class_map.txt"
    if not src_map.exists():
        src_map = src / "class_map.txt"
    if src_map.exists():
        shutil.copy2(src_map, out / "class_map.txt")

    print(f"\nTotal  train={total_train}  val={total_val}  "
          f"({total_val / (total_train + total_val):.1%} val)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src",       type=Path,  default=Path("pepsico/dataset/all"))
    parser.add_argument("--out",       type=Path,  default=Path("pepsico/split"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    split(args.src, args.out, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
