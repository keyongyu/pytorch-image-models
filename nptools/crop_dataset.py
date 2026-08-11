"""Crop bounding boxes from shelf photos into a classification dataset.

Usage:
    python nptools/crop_dataset.py \\
        --src  pepsico/Batch1-3_merge_pepsico \\
        --out  pepsico/dataset

Output layout:
    <out>/
        all/<class_id>/<photo_stem>_<box_idx>.jpg
        class_map.txt           # one class_id per line, line = index
"""
import argparse
import json
import sys
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Class map
# ---------------------------------------------------------------------------

def build_class_map(templates_path: Path) -> tuple[dict[str, str], dict[str, dict]]:
    """Return (variant_to_main, class_map).

    variant_to_main : annotation id → canonical main class id
    class_map       : main class id → {idx, name}
    """
    data = json.loads(templates_path.read_text(encoding="utf-8"))
    skus = data["categories"][0]["skus"]

    # Pass 1: collect exportable main classes in seq order
    # Exclude only no_export entries; sample_img may be empty (e.g. Other Beer)
    mains: list[dict] = []
    for s in skus:
        if s.get("is_main") and not s.get("no_export"):
            mains.append(s)
    mains.sort(key=lambda s: s["seq"])

    class_map: dict[str, dict] = {}
    for idx, s in enumerate(mains):
        class_map[s["id"]] = {"idx": idx, "name": s["name"]}

    # Pass 2: build variant → main lookup
    variant_to_main: dict[str, str] = {}
    for s in skus:
        sid = s["id"]
        if s.get("is_main"):
            variant_to_main[sid] = sid
        elif "main_id" in s:
            variant_to_main[sid] = s["main_id"]

    return variant_to_main, class_map


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def crop_datasets(src: Path, out: Path) -> None:
    templates_path = src / "templates.json"
    photos_dir     = src / "photos"
    ann_dir        = photos_dir / "Annotations"

    variant_to_main, class_map = build_class_map(templates_path)

    all_dir = out / "all"
    all_dir.mkdir(parents=True, exist_ok=True)

    # Save class map: one class_id per line, line number = class index
    # timm's load_class_map(.txt) uses enumerate(f), so the line IS the key
    lines = sorted(class_map.items(), key=lambda kv: kv[1]["idx"])
    (out / "class_map.txt").write_text(
        "\n".join(k for k, _ in lines) + "\n",
        encoding="utf-8",
    )
    print(f"Class map: {len(class_map)} exportable classes")

    ann_files = sorted(ann_dir.glob("*.json"))
    print(f"Annotation files: {len(ann_files)}")

    total_saved = 0
    total_skipped_ignore = 0
    total_skipped_unknown = 0

    for ann_path in ann_files:
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        img_name = ann["filename"]
        img_path = photos_dir / img_name

        if not img_path.exists():
            print(f"  [WARN] image not found: {img_path}", file=sys.stderr)
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] failed to read: {img_path}", file=sys.stderr)
            continue

        h_img, w_img = img.shape[:2]
        stem = Path(img_name).stem

        for box_idx, box in enumerate(ann.get("bndboxes", [])):
            if box.get("ignore"):
                total_skipped_ignore += 1
                continue

            ann_id = box["id"]
            main_id = variant_to_main.get(ann_id)
            if main_id is None or main_id not in class_map:
                total_skipped_unknown += 1
                if main_id is None:
                    print(f"  [WARN] unknown id '{ann_id}' in {ann_path.name}", file=sys.stderr)
                continue

            # Clamp box to image bounds
            x1 = max(0, int(box["x"]))
            y1 = max(0, int(box["y"]))
            x2 = min(w_img, int(box["x"] + box["w"]))
            y2 = min(h_img, int(box["y"] + box["h"]))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            class_dir = all_dir / main_id
            class_dir.mkdir(exist_ok=True)

            out_path = class_dir / f"{stem}_{box_idx:04d}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            total_saved += 1

    print(f"Saved  : {total_saved} crops")
    print(f"Ignored: {total_skipped_ignore} (ignore=true)")
    print(f"Skipped: {total_skipped_unknown} (unknown/no_export class)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("pepsico/Batch1-3_merge_pepsico"),
        help="Dataset root (contains templates.json and photos/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pepsico/dataset"),
        help="Output directory (will contain all/, class_map.txt)",
    )
    args = parser.parse_args()
    crop_datasets(args.src, args.out)


if __name__ == "__main__":
    main()
