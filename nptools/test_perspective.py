"""Test perspective transform using albumentations: bottom-bigger vs top-bigger effect.

BottomBiggerPerspective / TopBiggerPerspective subclass A.Perspective and override
get_params_dependent_on_data so the corner squeeze is directional instead of random.

Usage:
    uv run python nptools/test_perspective.py <src.png> [dest_folder]
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

import albumentations as A
from nptools.npaug import BottomBiggerPerspective, TopBiggerPerspective

# (label, direction, scale)
CASES = [
    ("orig",          None,       0.00),
    ("bot+0.05",      "bottom",   0.05),
    ("bot+0.10",      "bottom",   0.10),
    ("bot+0.15",      "bottom",   0.15),
    ("bot+0.20",      "bottom",   0.20),
    ("top+0.05",      "top",      0.05),
    ("top+0.10",      "top",      0.10),
    ("top+0.15",      "top",      0.15),
    ("top+0.20",      "top",      0.20),
]



def make_transform(direction, scale):
    if direction == "bottom":
        return BottomBiggerPerspective(scale=(scale, scale), p=1.0)
    if direction == "top":
        return TopBiggerPerspective(scale=(scale, scale), p=1.0)
    if direction == "symmetric":
        return A.Perspective(scale=(scale, scale), p=1.0)
    return None


def _rgba_to_tile_bgr(rgba: np.ndarray, bg: int = 20) -> np.ndarray:
    """Composite RGBA onto a solid background for grid display (BGR output)."""
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    bgr = rgb[:, :, ::-1] * alpha + bg * (1.0 - alpha)
    return bgr.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", nargs="?", default="nptools/scancode.png",
                        help="input PNG file (default: nptools/scancode.png)")
    parser.add_argument("dest", nargs="?", default="nptools/perspective",
                        help="output folder (default: nptools/perspective)")
    args = parser.parse_args()

    src = Path(args.src)
    out_dir = Path(args.dest)

    from PIL import Image as PILImage
    pil = PILImage.open(src)
    is_rgba = pil.mode == "RGBA"
    # keep native mode: RGBA stays RGBA, anything else → RGB
    img = np.array(pil if is_rgba else pil.convert("RGB"))
    ext = "png" if is_rgba else "jpg"

    out_dir.mkdir(parents=True, exist_ok=True)

    LABEL_H = 32
    font = cv2.FONT_HERSHEY_SIMPLEX
    tiles = []

    for label, direction, scale in CASES:
        tf = make_transform(direction, scale)
        out = img.copy() if tf is None else tf(image=img)["image"]

        out_path = out_dir / f"{label}.{ext}"
        if is_rgba:
            PILImage.fromarray(out, mode="RGBA").save(str(out_path))
        else:
            cv2.imwrite(str(out_path), out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])

        h, w = out.shape[:2]
        tile_bgr = _rgba_to_tile_bgr(out) if is_rgba else out[:, :, ::-1]
        bar = np.full((LABEL_H, w, 3), 40, dtype=np.uint8)
        cv2.putText(bar, label, (4, LABEL_H - 8), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, tile_bgr]))

    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    blank = np.full((th, tw, 3), 20, dtype=np.uint8)
    grid_rows = []
    for r in range(rows):
        row = tiles[r * cols: r * cols + cols]
        while len(row) < cols:
            row.append(blank)
        grid_rows.append(np.hstack(row))
    grid = np.vstack(grid_rows)

    grid_path = out_dir / "grid.jpg"
    cv2.imwrite(str(grid_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved grid  -> {grid_path.resolve()}")
    print(f"Individual  -> {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
