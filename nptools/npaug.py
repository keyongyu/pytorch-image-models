"""
Shared augmentation pipeline.
Imported by both augment_v2.py (offline) and dataset.py (online training).
"""

import glob
import os
import random
from typing import Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# This module is imported inside every DataLoader worker. By default OpenCV spins up a
# thread pool sized to the CPU count (32 here), so N workers => N*32 threads fighting for
# the cores and adding scheduling overhead with no real speedup on small (224px) images.
# Pin OpenCV to a single thread per worker so parallelism comes from `--workers` instead.
cv2.setNumThreads(1)

try:
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
except Exception:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# ── Global augmentation switch for the nobg train transforms ─────────────────────
#   True  -> build_aug_pipeline        (heavy geometric + photometric; DISTORTS shape/aspect)
#   False -> build_photometric_pipeline (photometric only; shape/aspect PRESERVED)
# Edit this default, or override via env: NOBG_FULL_AUG=1 (on) / 0 (off).
USE_FULL_AUG = os.environ.get('NOBG_FULL_AUG', '1') not in ('0', 'false', 'False', 'no', 'NO')

# ── Eval decode: let the JPEG decoder pre-shrink (see _draft_decode) ─────────────
# On by default -- the eval transform resizes to (h, w) immediately, so decoding the full 10 MP
# original only to throw it away is what left the GPU idle during validate(). Set
# NOBG_EVAL_DRAFT=0 to restore the full-resolution decode, which keeps eval preprocessing
# bit-identical to nptools/openset.py's CPU verification path (its ncnn deployment contract).
USE_EVAL_DRAFT = os.environ.get('NOBG_EVAL_DRAFT', '1') not in ('0', 'false', 'False', 'no', 'NO')

# ── Debug: dump post-augmentation images (what the model actually receives) ──────
# Enable by setting AUG_DUMP_DIR (e.g. AUG_DUMP_DIR=kytest/after_aug). Saves up to
# AUG_DUMP_MAX images per worker process as auto-incrementing PNGs. Disabled when unset.
_AUG_DUMP_DIR = os.environ.get('AUG_DUMP_DIR', '') or None
_AUG_DUMP_MAX = int(os.environ.get('AUG_DUMP_MAX', '200'))
_aug_dump_count = 0


def set_aug_dump_dir(path, max_per_worker=None):
    """Enable (or disable) dumping post-augmentation images for visual npaug testing.

    When `path` is set, each augmented image is saved under ``<path>/<class>/`` (class taken from
    the sample's filename parent folder). Call this in the MAIN process before iterating the loader
    so forked DataLoader workers inherit the setting. `max_per_worker` caps saves per worker.
    """
    global _AUG_DUMP_DIR, _AUG_DUMP_MAX, _aug_dump_count
    _AUG_DUMP_DIR = path or None
    if max_per_worker is not None:
        _AUG_DUMP_MAX = int(max_per_worker)
    _aug_dump_count = 0


def _maybe_dump(np_rgb, subdir=None):
    """If dumping is enabled, save the augmented HWC uint8 RGB image (grouped into `subdir`)."""
    global _aug_dump_count
    if _AUG_DUMP_DIR is None or _aug_dump_count >= _AUG_DUMP_MAX:
        return
    out_dir = _AUG_DUMP_DIR if not subdir else os.path.join(_AUG_DUMP_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    idx = _aug_dump_count
    _aug_dump_count += 1
    # pid prefix avoids collisions across DataLoader worker processes
    PILImage.fromarray(np_rgb).save(os.path.join(out_dir, f'{os.getpid()}_{idx:06d}.png'))


def _class_of(pil_img):
    """Class name for dump grouping = parent folder of the sample's filename (or 'unknown')."""
    fn = getattr(pil_img, 'filename', '') or ''
    return os.path.basename(os.path.dirname(fn)) or 'unknown'


class RandomSpotlight(A.ImageOnlyTransform):
    """Directional overexposure: a bright gradient beam from a random edge."""

    def __init__(self, intensity_range=(1.5, 3.0), p=0.3):
        super().__init__(p=p)
        self.intensity_range = intensity_range

    def apply(self, img, intensity, direction, **params):
        h, w = img.shape[:2]
        if direction == 0:
            grad = np.linspace(intensity, 1.0, w, dtype=np.float32)[None, :, None]
        elif direction == 1:
            grad = np.linspace(1.0, intensity, w, dtype=np.float32)[None, :, None]
        elif direction == 2:
            grad = np.linspace(intensity, 1.0, h, dtype=np.float32)[:, None, None]
        else:
            grad = np.linspace(1.0, intensity, h, dtype=np.float32)[:, None, None]
        return np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)

    def get_params(self):
        return {
            "intensity": random.uniform(*self.intensity_range),
            "direction": random.randint(0, 3),
        }

    def get_transform_init_args_names(self):
        return ("intensity_range",)


class AtmosphericFog(A.ImageOnlyTransform):
    """Fast, vectorized fog using the atmospheric scattering (Koschmieder) model.

    Drop-in replacement for A.RandomFog, which simulates individual fog particles in a
    Python loop (~375 ms/img); this is a single elementwise blend (~1-3 ms/img):

        out = img * t + A * (1 - t)

    where A is the atmospheric light (fog color) and t is a spatially-varying transmission
    map. t is built from a smooth low-frequency field (tiny random grid upsampled) times a
    vertical gradient (denser toward the top), so the fog reads as depth rather than a flat
    haze wash.
    """

    def __init__(self, fog_coef_range=(0.1, 0.4), light_range=(190, 255), p=0.5):
        super().__init__(p=p)
        self.fog_coef_range = fog_coef_range
        self.light_range = light_range

    def apply(self, img, coef, light, field, **params):
        h, w = img.shape[:2]
        # smooth low-frequency transmission field upsampled from the tiny grid
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
        vgrad = np.linspace(1.0, 0.6, h, dtype=np.float32)[:, None]  # denser toward the top
        t = (1.0 - coef * field * vgrad)[..., None]                  # transmission in (0, 1]
        return (img.astype(np.float32) * t + light * (1.0 - t)).astype(np.uint8)

    def get_params(self):
        return {
            "coef": random.uniform(*self.fog_coef_range),
            "light": random.randint(*self.light_range),
            "field": np.random.rand(8, 8).astype(np.float32),
        }

    def get_transform_init_args_names(self):
        return ("fog_coef_range", "light_range")


class BottomBiggerPerspective(A.Perspective):
    """Squeeze the top edge inward so the bottom appears wider (camera-above look).

    Maps full image corners → trapezoid with narrowed top; warpPerspective fills
    the vacated top corners with the border mode instead of zooming in.
    """

    def get_params_dependent_on_data(self, params, data):
        h, w = params["shape"][:2]
        scale = self.py_random.uniform(*self.scale)
        dx = int(w * scale)
        src = np.float32([[0, 0],  [w-1, 0],    [w-1, h-1], [0, h-1]])
        dst = np.float32([[dx, 0], [w-1-dx, 0], [w-1, h-1], [0, h-1]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        return {"matrix": matrix, "max_height": h, "max_width": w, "matrix_bbox": matrix}


class TopBiggerPerspective(A.Perspective):
    """Squeeze the bottom edge inward so the top appears wider (camera-below look)."""

    def get_params_dependent_on_data(self, params, data):
        h, w = params["shape"][:2]
        scale = self.py_random.uniform(*self.scale)
        dx = int(w * scale)
        src = np.float32([[0, 0], [w-1, 0], [w-1, h-1],    [0, h-1]])
        dst = np.float32([[0, 0], [w-1, 0], [w-1-dx, h-1], [dx, h-1]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        return {"matrix": matrix, "max_height": h, "max_width": w, "matrix_bbox": matrix}


def make_bottom_top_perspective_transform(scale=(0.02, 0.15)):
    """Random directional perspective: 25% bottom-bigger, 25% top-bigger, 50% unchanged."""
    return A.OneOf([
        BottomBiggerPerspective(scale=scale, p=1.0),
        TopBiggerPerspective(scale=scale, p=1.0),
    ], p=0.5)


# ── Pipeline builder ──────────────────────────────────────────────────────────


def build_aug_pipeline(img_size=224):
    transforms = [
        # — Geometric: pick ONE distortion type (mild; strong warps fold/tear the object) —
        A.OneOf(
            [
                A.Perspective(scale=(0.02, 0.06)),
                A.ElasticTransform(alpha=20, sigma=15),
                A.GridDistortion(num_steps=5, distort_limit=0.15),
                A.OpticalDistortion(distort_limit=0.15),
            ],
            p=0.3,
        ),
        # — Spatial —
        # A.HorizontalFlip(p=0.5),
        A.OneOf(
            [
                A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101),
                A.Affine(
                    scale=(0.75, 1.25),
                    translate_percent={"x": (-0.12, 0.12), "y": (-0.12, 0.12)},
                    rotate=(-15, 15),
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
            ],
            p=0.6,
        ),
        # — (removed) standalone heavy pre-crop Defocus(10-15): it stacked an extra strong blur
        #   outside the degradation budget below, destroying the object. Blur is handled there now.
        # — Normalize to square so crop ratio math works for any aspect ratio —
        A.Resize(height=img_size * 2, width=img_size * 2),
        # — Crop —
        A.RandomResizedCrop(
            size=(img_size, img_size), scale=(0.9, 1.0), ratio=(0.9, 1.11), p=0.5
        ),
        # — Lighting: all effects compete in one OneOf —
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.5, 0.3), contrast_limit=(-0.3, 0.4)
                ),
                A.HueSaturationValue(
                    #hue_shift_limit=25, sat_shift_limit=50, val_shift_limit=30
                    hue_shift_limit=2, sat_shift_limit=20, val_shift_limit=30
                ),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25),
                A.RandomGamma(gamma_limit=(80, 150)),
                A.RandomToneCurve(scale=0.3),
                A.CLAHE(clip_limit=4.0),
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_limit=(1, 2),
                    shadow_intensity_range=(0.3, 0.5),
                ),
                RandomSpotlight(intensity_range=(1.5, 3.0)),
            ],
            p=0.7,
        ),
        # — Noise: pick ONE —
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.02, 0.15)),
                A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.1, 0.6)),
                A.MultiplicativeNoise(multiplier=(0.7, 1.3)),
            ],
            p=0.4,
        ),
        # — Degradation group: blur / weather / downscale / occlusion —
        # SomeOf(n=2) applies at most TWO of these four families (never 3+ together), so an image
        # can't be simultaneously blurred + rained-on + downscaled + occluded into noise.
        A.SomeOf(
            [
                # blur family (pick one)
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=(3, 9)),
                        A.GaussianBlur(blur_limit=(3, 9)),
                        A.Defocus(radius=(2, 7)),
                        A.ZoomBlur(max_factor=1.15),
                    ],
                    p=1.0,
                ),
                # weather family (pick one)
                A.OneOf(
                    [
                        # A.RandomFog(fog_coef_range=(0.1, 0.4), alpha_coef=0.1),  # ~375ms/img -> replaced
                        AtmosphericFog(fog_coef_range=(0.3, 0.7)),
                        A.RandomRain(
                            slant_range=(-15, 15),
                            drop_length=10,
                            drop_width=1,
                            drop_color=(200, 200, 200),
                            blur_value=3,
                            brightness_coefficient=0.85,
                            rain_type="default",
                        ),
                    ],
                    p=1.0,
                ),
                # downscale
                A.Downscale(
                    scale_range=(0.50, 0.90),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                ),
                # occlusion
                A.CoarseDropout(
                    num_holes_range=(2, 6),
                    hole_height_range=(0.01, 0.1),
                    hole_width_range=(0.01, 0.1),
                    p=1.0,
                ),
            ],
            n=2,
            replace=False,
            p=0.5,
        ),
        # — Compression artefacts —
        A.ImageCompression(quality_range=(10, 95), p=0.3),
        # — Occasional color-space shift: pick ONE —
        A.OneOf(
            [
                # A.ToGray(),
                # A.Solarize(),   # inverts bright pixels -> false/psychedelic colors (destroys label color)
                # A.Equalize(),   # per-channel histogram eq -> color shifts (destroys label color)
                A.Posterize(num_bits=4),
            ],
            p=0.1,
        ),
        # — Final resize to model input size —
        A.Resize(height=img_size, width=img_size),
    ]

    #if normalize:
    #    transforms.append(
    #        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #    )

    return A.Compose(transforms)


def build_shape_preserving_pipeline(img_size=224, fill=(255, 255, 255)):
    """Augment appearance while preserving the object's silhouette/proportions.

    For tasks where the class cue is the SHAPE (e.g. bottle shoulder steepness / body
    straightness across 500ml/1L/1.5L), geometry-distorting augmentations (perspective,
    elastic, grid/optical distortion, aspect-changing crops, scale/large rotation) are harmful
    because they warp the very feature that separates the classes. This pipeline:
      - resizes with aspect ratio preserved (letterbox / pad), never squashing;
      - applies only mild rotation/translation (no scale change, no perspective/elastic);
      - applies heavy photometric aug (lighting/noise/blur/weather) to bridge domain gaps.
    """
    return A.Compose([
        # aspect-preserving resize to a square canvas (letterbox) -> keeps true proportions
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, fill=fill),
        # mild geometry only: small rotate + translate, uniform scale (keep_ratio) -> shape intact
        A.Affine(
            rotate=(-8, 8),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.92, 1.08),
            keep_ratio=True,
            border_mode=cv2.BORDER_CONSTANT,
            fill=fill,
            p=0.5,
        ),
        # — Lighting (bridge the bright-train -> dark-val gap); does not change shape —
        A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.3), contrast_limit=(-0.3, 0.4)),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30),
                A.RandomGamma(gamma_limit=(80, 150)),
                A.RandomToneCurve(scale=0.3),
                A.CLAHE(clip_limit=4.0),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2),
                               shadow_intensity_range=(0.3, 0.5)),
                RandomSpotlight(intensity_range=(1.5, 3.0)),
            ],
            p=0.8,
        ),
        # — Noise —
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.02, 0.15)),
                A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.1, 0.6)),
                A.MultiplicativeNoise(multiplier=(0.7, 1.3)),
            ],
            p=0.4,
        ),
        # — Blur / degradation —
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 11)),
                A.GaussianBlur(blur_limit=(3, 9)),
                A.Defocus(radius=(2, 6)),
                A.Downscale(scale_range=(0.25, 0.5),
                            interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}),
            ],
            p=0.4,
        ),
        # — Weather —
        A.OneOf(
            [
                # A.RandomFog(fog_coef_range=(0.1, 0.4), alpha_coef=0.1),  # ~375ms/img -> replaced
                AtmosphericFog(fog_coef_range=(0.3, 0.7)),
                A.RandomRain(slant_range=(-15, 15), drop_length=10, drop_width=1,
                             drop_color=(200, 200, 200), blur_value=3,
                             brightness_coefficient=0.85, rain_type="default"),
            ],
            p=0.2,
        ),
        # — Compression artefacts (no shape change) —
        A.ImageCompression(quality_range=(15, 95), p=0.3),
    ])


def build_photometric_pipeline():
    """Appearance-only aug (no geometry, no resize) — for use after manual letterbox/compositing."""
    return A.Compose([
        A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.3), contrast_limit=(-0.3, 0.4)),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30),
                A.RandomGamma(gamma_limit=(80, 150)),
                A.RandomToneCurve(scale=0.3),
                A.CLAHE(clip_limit=4.0),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2),
                               shadow_intensity_range=(0.3, 0.5)),
                RandomSpotlight(intensity_range=(1.5, 3.0)),
            ],
            p=0.8,
        ),
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.02, 0.15)),
                A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.1, 0.6)),
                A.MultiplicativeNoise(multiplier=(0.7, 1.3)),
            ],
            p=0.4,
        ),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 11)),
                A.GaussianBlur(blur_limit=(3, 9)),
                A.Defocus(radius=(2, 6)),
            ],
            p=0.4,
        ),
        A.OneOf(
            [
                # A.RandomFog(fog_coef_range=(0.1, 0.4), alpha_coef=0.1),  # ~375ms/img -> replaced
                AtmosphericFog(fog_coef_range=(0.3, 0.7)),
                A.RandomRain(slant_range=(-15, 15), drop_length=10, drop_width=1,
                             drop_color=(200, 200, 200), blur_value=3,
                             brightness_coefficient=0.85, rain_type="default"),
            ],
            p=0.2,
        ),
        A.ImageCompression(quality_range=(15, 95), p=0.3),
    ])


def _letterbox_rgba(rgba, h, w):
    """Resize an RGBA image to fit an h x w box, preserving aspect, transparent padding."""
    im = PILImage.fromarray(rgba, mode='RGBA')
    im.thumbnail((w, h), PILImage.BILINEAR)  # PIL size is (width, height)
    canvas = PILImage.new('RGBA', (w, h), (0, 0, 0, 0))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return np.array(canvas)


def _letterbox_rgb(rgb, h, w, fill):
    """Resize an RGB image to fit an h x w box, preserving aspect, padded with `fill`."""
    im = PILImage.fromarray(rgb, mode='RGB')
    im.thumbnail((w, h), PILImage.BILINEAR)
    canvas = PILImage.new('RGB', (w, h), tuple(fill))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return np.array(canvas)


def _bg_rect_rgb(paths, h, w, fill):
    """A random background center-cropped/resized to an h x w RGB rectangle (or solid fill)."""
    if paths:
        for _ in range(10):
            try:
                bg = PILImage.open(random.choice(paths)).convert('RGB')
            except (IOError, OSError):
                continue
            bw, bh = bg.size
            # center-crop to the target aspect ratio, then resize to (w, h)
            target_ar = w / h
            src_ar = bw / bh
            if src_ar > target_ar:  # too wide -> crop width
                cw = int(round(bh * target_ar))
                x0 = (bw - cw) // 2
                bg = bg.crop((x0, 0, x0 + cw, bh))
            else:  # too tall -> crop height
                ch = int(round(bw / target_ar))
                y0 = (bh - ch) // 2
                bg = bg.crop((0, y0, bw, y0 + ch))
            return np.array(bg.resize((w, h), PILImage.BILINEAR))
    return np.full((h, w, 3), fill, dtype=np.uint8)


#aug = build_aug_pipeline()

def _random_bg_crop(bg_paths, target_h, target_w):
    BG_MIN_BRIGHTNESS = 60 
    for _ in range(30):
        bg = cv2.imread(random.choice(bg_paths))
        if bg is None:
            continue
        if cv2.mean(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY))[0] < BG_MIN_BRIGHTNESS:
            continue
        bh, bw = bg.shape[:2]
        if bh < target_h or bw < target_w:
            scale = max(target_h / bh, target_w / bw) * 1.05
            bg = cv2.resize(bg, (int(bw * scale), int(bh * scale)),
                            interpolation=cv2.INTER_LANCZOS4)
            bh, bw = bg.shape[:2]
        y = random.randint(0, bh - target_h)
        x = random.randint(0, bw - target_w)
        return bg[y : y + target_h, x : x + target_w]
    return None


def _collect_bg_paths(src):
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(src, "**", e), recursive=True)
    return paths


# ── timm-compatible transform builders for background-removed (RGBA PNG) inputs ──
#
# transform_nobg_train / transform_nobg_eval mirror the signatures of
# timm.data.transforms_factory.transforms_imagenet_train / transforms_imagenet_eval,
# so they can be used as drop-in transform builders. They expect the input image to be
# loaded in RGBA mode (a background-removed PNG): the alpha channel is the object mask.
#   - train: composite the cutout onto a random background (bg-swap) + heavy albumentations aug
#   - eval:  composite the cutout onto a solid background, resize, normalize
#
# NOTE: most imagenet-style kwargs (auto_augment, re_prob, hflip, ...) are accepted for API
# compatibility but ignored — the albumentations pipeline defines the actual augmentation.
# The dataset must load images in RGBA mode (e.g. input_img_mode='RGBA'); otherwise the loader
# strips alpha to RGB and the bg-swap has nothing to composite.

# Background images for bg-swap during training. Set via env NOBG_BG_DIR or set_nobg_bg_dir().
_NOBG_BG_PATHS = None
_env_bg = os.environ.get('NOBG_BG_DIR')
if _env_bg and os.path.isdir(_env_bg):
    _NOBG_BG_PATHS = _collect_bg_paths(_env_bg)
# print(f'>>>>>>NOBG_BG_DIR<<<<<<<: {_NOBG_BG_PATHS}');

def set_nobg_bg_dir(bg_dir):
    """Configure the background-image directory used for nobg training bg-swap."""
    global _NOBG_BG_PATHS
    _NOBG_BG_PATHS = _collect_bg_paths(bg_dir) if bg_dir else None


def _as_size(img_size):
    if isinstance(img_size, (tuple, list)):
        return int(img_size[0]), int(img_size[1])
    return int(img_size), int(img_size)


def _random_bg_crop_rgb(bg_paths, target_h, target_w):
    """Pick a random background (loaded via PIL as RGB) and crop a target_h x target_w region."""
    BG_MIN_BRIGHTNESS = 60
    for _ in range(30):
        try:
            bg = np.array(PILImage.open(random.choice(bg_paths)).convert('RGB'))
        except (IOError, OSError):
            continue
        if bg.mean() < BG_MIN_BRIGHTNESS:
            continue
        bh, bw = bg.shape[:2]
        if bh < target_h or bw < target_w:
            scale = max(target_h / bh, target_w / bw) * 1.05
            bg = np.array(PILImage.fromarray(bg).resize(
                (int(bw * scale), int(bh * scale)), PILImage.LANCZOS))
            bh, bw = bg.shape[:2]
        y = random.randint(0, bh - target_h)
        x = random.randint(0, bw - target_w)
        return bg[y:y + target_h, x:x + target_w]
    return None


def _composite_rgba(fg_rgba, bg_rgb):
    """Alpha-composite an RGBA foreground onto an RGB background of the same size -> RGB uint8."""
    alpha = fg_rgba[:, :, 3:4].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)[:, :, None]  # smooth mask edges (single channel, color-agnostic)
    fg_rgb = fg_rgba[:, :, :3].astype(np.float32)
    return (fg_rgb * alpha + bg_rgb * (1.0 - alpha)).astype(np.uint8)


def _finalize_tensor(np_rgb, mean, std, normalize, use_prefetcher, dump_subdir=None):
    """HWC uint8 RGB -> CHW tensor; normalized float unless prefetcher / normalize=False (uint8)."""
    _maybe_dump(np_rgb, dump_subdir)  # optional debug dump of the post-augmentation image
    t = torch.from_numpy(np.ascontiguousarray(np_rgb)).permute(2, 0, 1).contiguous()
    if use_prefetcher or not normalize:
        return t  # uint8; prefetcher scales & normalizes on device
    t = t.float().div_(255.0)
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    return t.sub_(mean_t).div_(std_t)


def _resize_rgb(np_rgb, h, w):
    """Resize an RGB numpy image to (h, w) via PIL (stretch; ignores aspect)."""
    if np_rgb.shape[0] == h and np_rgb.shape[1] == w:
        return np_rgb
    return np.array(PILImage.fromarray(np_rgb).resize((w, h), PILImage.BILINEAR))


def _draft_decode(pil_img, h, w):
    """Let the JPEG decoder do the first downscale, in the DCT domain (1/2, 1/4, 1/8 only).

    Sources here run to 10 MP while nothing above (h, w) survives the pipeline, and decoding
    then resizing full resolution costs ~10ms of the ~14ms per sample -- more than the entire
    augmentation. draft() moves that shrink inside the decoder, which is several times cheaper.

    Only effective while the JPEG is still lazy: train.py passes input_img_mode=None under
    --nobg (see train.py:702) so timm's dataset does not .convert() -- and therefore does not
    decode -- before the transform runs. Safe otherwise: draft() is a documented no-op once the
    image is loaded, and for non-JPEG formats, so PNG cutouts are untouched. It also never
    scales below the request (it picks a power-of-two reduction that keeps BOTH axes >= the
    requested size), so an elongated 2766x804 source is left alone rather than crushed.

    mode is passed as None deliberately: the caller converts right after, and asking draft()
    for a mode change can switch a JPEG to L/YCbCr decoding.
    """
    try:
        pil_img.draft(None, (w, h))
    except (AttributeError, ValueError, OSError):
        pass  # already loaded, or a format without draft support -> full-size decode
    return pil_img


def _stretch_rgba(rgba, h, w):
    """Resize an RGBA numpy image to (h, w) via PIL (stretch; ignores aspect)."""
    return np.array(PILImage.fromarray(rgba, mode='RGBA').resize((w, h), PILImage.BILINEAR))


def _fit_rgba(rgba, h, w, squash):
    """squash -> stretch to (h,w); else -> aspect-preserving letterbox (transparent pad)."""
    return _stretch_rgba(rgba, h, w) if squash else _letterbox_rgba(rgba, h, w)


def _fit_rgb(rgb, h, w, squash, fill):
    """squash -> stretch to (h,w); else -> aspect-preserving letterbox (padded with fill)."""
    return _resize_rgb(rgb, h, w) if squash else _letterbox_rgb(rgb, h, w, fill)


class _NoBgTrainTransform:
    """Fit the input to the pipeline's canvas (2*img_size under build_aug_pipeline, else img_size),
    then apply the full augmentation pipeline. Handles both:
      - RGBA cutouts (background-removed PNGs) -> composite onto a random background (bg-swap);
      - RGB photos -> fit only (no composite), so the whole dataset shares this albumentations path.
    Note build_aug_pipeline does its own square resize/crop, so it distorts aspect (not
    shape-preserving); the `squash` fit only controls the initial placement.
    """

    def __init__(self, img_size, mean, std, normalize, use_prefetcher, squash=False, fill=(255, 255, 255)):
        self.h, self.w = _as_size(img_size)
        self.squash = squash
        self.full_aug = USE_FULL_AUG
        self.aug = build_aug_pipeline(img_size=min(self.h, self.w)) if self.full_aug \
            else build_photometric_pipeline()
        # CANVAS the pipeline actually consumes, which is NOT the model's input size:
        # build_aug_pipeline normalises to a 2*img_size square (the A.Resize above) and only crops
        # back to img_size at the end. Fitting the input to (h, w) instead meant handing it a 224
        # image that A.Resize immediately upsampled to 448 -- every geometric and photometric op
        # then ran on interpolated pixels, and nothing above 224 ever reached the model no matter
        # how large the source was. Fitting to the canvas makes that A.Resize a no-op on real
        # pixels. Doubles as the _draft_decode budget, since it is exactly the detail we can use.
        # The photometric-only pipeline never resizes, so (h, w) is right for it.
        self.canvas_h, self.canvas_w = (2 * self.h, 2 * self.w) if self.full_aug else (self.h, self.w)
        self.persp = make_bottom_top_perspective_transform()
        self.mean, self.std = mean, std
        self.normalize, self.use_prefetcher = normalize, use_prefetcher
        self.fill = tuple(fill)  # RGB

    @staticmethod
    def _is_nobg(img) -> bool:
        name = os.path.basename(getattr(img, 'filename', '') or '')
        if name:
            return name.startswith('nobg_') and name.lower().endswith('.png')
        return getattr(img, 'mode', '') == 'RGBA'  # fallback when filename unavailable

    def __call__(self, pil_img):
        _draft_decode(pil_img, self.canvas_h, self.canvas_w)  # must precede any decode below
        ch, cw = self.canvas_h, self.canvas_w
        #if pil_img.mode == 'RGBA':
        if self._is_nobg(pil_img):
            # background-removed cutout: composite onto a random background (bg-swap)
            fg = self.persp(image=np.array(pil_img.convert('RGBA')))['image']
            fg = _fit_rgba(fg, ch, cw, self.squash)  # (canvas,4)
            alpha = fg[:, :, 3:4].astype(np.float32) / 255.0
            bg = _bg_rect_rgb(_NOBG_BG_PATHS, ch, cw, self.fill)     # (canvas,3) real bg
            comp = (fg[:, :, :3].astype(np.float32) * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        else:
            # plain RGB photo: fit to the canvas, no composite
            comp = _fit_rgb(np.array(pil_img.convert('RGB')), ch, cw, self.squash, self.fill)
        out = self.aug(image=comp)['image']
        if self.full_aug:
            out = _resize_rgb(out, self.h, self.w)     # build_aug_pipeline outputs square -> (h,w)
        return _finalize_tensor(out, self.mean, self.std, self.normalize, self.use_prefetcher,
                                dump_subdir=_class_of(pil_img))


class _NoBgEvalTransform:
    def __init__(self, img_size, mean, std, normalize, use_prefetcher, squash=False, fill=(255, 255, 255)):
        self.h, self.w = _as_size(img_size)
        self.squash = squash
        self.mean, self.std = mean, std
        self.normalize, self.use_prefetcher = normalize, use_prefetcher
        self.fill = tuple(fill)  # RGB

    def __call__(self, pil_img):
        # Eval never augments, so (h, w) IS the whole decode budget -- no 2x aug canvas to feed.
        # Without this, validation was the only remaining full-resolution decode in the run and
        # starved the GPU: measured sm 7% during validate() vs 63% while training.
        if USE_EVAL_DRAFT:
            _draft_decode(pil_img, self.h, self.w)
        # deterministic: fit input to (h,w) per crop mode (eval never augments)
        if pil_img.mode == 'RGBA':
            # cutout: composite onto solid fill
            fg = _fit_rgba(np.array(pil_img.convert('RGBA')), self.h, self.w, self.squash)
            alpha = fg[:, :, 3:4].astype(np.float32) / 255.0
            bg = np.full((self.h, self.w, 3), self.fill, dtype=np.uint8)
            comp = (fg[:, :, :3].astype(np.float32) * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        else:
            # plain RGB photo: fit to the box, no composite
            comp = _fit_rgb(np.array(pil_img.convert('RGB')), self.h, self.w, self.squash, self.fill)
        return _finalize_tensor(comp, self.mean, self.std, self.normalize, self.use_prefetcher,
                                dump_subdir=_class_of(pil_img))


def transform_nobg_train(
        img_size: Union[int, Tuple[int, int]] = 224,
        scale: Optional[Tuple[float, float]] = None,
        ratio: Optional[Tuple[float, float]] = None,
        train_crop_mode: Optional[str] = None,
        hflip: float = 0.5,
        vflip: float = 0.,
        color_jitter: Union[float, Tuple[float, ...]] = 0.4,
        color_jitter_prob: Optional[float] = None,
        force_color_jitter: bool = False,
        grayscale_prob: float = 0.,
        gaussian_blur_prob: float = 0.,
        auto_augment: Optional[str] = None,
        interpolation: str = 'random',
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
        re_prob: float = 0.,
        re_mode: str = 'const',
        re_count: int = 1,
        re_num_splits: int = 0,
        use_prefetcher: bool = False,
        normalize: bool = True,
        separate: bool = False,
        naflex: bool = False,
        patch_size: Union[int, Tuple[int, int]] = 16,
        max_seq_len: int = 576,
        patchify: bool = False,
        patchify_channels_last: bool = True,
):
    """Training transform for background-removed RGBA PNG inputs (bg-swap + albumentations aug).

    Signature mirrors transforms_imagenet_train for drop-in use; imagenet-style aug kwargs are
    accepted but ignored (the albumentations pipeline governs augmentation). `train_crop_mode`
    selects fit: 'squash' stretches to the box, anything else letterboxes (aspect preserved).
    """
    return _NoBgTrainTransform(
        img_size, mean, std, normalize, use_prefetcher, squash=(train_crop_mode == 'squash'))


def transform_nobg_eval(
        img_size: Union[int, Tuple[int, int]] = 224,
        crop_pct: Optional[float] = None,
        crop_mode: Optional[str] = None,
        crop_border_pixels: Optional[int] = None,
        interpolation: str = 'bilinear',
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
        use_prefetcher: bool = False,
        normalize: bool = True,
        naflex: bool = False,
        patch_size: Union[int, Tuple[int, int]] = 16,
        max_seq_len: int = 576,
        patchify: bool = False,
        patchify_channels_last: bool = True,
):
    """Eval transform for background-removed RGBA PNG inputs (composite on solid bg, resize, norm).

    Signature mirrors transforms_imagenet_eval for drop-in use. `crop_mode='squash'` stretches to
    the box; anything else letterboxes (aspect preserved).
    """
    return _NoBgEvalTransform(
        img_size, mean, std, normalize, use_prefetcher, squash=(crop_mode == 'squash'))


# def test():
#       path = _collect_bg_paths('C:\\shennan\\val2017')
#       pil_in= PILImage.open('C:\\shennan\\crop\\306_crop\\13_140_508_768.jpg')
#       fg_rgba = np.array(rembg_remove(pil_in, session=REMBG_SESSION))   # RGBA, computed once
#       out =_bgswap(fg_rgba,path)
#       out.save('C:\\shennan\\crop\\306_crop_aug\\ppppp.jpg')


# def test2():
#       path = _collect_bg_paths('/home/keyong/cls/DMTrain_keyong/experiments')
#       pil_in= PILImage.open('/home/keyong/cls/pepsi_merge/train/CRM060109/0_4_.jpg')
#       fg_rgba = np.array(rembg_remove(pil_in, session=REMBG_SESSION))   # RGBA, computed once
      

#       PILImage.fromarray(fg_rgba).save(
#             "/home/keyong/cls/DMTrain_keyong/newspage/classification/tests/foreground.png"
#         )
#       PILImage.fromarray(fg_rgba[:, :, 3]).save(
#             "/home/keyong/cls/DMTrain_keyong/newspage/classification/tests/mask.png"
#         )

#       out =_bgswap(fg_rgba,path)
#       out.save('/home/keyong/cls/DMTrain_keyong/newspage/classification/tests/ppppp.jpg')



# if __name__ == "__main__":
#     test2()