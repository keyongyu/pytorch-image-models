"""Open-set recognition on top of a trained closed-set classifier.

Uses the pre-logits feature (pooled vector before the FC layer) and cosine distance
to per-class mean prototypes. A test image whose nearest prototype is farther than a
threshold is rejected as 'unknown'.
"""
import os
import glob
import shutil
from pathlib import Path
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import timm
import onnxruntime as ort
import onnx
import cv2
from timm.models import load_checkpoint
from torchvision import transforms


_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'tf_efficientnet_lite0.in1k'
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
# Downscale filter the CLIENT must use. Part of the preprocessing contract, like MEAN/STD: an
# aliasing filter shifts features silently. cv2's default (INTER_LINEAR) reads only ~2% of source
# pixels on a large reduction and does NOT anti-alias -- clients must pass INTER_AREA explicitly.
# `load_image_uint8` below uses exactly this filter, so prototypes are built on the same
# preprocessing the client runs rather than something merely equivalent to it.
RESIZE_FILTER = 'area'
# Default floor on the reject threshold, overridable with --min-threshold. A small threshold
# rejects genuine product images: measured on posmlv, 0.02 costs 10% false-reject and 0.007 costs
# 50%, while buying almost nothing in false-accept (which stays under 0.2% across that whole
# range). Calibration is clamped up to the floor and warns when the clamp binds -- a suggestion
# below it means the data is wrong, not that the threshold should be tiny.
DEFAULT_MIN_THRESHOLD = 0.2

# defaults for the CLI args (--data-dir / --img-size)
_DEFAULT_DATA_DIR = os.path.join(_HERE, '..', 'posmlv')
_DEFAULT_IMG_SIZE = 224


def _short_path(path, root) -> str:
    """Path relative to `root` for compact stdout logging.

    Falls back to the original string when `path` is not under `root` (e.g. a `--image` elsewhere
    or a different drive), so the printed path is always valid.
    """
    try:
        rel = os.path.relpath(str(path), str(root))
    except ValueError:                     # different drive on Windows
        return str(path)
    return rel if not rel.startswith('..') else str(path)


def _write_aligned_csv(path, header, rows) -> None:
    """Write `header` + `rows` as a comma-separated file, space-padded so columns line up.

    Each column is padded to its widest cell; any CSV reader that strips surrounding whitespace
    parses it unchanged. Creates the parent directory if needed.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table = [tuple(header)] + [tuple(r) for r in rows]
    widths = [max(len(row[c]) for row in table) for c in range(len(header))]
    with open(path, 'w') as f:
        for row in table:
            f.write(', '.join(cell.ljust(widths[c]) for c, cell in enumerate(row)).rstrip() + '\n')


def _read_class_names(class_map_path: str) -> list:
    """Ordered class names from the class-map file (one non-empty line per class).

    Line order defines class index order; num_classes = len(returned list).
    """
    with open(class_map_path) as f:
        return [line.strip() for line in f if line.strip()]


def _scan_eval_images(data_dir: str) -> list:
    """Collect images to predict from data_dir's val/ and test/ folders (recursively)."""
    paths = []
    for split in ('val', 'test'):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        found = sorted(
            p for p in glob.glob(os.path.join(split_dir, '**', '*'), recursive=True)
            if os.path.isfile(p) and p.lower().endswith(_IMG_EXTS)
        )
        print(f'Scanning {split_dir}: {len(found)} images')
        paths += found
    return paths


def load_image_uint8(path: str, img_size: int) -> np.ndarray:
    """Decode and squash-resize one image to (img_size, img_size); returns HWC uint8 RGB.

    The single decode primitive for this module. OpenCV + INTER_AREA is used deliberately: the ncnn
    client decodes with OpenCV too, so prototypes are built on exactly the deployment
    preprocessing rather than something merely equivalent to it.

    INTER_AREA is not interchangeable with cv2's default. `INTER_LINEAR`/`INTER_CUBIC` use a fixed
    2x2/4x4 kernel regardless of scale, so on a large downscale they read ~2-3% of the source and
    alias; INTER_AREA reads 100%. See RESIZE_FILTER and openset.md section 5.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)          # 3-channel; alpha dropped, as PIL convert('RGB') did
    if bgr is None:
        raise OSError(f'could not decode image: {path}')
    bgr = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)   # squash, not letterbox
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_transform(img_size: int, mean=MEAN, std=STD):
    """Return ``fn(path) -> [3, H, W] float32`` normalized tensor, for single-image paths.

    Takes a PATH rather than a PIL image: decoding is part of the preprocessing contract (see
    `load_image_uint8`), so it belongs inside the transform rather than at each call site.
    """
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def _tf(path: str) -> torch.Tensor:
        arr = load_image_uint8(path, img_size)
        t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
        return (t - mean_t) / std_t

    return _tf


class _NpAugTransform:
    """npaug (build_aug_pipeline) augmentation -> normalized tensor, for threshold calibration.

    Used during calibration so the in-distribution distance spread reflects real-world variation
    (lighting/blur/geometry) instead of the optimistic near-duplicate video frames. The prototype
    itself is still built from clean images.
    """

    def __init__(self, img_size: int):
        from nptools.npaug import build_aug_pipeline
        self.aug = build_aug_pipeline(img_size=img_size)     # ends with Resize -> (img_size, img_size)
        self.norm = transforms.Normalize(mean=MEAN, std=STD)

    def __call__(self, path: str):
        # Decoded at FULL size, not via load_image_uint8: the npaug pipeline runs its geometric
        # transforms before its own internal Resize, so pre-shrinking here would change them.
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f'could not decode image: {path}')
        out = self.aug(image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))['image']   # HWC uint8 RGB
        t = torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1).float().div_(255.0)
        return self.norm(t)


class _GpuAugBatch(torch.nn.Module):
    """Batch augmentation on GPU for threshold calibration.

    Input:  [B, 3, H, W] float in [0, 1] on the target device.
    Output: [B, 3, H, W] augmented + normalized, ready for the backbone.

    Workers only decode+resize (fast), all heavy augmentation runs on GPU.

    The augmentation is applied ONE IMAGE AT A TIME: torchvision v2 transforms treat a batched
    tensor as a single sample with leading batch dims and draw one parameter set per call, so
    passing [B, 3, H, W] straight through would give every image in the batch the same jitter/blur
    and warp them all-or-none. That collapses the distance spread this pass exists to widen.
    """

    def __init__(self, img_size: int):
        super().__init__()
        from torchvision.transforms import v2
        self.aug = v2.Compose([
            v2.RandomPerspective(distortion_scale=0.2, p=0.5),
            v2.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.05),
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])
        self.norm = transforms.Normalize(mean=MEAN, std=STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.stack([self.aug(img) for img in x])   # per-sample params, see class docstring
        return self.norm(out)


@torch.no_grad()
def extract_feature(model: torch.nn.Module, transform, img_path: str, device: str) -> torch.Tensor:
    x = transform(img_path).unsqueeze(0).to(device)
    feat = model.forward_head(model.forward_features(x), pre_logits=True)  # [1, 1280]
    return F.normalize(feat, dim=1).squeeze(0).cpu()  # L2-normalized [1280]


_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def _worker_resize_uint8(img_size: int):
    """Return ``fn(path) -> [3, H, W] uint8`` tensor, for the bulk extraction path.

    Stops before the float conversion and normalize so the worker hands back **uint8**: each
    3x224x224 sample is then 150 KB instead of 602 KB crossing the worker->main boundary.
    `_gpu_normalize` finishes the job on-device, so the result matches `build_transform(img_size)`
    exactly — both go through `load_image_uint8`, so there is one decode path, not two.
    """
    def _tf(path: str) -> torch.Tensor:
        return torch.from_numpy(load_image_uint8(path, img_size)).permute(2, 0, 1)

    return _tf


_GPU_NORM_CACHE: dict = {}


def _gpu_normalize(x: torch.Tensor) -> torch.Tensor:
    """(x/255 - MEAN) / STD on-device. `x` is float in [0, 1]; mean/std tensors are cached."""
    key = (x.device, x.dtype)
    if key not in _GPU_NORM_CACHE:
        _GPU_NORM_CACHE[key] = (
            torch.tensor(MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1),
            torch.tensor(STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1),
        )
    mean, std = _GPU_NORM_CACHE[key]
    return (x - mean) / std


class _FeatDataset(torch.utils.data.Dataset):
    """Yield (image_tensor, class_idx) for prototype feature extraction.

    `transform` is a ``fn(path) -> [3, H, W] tensor``; decode runs inside it, in the DataLoader
    worker processes. Unreadable images return a zero tensor with class_idx = -1 so the batch loop
    can skip them without crashing.

    `out_dtype` must match what `transform` returns, so the zero tensor for an unreadable image
    collates with the rest of its batch instead of raising a dtype mismatch.
    """

    def __init__(self, samples, transform, img_size, out_dtype=torch.uint8):
        self.samples = samples          # list of (path, class_idx)
        self.transform = transform
        self.img_size = img_size
        self.out_dtype = out_dtype

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cls = self.samples[i]
        try:
            return self.transform(path), cls
        except (IOError, OSError, cv2.error):
            return torch.zeros(3, self.img_size, self.img_size, dtype=self.out_dtype), -1


def _worker_init(worker_id: int) -> None:
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    # npaug/albumentations draw from the python and numpy RNGs, which DataLoader does not seed for
    # us. Derive both from torch's per-worker seed so --cpu-aug is reproducible too.
    import random as _random
    seed = torch.initial_seed() % (2 ** 32)
    _random.seed(seed)
    np.random.seed(seed)


def _nearest_other_class(feat: torch.Tensor, own_cls: str, proto_mat, proto_names):
    """Nearest prototype to `feat` among classes other than `own_cls`.

    Args:
        feat: L2-normalized feature [D].
        own_cls: the image's labeled class, excluded from the search.
        proto_mat: L2-normalized prototype matrix [C, D] (or None if unavailable).
        proto_names: class names aligned with proto_mat rows.

    Returns:
        (class_name, cosine_distance) for the closest other class, or (None, None) when
        there is no other class to compare against.
    """
    if proto_mat is None or len(proto_names) < 2:
        return None, None
    dists = 1.0 - (proto_mat @ feat)                     # [C] cosine distance to every class
    dists = dists.clone()
    dists[proto_names.index(own_cls)] = float('inf')     # mask out the own class
    nn_idx = int(dists.argmin())
    return proto_names[nn_idx], float(dists[nn_idx])


def _collect_samples(train_root: str, class_map_names) -> tuple:
    """Collect (class_names, samples) from `train_root`, following class-map order.

    Only class-map folders are used, in that order; folders not in the map are ignored. A missing
    or empty folder is warned about and skipped, so `class_names` may be shorter than
    `class_map_names`. `samples` is a flat list of (path, class_idx) indexing into `class_names`.
    """
    class_names, samples = [], []
    for cls in class_map_names:
        cls_dir = os.path.join(train_root, cls)
        if not os.path.isdir(cls_dir):
            print(f'{cls:12s}: WARNING — listed in class-map but no folder under {train_root}')
            continue
        imgs = [p for p in sorted(glob.glob(os.path.join(cls_dir, '*')))
                if p.lower().endswith(_IMG_EXTS) and not os.path.islink(p)]
        if not imgs:
            print(f'{cls:12s}: WARNING — folder exists but has no images — skipped')
            continue
        if len(imgs) < 10:
            print(f'{cls:12s}: WARNING — only {len(imgs)} images (<10); prototype '
                  f'may be unreliable')
        idx = len(class_names)
        class_names.append(cls)
        samples += [(p, idx) for p in imgs]

    if not class_names:
        raise RuntimeError(f'No class-map class had images under {train_root}')
    return class_names, samples


def _extract_features(model, transform, samples, n_classes: int, img_size: int, device: str,
                      batch_size: int = 64, num_workers: int = 16, repeat: int = 1,
                      gpu_aug: torch.nn.Module | None = None, track_paths: bool = False):
    """Feature-extract `samples` with `transform`, grouped by class index.

    `repeat` > 1 tiles the list so stochastic augmentation produces multiple independent views in
    a single DataLoader pass. `gpu_aug`: optional _GpuAugBatch applied on-device before the
    backbone. `track_paths`: if True, also return pbc (paths by class, parallel to fbc).

    Dispatches on the dtype the worker hands back, so both pipelines share this loop:
    - **uint8** (the fast bulk path, `_worker_resize_uint8`): scaled to [0,1] here, then either
      normalized on-device or handed to `gpu_aug`, which normalizes itself.
    - **float32** (`_NpAugTransform`): already scaled and normalized in the worker; passed through.
    """
    actual = samples * repeat if repeat > 1 else samples
    # uint8 for the bulk cv2 loader, float32 for _NpAugTransform, which normalizes in the worker
    out_dtype = torch.float32 if isinstance(transform, _NpAugTransform) else torch.uint8
    loader = torch.utils.data.DataLoader(
        _FeatDataset(actual, transform, img_size, out_dtype=out_dtype),
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=(str(device) != 'cpu'),
        worker_init_fn=_worker_init,
    )
    fbc = [[] for _ in range(n_classes)]
    pbc = [[] for _ in range(n_classes)] if track_paths else None
    path_iter = iter(p for p, _ in actual) if track_paths else None
    with torch.inference_mode():
        for x, cls in loader:
            x = x.to(device, non_blocking=True)
            if x.dtype == torch.uint8:
                x = x.float().div_(255.0)
                if gpu_aug is None:
                    x = _gpu_normalize(x)
            if gpu_aug is not None:
                x = gpu_aug(x)
            f = model.forward_head(model.forward_features(x), pre_logits=True)  # [B, 1280]
            f = F.normalize(f, dim=1).cpu()
            for fi, ci in zip(f, cls.tolist()):
                path = next(path_iter) if track_paths else None
                if ci >= 0:                               # skip unreadable (ci == -1)
                    fbc[ci].append(fi)
                    if pbc is not None:
                        pbc[ci].append(path)
    return (fbc, pbc) if track_paths else fbc


def _aug_probe_features(model, samples, class_names, img_size: int, device: str, batch_size: int,
                        num_workers: int, aug_views: int, aug_max_per_class: int, cpu_aug: bool,
                        seed: int = 0):
    """Augmented probe features per class, for realistic calibration distances.

    Subsamples to `aug_max_per_class` per class, then takes `aug_views` stochastic views of each.
    GPU mode (default): workers decode+resize, _GpuAugBatch runs on device — fast. CPU mode
    (`cpu_aug`): npaug/albumentations in the workers — faithful to training augmentation, slower.
    Prototypes are never built from these; only the probe distances are.

    `seed` fixes both the subsample choice and the augmentation itself. That matters because with
    `--threshold auto` the calibrated value is baked into the exported model: an unseeded RNG would
    make the same checkpoint export a different deployed artifact on every run.
    """
    import random as _random
    torch.manual_seed(seed)                    # GPU aug + the DataLoader's per-worker seeds
    by_class: list[list] = [[] for _ in class_names]
    for p, ci in samples:
        by_class[ci].append((p, ci))
    rng = _random.Random(seed)
    chosen_all: list = []
    for cls_samples in by_class:
        cap = aug_max_per_class if aug_max_per_class > 0 else len(cls_samples)
        chosen_all.extend(rng.sample(cls_samples, min(cap, len(cls_samples))))
    views = max(1, aug_views)
    backend = 'CPU npaug' if cpu_aug else 'GPU torchvision'
    print(f'aug probe pass: {len(chosen_all)} images x {views} views = {len(chosen_all) * views} '
          f'total (capped at {aug_max_per_class}/class, {backend})')
    if cpu_aug:
        return _extract_features(model, _NpAugTransform(img_size), chosen_all, len(class_names),
                                 img_size, device, batch_size, num_workers, repeat=views)
    gpu_aug = _GpuAugBatch(img_size).to(device).eval()
    return _extract_features(model, _worker_resize_uint8(img_size), chosen_all, len(class_names),
                             img_size, device, batch_size, num_workers, repeat=views,
                             gpu_aug=gpu_aug)


def extract_train_features(model, transform, device: str, data_dir: str, img_size: int,
                           class_map_names, batch_size: int = 64, num_workers: int = 16) -> dict:
    """One clean feature pass over ``<data_dir>/train/``, shareable by every consumer.

    Prototype building and threshold calibration both need exactly this, so running it once and
    passing the result to both (`feats=` on either) halves the work of a `--threshold auto` export.
    """
    class_names, samples = _collect_samples(os.path.join(data_dir, 'train'), class_map_names)
    model.eval()
    # `transform` is accepted for signature symmetry but the bulk path uses the uint8 worker
    # transform + `_gpu_normalize`, which reproduces build_transform(img_size) exactly while
    # moving 4x less data across the worker boundary. Verified feature-identical to 1e-6.
    feats_by_class, paths_by_class = _extract_features(
        model, _worker_resize_uint8(img_size), samples, len(class_names), img_size, device,
        batch_size=batch_size, num_workers=num_workers, track_paths=True)
    return {'class_names': class_names, 'samples': samples,
            'feats_by_class': feats_by_class, 'paths_by_class': paths_by_class}


def build_prototypes(model, transform, device: str, data_dir: str, img_size: int,
                     class_map_names, batch_size: int = 64, num_workers: int = 16,
                     threshold: float = DEFAULT_MIN_THRESHOLD, copy_inliers: bool = False,
                     feats: dict | None = None, min_threshold: float = DEFAULT_MIN_THRESHOLD):
    """Compute an L2-normalized mean feature per class from the training split.

    Only the classes listed in `class_map_names` are used, in that order (folders not in the
    class-map are ignored). Parallelized: decode across `num_workers` workers, backbone on batches.

    Returns one prototype per class-map entry, so `class_names` == `class_map_names` and a class
    index means the same thing everywhere (class-map file, exported matrix, client). A class-map
    entry with no usable images -- folder missing, folder empty, or every image an outlier -- is
    warned about and gets a ZERO prototype, which always scores the maximum cosine distance (1.0)
    and is therefore always rejected. See the comment at the assignment for why zero is the correct
    sentinel and not just a convenient one.

    Every class gets the SAME reject threshold (`threshold`). Per-class thresholds taken from each
    class's own training spread were removed: with one video per class the training crops are
    near-duplicate frames, so that spread measures how the clip happened to be shot rather than how
    the product varies, and it never sees a negative at all. Use `calibrate_threshold`
    (`--calibrate`) to choose the constant from measured false-reject / false-accept rates.

    If `copy_inliers` is set, the good inlier crops (cosine dist <= OUTLIER_THR to their class
    prototype) are copied into `<data_dir>/inlier/<class>/` for training a clean classifier.
    """
    if threshold < min_threshold:
        raise ValueError(f'threshold {threshold} is below the {min_threshold} floor; see '
                         f'DEFAULT_MIN_THRESHOLD for why. Use calibrate_threshold() to pick one, '
                         f'or lower the floor with --min-threshold if you really mean it.')
    t_start = time.time()
    # PROTOTYPE from clean images (matches inference preprocessing; keep it on the clean manifold).
    if feats is None:
        feats = extract_train_features(model, transform, device, data_dir, img_size,
                                       class_map_names, batch_size, num_workers)
    class_names, samples = feats['class_names'], feats['samples']
    feats_by_class, paths_by_class = feats['feats_by_class'], feats['paths_by_class']

    OUTLIER_THR = 0.5   # cosine dist > 0.5 ≈ nearly orthogonal; same-class images should be << 0.2
    outlier_root = Path(data_dir) / 'outlier'
    inlier_root = Path(data_dir) / 'inlier'   # clean (dist<=thr) images, for training a clean classifier
    n_inliers_copied = 0
    outlier_rows = []   # CSV rows: (classtype/file, bad_dist, nearest_type, nearest_dist)
    inlier_rows = []    # CSV rows: (classtype/file, distance) — only when copy_inliers

    # Pre-pass: an initial (pre-exclusion) mean prototype per class, used only to tell WHICH other
    # class an outlier is closest to (mislabel diagnosis vs. genuine unknown). Not the final proto.
    init_proto_names = [cls for i, cls in enumerate(class_names) if feats_by_class[i]]
    init_proto_mat = torch.stack([
        F.normalize(torch.stack(feats_by_class[i]).mean(dim=0), dim=0)
        for i, cls in enumerate(class_names) if feats_by_class[i]
    ]) if init_proto_names else None                     # [C, D], L2-normalized

    prototypes = {}
    for i, cls in enumerate(class_names):
        if not feats_by_class[i]:
            continue
        feats = torch.stack(feats_by_class[i])               # [N, 1280] clean

        # Robust prototype: exclude outliers before computing the final mean.
        proto0 = F.normalize(feats.mean(dim=0), dim=0)
        dists0 = 1.0 - (feats @ proto0)
        outlier_mask = dists0 > OUTLIER_THR
        n_out = int(outlier_mask.sum())
        if n_out:
            paths = paths_by_class[i]
            outlier_cls_dir = outlier_root / cls
            outlier_cls_dir.mkdir(parents=True, exist_ok=True)
            print(f'{cls:12s}: WARNING — {n_out}/{len(feats)} outlier(s) excluded '
                  f'(dist>{OUTLIER_THR}) → {_short_path(outlier_cls_dir, data_dir)}')
            for j, is_out in enumerate(outlier_mask.tolist()):
                if is_out:
                    src = Path(paths[j])
                    dst = outlier_cls_dir / src.name
                    shutil.copy2(src, dst)
                    # Nearest OTHER class: small → likely mislabel; large → genuine unknown.
                    near_cls, near_dist = _nearest_other_class(
                        feats[j], cls, init_proto_mat, init_proto_names)
                    near_str = (f'nearest_other={near_cls} ({near_dist:.4f})'
                                if near_cls is not None else 'nearest_other=n/a')
                    print(f'               dist={dists0[j]:.4f}  {near_str}  '
                          f'{_short_path(paths[j], data_dir)}')
                    outlier_rows.append((
                        f'{cls}/{src.name}', f'{float(dists0[j]):.4f}',
                        near_cls if near_cls is not None else '',
                        f'{near_dist:.4f}' if near_dist is not None else '',
                    ))
            feats = feats[~outlier_mask]
            if len(feats) == 0:
                print(f'{cls:12s}: WARNING — all images are outliers, class skipped')
                continue

        # Copy the good inliers (dist<=thr) into inlier/<cls>/ for training a clean classifier,
        # recording each one's distance to its own class prototype for inlier.txt.
        if copy_inliers:
            inlier_cls_dir = inlier_root / cls
            inlier_cls_dir.mkdir(parents=True, exist_ok=True)
            for j, is_out in enumerate(outlier_mask.tolist()):
                if is_out:
                    continue
                src = Path(paths_by_class[i][j])
                shutil.copy2(src, inlier_cls_dir / src.name)
                inlier_rows.append((f'{cls}/{src.name}', f'{float(dists0[j]):.4f}'))
                n_inliers_copied += 1

        proto = F.normalize(feats.mean(dim=0), dim=0)        # normalized class mean
        prototypes[cls] = proto
        dists = 1.0 - (feats @ proto)                        # reported only; not used for the thr
        print(f'{cls:12s}: {len(feats):4d} imgs, mean={dists.mean():.4f}, max={dists.max():.4f}')

    # Keep ONE prototype per class-map entry, in class-map order, so a class index means the same
    # thing in the class-map file, the exported matrix and the client. Dropping data-less classes
    # instead (as this used to) silently shifts every index after the first gap, which is
    # unrecoverable on the client side -- it only has the class-map file to index with.
    #
    # A class with no data gets a ZERO prototype, which is the correct sentinel rather than merely a
    # convenient one: the pre-logits feature is post-ReLU6 and therefore non-negative, so cosine
    # similarity lies in [0, 1] and cosine distance in [0, 1]. A zero row scores sims = 0, i.e.
    # dist == 1.0, the MAXIMUM attainable distance -- so it can never strictly win the argmin
    # against a class that has data, and if it ever tied at 1.0 the margin (1.0 - threshold) is
    # positive, i.e. the verdict is 'unknown' anyway. Note F.normalize() is deliberately NOT applied
    # here; normalizing a zero vector is NaN, which would poison argmin instead of losing it.
    missing = [c for c in class_map_names if c not in prototypes]
    if missing:
        if not prototypes:
            raise RuntimeError(
                f'no class produced a prototype ({len(class_map_names)} class-map entries, all '
                f'without usable images under {data_dir}/train) -- cannot infer the feature '
                f'dimension for placeholders, and an all-placeholder model would reject everything')
        dim = next(iter(prototypes.values())).shape[0]
        proto_dtype = next(iter(prototypes.values())).dtype
        for cls in missing:
            prototypes[cls] = torch.zeros(dim, dtype=proto_dtype)
        print(f'{len(missing)} class(es) without usable images get a zero prototype '
              f'(distance always 1.0 -> always rejected): {", ".join(missing)}')
    class_names = list(class_map_names)
    if outlier_rows:
        csv_path = outlier_root / 'outlier.txt'
        _write_aligned_csv(
            csv_path, ('classtype/file', 'bad_dist', 'nearest_type', 'nearest_dist'), outlier_rows)
        print(f'outlier list: {len(outlier_rows)} rows → {_short_path(csv_path, data_dir)}')
    if copy_inliers:
        print(f'inliers copied: {n_inliers_copied} images → {_short_path(inlier_root, data_dir)}')
        if inlier_rows:
            _write_aligned_csv(inlier_root / 'inlier.txt', ('classtype/file', 'distance'), inlier_rows)
    print(f'build_prototypes: {len(class_names)} classes '
          f'({len(class_names) - len(missing)} with data + {len(missing)} placeholder), '
          f'{len(samples)} images, constant threshold={threshold:.4f}, in {time.time() - t_start:.1f}s')
    return prototypes, class_names


def _loo_positive_dists(feats: torch.Tensor) -> torch.Tensor:
    """Leave-one-image-out distance from each feature to its own class prototype.

    The plain distance ``1 - f_j . normalize(mean(F))`` is biased low because f_j is itself part of
    the mean it is scored against — and with one video per class the crops are near-duplicate
    frames, so the bias is large. Excluding f_j from the mean removes it exactly:
    ``proto_(-j) = normalize((S - f_j) / (N-1))`` with ``S = sum(F)``; the 1/(N-1) cancels under
    normalize, so this is one vectorized expression over all j.

    Returns [N] distances, or an empty tensor when N < 3 (LOO is meaningless for a tiny class).
    """
    if feats.shape[0] < 3:
        return feats.new_empty(0)
    loo = F.normalize(feats.sum(dim=0, keepdim=True) - feats, dim=1)   # [N, D]
    return 1.0 - (feats * loo).sum(dim=1)                              # [N]


@torch.no_grad()
def calibrate_threshold(model, transform, device: str, data_dir: str, img_size: int,
                        class_map_names, batch_size: int = 64, num_workers: int = 16,
                        aug: bool = True, aug_views: int = 2, aug_max_per_class: int = 100,
                        cpu_aug: bool = False, seed: int = 0, feats: dict | None = None,
                        min_threshold: float = DEFAULT_MIN_THRESHOLD):
    """Sweep candidate constant thresholds, reporting MEASURED false-reject / false-accept rates.

    Uses only trusted labelled data under ``<data_dir>/train/`` — no curated negative set needed.
    Two leave-one-out estimators, both from a single feature-extraction pass:

    - FALSE REJECT (a known product wrongly rejected): the probe must not be inside the prototype
      it is scored against. With ``aug`` the probe is an augmented view, which is never part of the
      (clean) prototype, so the plain distance is already unbiased. Without ``aug`` the probe IS one
      of the prototype's own images, so `_loo_positive_dists` removes it from the mean first.
    - FALSE ACCEPT (an un-enrolled product wrongly accepted): leave-one-CLASS-out. For class c its
      own prototype is masked out and its images are scored against the remaining ones — by
      construction c is now a product that was never enrolled. Each prototype is built only from its
      own class's images, so dropping c leaves the others unchanged: a column mask, not a rebuild.

    Returns (suggested_threshold, d_pos, d_neg).
    """
    t_start = time.time()
    # Clean features -> prototypes, using the SAME robust construction as build_prototypes.
    if feats is None:
        feats = extract_train_features(model, transform, device, data_dir, img_size,
                                       class_map_names, batch_size, num_workers)
    class_names, samples = feats['class_names'], feats['samples']
    feats_by_class, paths_by_class = feats['feats_by_class'], feats['paths_by_class']
    n_all = len(class_names)
    print(f'\nCalibrating on {len(samples)} images across {n_all} classes '
          f'(of {len(class_map_names)} in the class-map)')
    OUTLIER_THR = 0.5
    protos, keep_names, keep_idx, keep_feats = [], [], [], []
    clean_samples, n_outliers = [], 0
    for i, cls in enumerate(class_names):
        if not feats_by_class[i]:
            continue
        f = torch.stack(feats_by_class[i])
        keep = (1.0 - (f @ F.normalize(f.mean(dim=0), dim=0))) <= OUTLIER_THR
        n_outliers += int((~keep).sum())
        f = f[keep]
        if len(f) == 0:
            continue
        protos.append(F.normalize(f.mean(dim=0), dim=0))
        keep_names.append(cls)
        keep_idx.append(i)
        keep_feats.append(f)
        # Outliers are dropped from the PROBE set too, not just from the prototype. They are
        # mislabelled crops — far from their own class, near another — so scoring them charges the
        # threshold for a false reject AND, under leave-one-class-out, a false accept, when in both
        # cases the model is behaving correctly. Leaving them in manufactures an error floor, which
        # then widens the plateau (defined as floor + 0.2pp) and drags the suggestion with it.
        clean_samples += [(paths_by_class[i][j], i) for j, k in enumerate(keep.tolist()) if k]
    if len(keep_names) < 2:
        raise RuntimeError('calibration needs at least 2 classes with usable images')
    proto_mat = torch.stack(protos)                                # [C, D]
    if n_outliers:
        print(f'excluded {n_outliers} outlier crop(s) (dist>{OUTLIER_THR}) from prototypes AND probes')

    # Probe features. Augmented views give a realistic in-distribution spread; the clean training
    # crops are near-duplicate video frames and read optimistically tight. Prototypes stay clean.
    if aug:
        probe_by_class = _aug_probe_features(model, clean_samples, class_names, img_size, device,
                                             batch_size, num_workers, aug_views,
                                             aug_max_per_class, cpu_aug, seed=seed)
        probes = [torch.stack(probe_by_class[i]) if probe_by_class[i] else None for i in keep_idx]
    else:
        probes = list(keep_feats)

    d_pos_by_class, d_neg_by_class = [], []
    for k, pf in enumerate(probes):
        if pf is None or len(pf) == 0:
            d_pos_by_class.append(torch.empty(0))
            d_neg_by_class.append(torch.empty(0))
            continue
        # positives: unbiased already when augmented, else leave-one-image-out
        d_pos_by_class.append((1.0 - (pf @ proto_mat[k])) if aug else _loo_positive_dists(pf))
        d = 1.0 - (pf @ proto_mat.t())          # [n, C]
        d[:, k] = float('inf')                  # leave-one-CLASS-out: class k is not enrolled
        d_neg_by_class.append(d.min(dim=1).values)

    d_pos = torch.cat([d for d in d_pos_by_class if d.numel()]).numpy()
    d_neg = torch.cat([d for d in d_neg_by_class if d.numel()]).numpy()
    if not len(d_pos) or not len(d_neg):
        raise RuntimeError('calibration produced no usable distances')

    # Sweep. Sorting once turns every threshold into a searchsorted, so the grid is nearly free.
    sp, sn = np.sort(d_pos), np.sort(d_neg)
    frr = lambda t: 1.0 - np.searchsorted(sp, t, side='right') / len(sp)   # known rejected
    far = lambda t: np.searchsorted(sn, t, side='right') / len(sn)         # un-enrolled accepted

    # When the two classes separate cleanly the space between them is empty, and a uniform grid
    # wastes every row on it. Sample where the curves actually move instead: the upper tail of the
    # positives (drives FRR) and the lower tail of the negatives (drives FAR).
    display = np.unique(np.concatenate([
        np.percentile(sp, [50, 75, 90, 95, 98, 99, 99.5, 99.9]),
        np.percentile(sn, [0.1, 0.5, 1, 2, 5, 10, 25]),
    ]))

    # Suggested operating point: the centre of the widest plateau where the worse of the two error
    # rates stays near its achievable floor. Both curves are flat across the empty gap, so an
    # equal-error point would be arbitrary; the plateau midpoint is the value that survives the most
    # distribution shift in either direction before it starts costing anything.
    grid = np.linspace(float(min(sp[0], sn[0])), float(max(sp[-1], sn[-1])), 4000)
    worse = np.maximum([frr(t) for t in grid], [far(t) for t in grid])
    ok = np.flatnonzero(worse <= worse.min() + 0.002)                      # within 0.2pp of floor
    splits = np.flatnonzero(np.diff(ok) > 1)
    runs = np.split(ok, splits + 1)
    plateau = max(runs, key=len)
    lo_p, hi_p = float(grid[plateau[0]]), float(grid[plateau[-1]])
    suggested = 0.5 * (lo_p + hi_p)
    if lo_p <= round(suggested, 2) <= hi_p:   # don't imply precision the plateau can't support
        suggested = round(suggested, 2)
    if suggested < min_threshold:
        print(f'\nWARNING - calibration suggested {suggested:.4f}, below the {min_threshold} floor; '
              f'clamped to {min_threshold}.\n          A suggestion this low means the positives and '
              f'negatives are not well separated:\n          check for mislabelled crops (see '
              f'outlier.txt) or classes that are near-duplicates.\n          Override the floor '
              f'with --min-threshold if the low value is genuinely what you want.')
        suggested = min_threshold

    print(f'\npositives (known, {"augmented" if aug else "leave-one-image-out"}): n={len(sp)}  '
          f'med={np.median(sp):.4f}  p95={np.percentile(sp, 95):.4f}  max={sp[-1]:.4f}')
    print(f'negatives (leave-one-class-out)                    : n={len(sn)}  '
          f'med={np.median(sn):.4f}  p05={np.percentile(sn, 5):.4f}  min={sn[0]:.4f}')
    print(f'\n{"thr":>8}  {"FRR (known rejected)":>22}  {"FAR (unenrolled accepted)":>26}')
    print(f'{"-" * 60}')
    for t in display:
        print(f'{t:8.4f}  {frr(t) * 100:21.2f}%  {far(t) * 100:25.2f}%')
    print(f'\nsafe plateau: [{lo_p:.4f} .. {hi_p:.4f}] — anywhere in here performs within 0.2pp')
    print(f'suggested   : {suggested:.4f}  '
          f'(FRR={frr(suggested) * 100:.2f}%, FAR={far(suggested) * 100:.2f}%)   '
          f'[floor {min_threshold}]')

    # Per-class breakdown at the suggested value: a bad class should be visible, not averaged away.
    rows = []
    for k, cls in enumerate(keep_names):
        dp, dn = d_pos_by_class[k], d_neg_by_class[k]
        rows.append((cls,
                     float((dp > suggested).float().mean()) if dp.numel() else float('nan'),
                     float((dn <= suggested).float().mean()) if dn.numel() else float('nan')))
    worst_frr = sorted(rows, key=lambda r: -(r[1] if r[1] == r[1] else -1))[:8]
    worst_far = sorted(rows, key=lambda r: -(r[2] if r[2] == r[2] else -1))[:8]
    print(f'\nworst classes at thr={suggested:.4f}')
    print(f'  {"by false-reject":<28}{"by false-accept":<28}')
    for a, b in zip(worst_frr, worst_far):
        print(f'  {a[0][:18]:<20}{a[1] * 100:6.1f}%  {b[0][:18]:<20}{b[2] * 100:6.1f}%')

    print('\nCAVEATS')
    print('  - LOCO negatives are other posm standees: same domain, same photographic style, so')
    print('    they sit closer to the prototypes than a random shelf photo would. FAR here is a')
    print('    PESSIMISTIC bound — real-world unknowns should be rejected at least this well.')
    print('  - No val/test split exists, so every number is train-domain.')
    print('  - The suggestion is the plateau midpoint, which weights false-reject and false-accept')
    print('    equally. There is currently no flag to bias it toward one; --min-threshold can only')
    print('    raise the floor (more permissive), not make the decision stricter.')
    print(f'\ncalibrate_threshold: {time.time() - t_start:.1f}s')
    return {
        'suggested': suggested, 'plateau': (lo_p, hi_p),
        'class_names': keep_names, 'd_pos': d_pos, 'd_neg': d_neg,
        'd_pos_by_class': [d.numpy() for d in d_pos_by_class],
        'd_neg_by_class': [d.numpy() for d in d_neg_by_class],
    }


class _OpenSetWrapper(torch.nn.Module):
    """Backbone + prototype constants + cosine-distance head, ready for ONNX/NCNN export.

    The reject threshold is a single SCALAR shared by every class, not a [C] vector. Calibration
    measured that per-class thresholds overfit badly (3.8x worse on held-out data, with a ceiling of
    only +0.03pp even fitted perfectly), so there is nothing to gain from carrying C copies.

    Two output modes (selected by ``no_argmin``):

    - ``no_argmin=False`` (default) — IN-GRAPH argmin: the nearest-prototype selection and the
      reject decision are baked into the graph, so the model emits scalars directly. The scalar
      threshold drops the ``thresh[best_idx]`` gather this used to need, but ``best_dist =
      dists[best_idx]`` is still a dynamic-index gather, so the tail remains ``ArgMin → Gather →
      Sub``. A stock pip ncnn wheel supports neither (ncnn has no ``ArgMin`` layer at all; its
      ``ArgMax`` exists but is OFF by default and is a different op), so this path STILL needs the
      custom decision layer — the scalar threshold does not change that.
      Outputs: ``best_class_idx`` (int64 scalar), ``best_dist`` (float32), ``margin`` (float32; >0
      → unknown).

    - ``no_argmin=True`` — CLIENT-SIDE argmin: the graph emits only per-class vectors and the
      argmin + reject move to the client. Every remaining op (conv, L2-normalize, matmul, subtract)
      is universally supported, so it converts and runs on a stock ncnn wheel.
      Outputs: ``dists`` (float32 [1, C]), ``margins`` (float32 [1, C]).
      Client: ``best = argmin(dists); unknown = margins[best] > 0; name = class_names[best]``.
    """

    def __init__(
            self,
            backbone: torch.nn.Module,
            proto_mat: torch.Tensor,
            threshold: float,
            no_argmin: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.no_argmin = no_argmin
        self.register_buffer('proto_mat', proto_mat)   # [C, D], L2-normalized
        self.register_buffer('thresh', torch.tensor(float(threshold)))  # scalar, shared by all C

    def forward(self, x: torch.Tensor) -> tuple:
        feat = self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)  # [1, D]
        feat = F.normalize(feat, dim=1)                     # [1, D] unit vector

        if self.no_argmin:
            # Client-side argmin: emit per-class vectors only (no argmax/gather in the graph).
            sims = feat @ self.proto_mat.t()                # [1, C] cosine similarity
            dists = 1.0 - sims                              # [1, C] cosine distance
            margins = dists - self.thresh                   # [1, C] reject margin (>0 unknown)
            return dists, margins

        # In-graph argmin: nearest prototype is the smallest cosine distance.
        feat = feat.squeeze(0)                              # [D]
        sims = self.proto_mat @ feat                        # [C] cosine similarities
        dists = 1.0 - sims                                  # [C] cosine distances
        best_idx = dists.argmin()
        best_dist = dists[best_idx]                         # cosine distance of the nearest prototype
        margin = best_dist - self.thresh                    # scalar subtract; >0 → unknown
        return best_idx, best_dist, margin


@torch.no_grad()
def export_openset_pt(model, prototypes, class_names, threshold, output_path, img_size=224,
                      no_argmin=False, meta_extra: dict | None = None):
    """Trace the open-set model (_OpenSetWrapper) to a TorchScript .pt for PNNX → ncnn.

    Uses the SAME _OpenSetWrapper as export_openset_onnx, so the ONNX and TorchScript exports are
    the identical model. The scalar reject threshold is baked in. Also writes a sidecar
    <output_path>.meta.json with the class order, the scalar ``threshold``, and the ``no_argmin``
    flag so the predict/load paths know the output format.

    Args:
        no_argmin: if True, export the client-side-argmin variant (per-class ``dists``/``margins``
            vectors, no argmax/gather in the graph); if False, the in-graph argmin scalar outputs.
    """
    import json
    proto_mat = torch.stack([prototypes[c] for c in class_names]).cpu()  # [C, D]
    # trace on CPU so the .pt is portable and PNNX/ncnn-friendly
    net = _OpenSetWrapper(model.cpu(), proto_mat, threshold, no_argmin=no_argmin).eval()
    example = torch.zeros(1, 3, img_size, img_size)
    ts = torch.jit.trace(net, example)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ts.save(output_path)
    meta = {'class_names': list(class_names),
            'threshold': float(threshold),
            'img_size': img_size, 'mean': list(MEAN), 'std': list(STD),
            'resize': RESIZE_FILTER, 'no_argmin': no_argmin}
    meta.update(meta_extra or {})       # threshold provenance, when it was calibrated
    with open(output_path + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    outputs = 'dists, margins (client argmin)' if no_argmin else 'best_class_idx, best_dist, margin'
    print(f'TorchScript open-set model → {output_path}  [outputs: {outputs}]')
    print(f'  meta (classes + scalar threshold) → {output_path}.meta.json')
    print(f'  next: pnnx {output_path} inputshape=[1,3,{img_size},{img_size}]')


@torch.no_grad()
def export_openset_onnx(
        model: torch.nn.Module,
        prototypes: dict,
        class_names: list,
        threshold: float,
        output_path: str,
        img_size: int = 224,
        no_argmin: bool = False,
        meta_extra: dict | None = None,
) -> None:
    """Export the full open-set pipeline as a single ONNX graph.

    The exported model embeds the class prototypes and the scalar threshold as constants,
    so at inference time only the raw image is needed as input.

    Args:
        model: Trained backbone (already on the target device, eval mode).
        prototypes: Dict mapping class name → L2-normalized prototype tensor [D].
        class_names: Ordered list of class names (determines output index mapping).
        threshold: Scalar cosine-distance reject threshold, shared by every class.
        output_path: Destination .onnx file path.
        img_size: Square input size for the exported graph.
        no_argmin: if True, export the client-side-argmin variant (per-class ``dists``/``margins``
            vectors, no argmax/gather in the graph); if False, the in-graph argmin scalar outputs.
    """
    proto_mat = torch.stack([prototypes[c] for c in class_names])  # [C, D]
    device = next(model.parameters()).device
    wrapper = _OpenSetWrapper(model, proto_mat.to(device), threshold, no_argmin=no_argmin).eval()

    output_names = ['dists', 'margins'] if no_argmin else ['best_class_idx', 'best_dist', 'margin']
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=['image'],
        output_names=output_names,
        opset_version=17,
        dynamic_axes={'image': {0: 'batch_size'}},
        dynamo=False,
    )
    import json
    meta = {'class_names': list(class_names),
            'threshold': float(threshold),
            'img_size': img_size, 'mean': list(MEAN), 'std': list(STD),
            'resize': RESIZE_FILTER, 'no_argmin': no_argmin}
    meta.update(meta_extra or {})       # threshold provenance, when it was calibrated
    with open(output_path + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'ONNX exported → {output_path}')
    print(f'  meta (classes + scalar threshold) → {output_path}.meta.json')


@torch.no_grad()
def predict_openset(model, transform, prototypes, class_names, threshold, img_path, device):
    feat = extract_feature(model, transform, img_path, device)   # [1280]
    proto_mat = torch.stack([prototypes[c] for c in class_names])  # [C, 1280]
    dists = 1.0 - (proto_mat @ feat)                              # cosine distance to each class
    best_idx = int(dists.argmin())
    best_dist = float(dists[best_idx])
    nearest = class_names[best_idx]
    # reject when the nearest prototype is farther than the shared threshold
    if best_dist > threshold:
        return 'unknown', best_dist, nearest
    return nearest, best_dist, nearest


@torch.no_grad()
def predict_openset_many(model, transform, prototypes, class_names, threshold, img_paths, device,
                         img_size, batch_size=64, num_workers=8):
    """Batched open-set prediction over many images (DataLoader decode + batched backbone).

    Much faster than calling predict_openset per image: workers decode/transform in parallel and the
    backbone runs on batches. Results are returned in the same order as `img_paths`; an unreadable
    image yields ('unreadable', nan, '').
    """
    proto_mat = torch.stack([prototypes[c] for c in class_names]).to(device)   # [C, D]
    model.eval()
    loader = torch.utils.data.DataLoader(
        _FeatDataset([(p, i) for i, p in enumerate(img_paths)], transform, img_size),
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=(str(device) != 'cpu'), worker_init_fn=_worker_init,
    )   # map-style + default sampler ⇒ order preserved, so results stay aligned with img_paths
    results = []
    with torch.inference_mode():
        for x, cls in loader:
            x = x.to(device, non_blocking=True)
            f = model.forward_head(model.forward_features(x), pre_logits=True)  # [B, D]
            f = F.normalize(f, dim=1)
            dists = 1.0 - (f @ proto_mat.t())                                   # [B, C]
            best = dists.argmin(dim=1)
            for b, ci in enumerate(cls.tolist()):
                if ci < 0:                                     # unreadable (marked by _FeatDataset)
                    results.append(('unreadable', float('nan'), ''))
                    continue
                bi = int(best[b])
                bd = float(dists[b, bi])
                nearest = class_names[bi]
                label = 'unknown' if bd > threshold else nearest
                results.append((label, bd, nearest))
    return results


def load_openset_onnx(onnx_path: str, use_gpu: bool = False) -> ort.InferenceSession:
    """Load an exported open-set ONNX model for inference.

    Args:
        onnx_path: Path to the .onnx file produced by export_openset_onnx.
        use_gpu: prefer CUDAExecutionProvider (falls back to CPU if unavailable). Default CPU,
            which mirrors the ncnn deployment target for verification.

    Returns:
        An onnxruntime InferenceSession ready for inference.
    """
    providers = ['CPUExecutionProvider']
    if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session


def predict_onnx(
        session: ort.InferenceSession,
        transform,
        class_names: list,
        img_path: str,
        no_argmin: bool = False,
) -> tuple:
    """Run open-set inference on a single image using an ONNX session.

    Args:
        session: InferenceSession from load_openset_onnx.
        transform: Same torchvision transform used during training.
        class_names: Ordered class names matching the exported prototype matrix.
        img_path: Path to the input image.
        no_argmin: True if the model was exported with client-side argmin (outputs the per-class
            ``dists``/``margins`` vectors); False for the in-graph scalar outputs.

    Returns:
        Tuple of (label, best_dist, nearest_class) where label is 'unknown'
        when the cosine distance exceeds the embedded threshold.
    """
    x = transform(img_path).unsqueeze(0).numpy()      # [1, 3, H, W] float32
    if no_argmin:
        dists, margins = session.run(['dists', 'margins'], {'image': x})
        dists, margins = dists[0], margins[0]         # [C] (drop batch dim)
        best_idx = int(dists.argmin())                # argmin on the client
        best_dist = float(dists[best_idx])
        nearest = class_names[best_idx]
        label = 'unknown' if float(margins[best_idx]) > 0 else nearest
        return label, best_dist, nearest
    best_idx, best_dist, margin = session.run(
        ['best_class_idx', 'best_dist', 'margin'],
        {'image': x},
    )
    best_idx = int(best_idx)
    best_dist = float(best_dist)
    nearest = class_names[best_idx]
    label = 'unknown' if float(margin) > 0 else nearest
    return label, best_dist, nearest


@torch.no_grad()
def predict_pt(ts_model, transform, class_names, img_path, no_argmin=False, device='cpu'):
    """Run open-set inference on one image via a TorchScript open-set model.

    Returns (label, best_dist, nearest) — 'unknown' when the baked-in per-class margin > 0.
    ``no_argmin`` mirrors the export flag: True → the model returns per-class ``dists``/``margins``
    vectors and the argmin is done here; False → the model returns the scalar decision directly.
    ``device`` must match where ``ts_model`` lives ('cpu' or 'cuda').
    """
    x = transform(img_path).unsqueeze(0).to(device)    # [1, 3, H, W] float32
    if no_argmin:
        dists, margins = ts_model(x)
        dists, margins = dists[0], margins[0]          # [C] (drop batch dim)
        best_idx = int(dists.argmin())                 # argmin on the client
        best_dist = float(dists[best_idx])
        nearest = class_names[best_idx]
        label = 'unknown' if float(margins[best_idx]) > 0 else nearest
        return label, best_dist, nearest
    best_idx, best_dist, margin = ts_model(x)
    best_idx = int(best_idx)
    best_dist = float(best_dist)
    nearest = class_names[best_idx]
    label = 'unknown' if float(margin) > 0 else nearest
    return label, best_dist, nearest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR, type=str,
                        help='dataset root containing train/ (and val/) subfolders')
    parser.add_argument('--img-size', default=None, type=int,
                        help=f'square input size (default: {_DEFAULT_IMG_SIZE}). Ignored with '
                             f'--load-onnx, which takes the size baked in at export from the '
                             f'sidecar .meta.json.')
    parser.add_argument('--class-map', default='', type=str,
                        help='class-map file (one class name per line); its line count sets '
                             'num_classes. Required for the PyTorch/export path (not for --load-onnx).')
    parser.add_argument('--ck', '--checkpoint', dest='checkpoint', default='', type=str)
    parser.add_argument('--image', default='', type=str, help='single image to classify (open-set)')
    parser.add_argument(
        '--export-onnx', nargs='?', const='', default=None, metavar='PATH',
        help='export open-set ONNX model; PATH defaults to <checkpoint>_openset.onnx',
    )
    parser.add_argument(
        '--export-pt', nargs='?', const='', default=None, metavar='PATH',
        help='export TorchScript open-set model (distance-vector output) for PNNX->ncnn; '
             'PATH defaults to <checkpoint>_openset.pt',
    )
    parser.add_argument(
        '--load-onnx', default='', type=str, metavar='PATH',
        help='run inference via ONNX (CPU by default, GPU with --gpu-predict); skips PyTorch model '
             'and prototype building. Class names are read from <PATH>.meta.json written at export.',
    )
    parser.add_argument(
        '--no-argmin', action='store_true', default=False,
        help='export the client-side-argmin variant: the model emits per-class dists/margins '
             'vectors (no argmax/gather in the graph) and the argmin + reject move to the client. '
             'Converts/runs on a stock ncnn wheel. Default keeps the in-graph argmin scalar outputs.',
    )
    parser.add_argument(
        '--min-threshold', default=DEFAULT_MIN_THRESHOLD, type=float, metavar='M',
        help=f'floor on the calibrated threshold (default: {DEFAULT_MIN_THRESHOLD}). A suggestion '
             f'below this is clamped up and warned about, since a tiny threshold rejects genuine '
             f'products (0.02 costs ~10%% false-reject on posmlv) while barely reducing '
             f'false-accepts. Lower it only if you know the low value is real.',
    )
    parser.add_argument(
        '--calibrate', action='store_true', default=False,
        help='report-only: print the false-reject / false-accept sweep and exit WITHOUT exporting. '
             'The same calibration runs on every export, so this is for inspecting the curve.',
    )
    parser.add_argument(
        '--aug', action=argparse.BooleanOptionalAction, default=True,
        help='during --calibrate, probe with npaug-augmented views so the in-distribution spread '
             'reflects real variation instead of near-duplicate video frames (prototypes stay '
             'clean); --no-aug probes with the clean crops and leave-one-image-out instead',
    )
    parser.add_argument('--aug-views', default=2, type=int,
                        help='augmented probe views per image during --calibrate (default: 2)')
    parser.add_argument('--aug-max-per-class', default=100, type=int,
                        help='max images per class for the aug probe pass (default: 100); 0 = use all')
    parser.add_argument('--cpu-aug', action='store_true', default=False,
                        help='use CPU npaug (albumentations) instead of GPU torchvision for the aug probe')
    parser.add_argument('--copy-inliers', action='store_true', default=False,
                        help='copy the good inlier crops (cosine dist<=0.5 to their class prototype) into '
                             '<data-dir>/inlier/<class>/ for training a clean classifier')
    parser.add_argument('--gpu-predict', action='store_true', default=False,
                        help='run prediction/verification on GPU (ONNX CUDA provider, TorchScript + '
                             'in-memory model on cuda). Default CPU, which mirrors the ncnn deployment '
                             'target. Note: only the in-memory PyTorch path decodes in batches; the '
                             'ONNX and TorchScript paths run one image at a time')
    args = parser.parse_args()

    # --img-size defaults lazily so --load-onnx can tell "user asked for N" from "user said nothing"
    # and prefer the exported size without silently overriding an explicit request.
    requested_img_size, args.img_size = args.img_size, args.img_size or _DEFAULT_IMG_SIZE

    # --- ONNX-only inference path ---
    if args.load_onnx:
        import json
        meta_path = args.load_onnx + '.meta.json'
        if not os.path.isfile(meta_path):
            raise SystemExit(
                f'{meta_path} not found. The exporter writes it next to the model; it holds the '
                f'exact class order baked into the model (needed to label predictions).'
            )
        meta = json.load(open(meta_path))
        class_names = meta['class_names']
        no_argmin = meta.get('no_argmin', False)      # output format baked at export time
        # Preprocessing must match what was baked in at export, not the CLI defaults: a wrong
        # img_size fails loudly on shape, but a wrong mean/std fails SILENTLY with bad distances.
        img_size = meta.get('img_size', args.img_size)
        mean, std = meta.get('mean', MEAN), meta.get('std', STD)
        resize = meta.get('resize', RESIZE_FILTER)
        if requested_img_size is not None and requested_img_size != img_size:
            print(f'WARNING — ignoring --img-size {requested_img_size}; the model was exported at '
                  f'{img_size} (from {meta_path})')
        transform = build_transform(img_size, mean=mean, std=std)
        print(f'ONNX model : {args.load_onnx}')
        print(f'Classes    : {len(class_names)} (from {meta_path})')
        print(f'Preprocess : {img_size}x{img_size}, mean={tuple(mean)}, std={tuple(std)}, '
              f'resize={resize} (from {meta_path})')
        print(f'Output     : {"dists/margins (client argmin)" if no_argmin else "scalar (in-graph argmin)"}\n')
        session = load_openset_onnx(args.load_onnx, use_gpu=args.gpu_predict)
        img_paths = [args.image] if args.image else _scan_eval_images(args.data_dir)
        print(f'  prediction via ONNX model')
        n_correct = 0
        for img_path in img_paths:
            label, dist, nearest = predict_onnx(session, transform, class_names, img_path,
                                                no_argmin=no_argmin)
            expected = Path(img_path).parent.name     # class subfolder under val/ or test/
            ok = (label == expected)
            n_correct += ok
            marker = '' if ok else '  ✗'
            print(f'{_short_path(img_path, args.data_dir)}')
            print(f'  expected: {expected}   detected: {label}   '
                  f'nearest: {nearest} (dist={dist:.4f}){marker}')
        if len(img_paths) > 1:
            print(f'\nmatched {n_correct}/{len(img_paths)} '
                  f'({n_correct / len(img_paths) * 100:.1f}%) against the folder label')
        return

    # --- PyTorch path ---
    transform = build_transform(args.img_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not args.checkpoint:
        raise SystemExit('--checkpoint is required (or use --load-onnx for ONNX inference)')
    if not args.class_map:
        raise SystemExit('--class-map is required ')
    checkpoint = args.checkpoint
    print(f'Checkpoint: {checkpoint}')
    print(f'Device: {device}\n')

    class_map_names = _read_class_names(args.class_map)
    num_classes = len(class_map_names)
    print(f'Classes (from {args.class_map}): {num_classes}')
    model = timm.create_model(MODEL_NAME, num_classes=num_classes, pretrained=False)
    load_checkpoint(model, checkpoint, weights_only=False)
    model.eval().to(device)

    # One clean feature pass, shared by calibration and prototype building.
    feats = extract_train_features(model, transform, device, args.data_dir, args.img_size,
                                   class_map_names)

    # The threshold is ALWAYS calibrated -- there is no way to pin one. A hand-set value goes stale
    # silently the moment the checkpoint or the class set changes, and nothing in the artifact would
    # show it. Calibration is seeded, so this stays reproducible.
    calib = calibrate_threshold(
        model, transform, device, args.data_dir, args.img_size, class_map_names,
        aug=args.aug, aug_views=args.aug_views,
        aug_max_per_class=args.aug_max_per_class or 0, cpu_aug=args.cpu_aug, feats=feats,
        min_threshold=args.min_threshold)
    if args.calibrate:                      # report-only mode: never exports
        return
    threshold = calib['suggested']
    # Record how the threshold was reached, so the artifact carries its own evidence rather
    # than that living in someone's shell history.
    meta_extra = {
        'threshold_source': 'calibrated',
        'threshold_frr': float(np.mean(calib['d_pos'] > threshold)),
        'threshold_far': float(np.mean(calib['d_neg'] <= threshold)),
        'threshold_plateau': [round(calib['plateau'][0], 4), round(calib['plateau'][1], 4)],
        'threshold_floor': args.min_threshold,
    }
    print(f'\nCalibrated threshold {threshold} (FRR={meta_extra["threshold_frr"] * 100:.2f}%, '
          f'FAR={meta_extra["threshold_far"] * 100:.2f}%)')

    print('Building class prototypes from training set...')
    prototypes, class_names = build_prototypes(
        model, transform, device, args.data_dir, args.img_size, class_map_names,
        threshold=threshold, copy_inliers=args.copy_inliers, feats=feats,
        min_threshold=args.min_threshold)
    print(f'\nConstant threshold {threshold:.4f} applied to {len(class_names)} classes\n')

    # Placeholder classes are exported like any other (index alignment is the point), so name them
    # in the sidecar -- otherwise a client has no way to tell "this class has no training data" from
    # a real class that simply never matches.
    empty_classes = [c for c in class_names if not bool(prototypes[c].any())]
    if empty_classes:
        meta_extra['empty_classes'] = empty_classes

    onnx_path = pt_path = None
    if args.export_onnx is not None:
        onnx_path = args.export_onnx or os.path.splitext(checkpoint)[0] + '_openset.onnx'
        export_openset_onnx(model, prototypes, class_names, threshold, onnx_path,
                            img_size=args.img_size, no_argmin=args.no_argmin,
                            meta_extra=meta_extra)

    if args.export_pt is not None:
        pt_path = args.export_pt or os.path.splitext(checkpoint)[0] + '_openset.pt'
        export_openset_pt(model, prototypes, class_names, threshold, pt_path,
                          img_size=args.img_size, no_argmin=args.no_argmin,
                          meta_extra=meta_extra)

    if args.image:
        img_paths = [args.image]
    else:
        img_paths = _scan_eval_images(args.data_dir)
        print()

    # Prediction/verification device: CPU by default (mirrors ncnn deployment), GPU with --gpu-predict.
    predict_device = 'cuda' if (args.gpu_predict and torch.cuda.is_available()) else 'cpu'

    # If a model was exported, verify by predicting through THAT model; otherwise use the
    # in-memory PyTorch prototypes. `predict_many(paths)` returns (label, dist, nearest) per path.
    if onnx_path is not None:
        session = load_openset_onnx(onnx_path, use_gpu=args.gpu_predict)
        print(f'Predicting via exported ONNX model: {onnx_path} [{predict_device}]\n')
        predict_many = lambda paths: [
            predict_onnx(session, transform, class_names, p, no_argmin=args.no_argmin) for p in paths]
    elif pt_path is not None:
        ts_model = torch.jit.load(pt_path, map_location=predict_device).eval()
        print(f'Predicting via exported TorchScript model: {pt_path} [{predict_device}]\n')
        predict_many = lambda paths: [
            predict_pt(ts_model, transform, class_names, p, no_argmin=args.no_argmin,
                       device=predict_device) for p in paths]
    else:
        model.to(predict_device)
        print(f'Predicting via in-memory PyTorch prototypes [{predict_device}]\n')
        predict_many = lambda paths: predict_openset_many(
            model, transform, prototypes, class_names, threshold, paths, predict_device,
            args.img_size)

    # Report per image: expected class (val/test subfolder), detected type, and nearest distance.
    n_correct = 0
    for img_path, (label, dist, nearest) in zip(img_paths, predict_many(img_paths)):
        expected = Path(img_path).parent.name         # class subfolder under val/ or test/
        ok = (label == expected)
        n_correct += ok
        marker = '' if ok else '  ✗'
        print(f'{_short_path(img_path, args.data_dir)}')
        print(f'  expected: {expected}   detected: {label}   '
              f'nearest: {nearest} (dist={dist:.4f}){marker}')
    if len(img_paths) > 1:
        print(f'\nmatched {n_correct}/{len(img_paths)} '
              f'({n_correct / len(img_paths) * 100:.1f}%) against the folder label')


def dump_layer():
    model = onnx.load('nptools/model_openset.onnx')
    for node in model.graph.node:
        inputs = ', '.join(node.input)
        outputs = ', '.join(node.output)
        print(f'{node.op_type:20s} [{node.name}]  in=({inputs})  out=({outputs})')

if __name__ == '__main__':
    main()
