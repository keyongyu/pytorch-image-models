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
from timm.models import load_checkpoint
from PIL import Image
from torchvision import transforms


_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'tf_efficientnet_lite0.in1k'
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

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


def build_transform(img_size: int, mean=MEAN, std=STD) -> transforms.Compose:
    # matches inference: squash (resize whole image to square) + normalize
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _build_raw_transform(img_size: int) -> transforms.Compose:
    """Resize + ToTensor only (no normalize) — used for the GPU aug pass."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])


class _NpAugTransform:
    """npaug (build_aug_pipeline) augmentation -> normalized tensor, for threshold calibration.

    Used only to widen the in-distribution distance spread so per-class reject thresholds reflect
    real-world variation (lighting/blur/geometry) instead of the optimistic clean-train spread.
    The prototype itself is still built from clean images.
    """

    def __init__(self, img_size: int):
        from nptools.npaug import build_aug_pipeline
        self.aug = build_aug_pipeline(img_size=img_size)     # ends with Resize -> (img_size, img_size)
        self.norm = transforms.Normalize(mean=MEAN, std=STD)

    def __call__(self, pil_rgb):
        out = self.aug(image=np.array(pil_rgb))['image']     # HWC uint8 RGB
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
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    feat = model.forward_head(model.forward_features(x), pre_logits=True)  # [1, 1280]
    return F.normalize(feat, dim=1).squeeze(0).cpu()  # L2-normalized [1280]


_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


class _FeatDataset(torch.utils.data.Dataset):
    """Yield (transformed_image, class_idx) for prototype feature extraction.

    Loading + transform run in DataLoader worker processes (parallel decode). Unreadable images
    return a zero tensor with class_idx = -1 so the batch loop can skip them without crashing.
    """

    def __init__(self, samples, transform, img_size):
        self.samples = samples          # list of (path, class_idx)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cls = self.samples[i]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), cls
        except (IOError, OSError):
            return torch.zeros(3, self.img_size, self.img_size), -1


def _worker_init(worker_id: int) -> None:
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass


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


def build_prototypes(model, transform, device: str, data_dir: str, img_size: int,
                     class_map_names, batch_size: int = 64, num_workers: int = 8,
                     threshold_quantile: float = 0.97, aug: bool = True, aug_views: int = 2,
                     aug_max_per_class: int = 100, cpu_aug: bool = False, copy_inliers: bool = False):
    """Compute an L2-normalized mean feature per class from the training split.

    Only the classes listed in `class_map_names` are used, in that order (folders not in the
    class-map are ignored). A class-map entry whose folder is missing or has no images is warned
    about and skipped. Parallelized: decode across `num_workers` workers, backbone on batches.
    `class_names` (return) stays aligned with the prototype matrix used for export/inference.

    If `copy_inliers` is set, the good inlier crops (cosine dist <= OUTLIER_THR to their class
    prototype) are copied into `<data_dir>/inlier/<class>/` for training a clean classifier.
    """
    t_start = time.time()
    train_root = os.path.join(data_dir, 'train')

    # collect (path, class_idx) following the class-map order (ignore non-map folders)
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
            print(f'{cls:12s}: WARNING — only {len(imgs)} images (<10); prototype/threshold '
                  f'may be unreliable')
        idx = len(class_names)
        class_names.append(cls)
        samples += [(p, idx) for p in imgs]

    if not class_names:
        raise RuntimeError(f'No class-map class had images under {train_root}')

    model.eval()

    def _extract(tf, sample_list=None, repeat: int = 1, gpu_aug: torch.nn.Module | None = None,
                 track_paths: bool = False):
        """Feature-extract samples with transform `tf`, grouped by class index.

        sample_list defaults to all samples. repeat > 1 tiles the list so stochastic
        augmentation produces multiple independent views in a single DataLoader pass.
        gpu_aug: optional _GpuAugBatch module applied on-device before the backbone.
        track_paths: if True, also return pbc (paths by class, parallel to fbc).
        """
        src = sample_list if sample_list is not None else samples
        actual = src * repeat if repeat > 1 else src
        loader = torch.utils.data.DataLoader(
            _FeatDataset(actual, tf, img_size),
            batch_size=batch_size, num_workers=num_workers,
            pin_memory=(str(device) != 'cpu'),
            worker_init_fn=_worker_init,
        )
        fbc = [[] for _ in class_names]
        pbc = [[] for _ in class_names] if track_paths else None
        path_iter = iter(p for p, _ in actual) if track_paths else None
        with torch.inference_mode():
            for x, cls in loader:
                x = x.to(device, non_blocking=True)
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

    # PROTOTYPE from clean images (matches inference preprocessing; keep it on the clean manifold).
    feats_by_class, paths_by_class = _extract(transform, track_paths=True)

    # THRESHOLD distances: augmented views to widen the in-distribution distance spread.
    # GPU mode (default): workers decode+resize only, _GpuAugBatch runs on GPU — fast.
    # CPU mode (--cpu-aug): _NpAugTransform (albumentations/npaug) runs in workers — faithful
    #   to training augmentation but slower.
    # Both modes subsample to aug_max_per_class per class for the aug pass.
    if aug:
        import random as _random
        by_class: list[list] = [[] for _ in class_names]
        for p, ci in samples:
            by_class[ci].append((p, ci))
        rng = _random.Random(0)
        aug_sample_list: list = []
        for cls_samples in by_class:
            cap = aug_max_per_class if aug_max_per_class > 0 else len(cls_samples)
            chosen = rng.sample(cls_samples, min(cap, len(cls_samples)))
            aug_sample_list.extend(chosen)
        n_aug = len(aug_sample_list) * max(1, aug_views)
        backend = 'CPU npaug' if cpu_aug else 'GPU torchvision'
        print(f'aug threshold pass: {len(aug_sample_list)} images × {aug_views} views = {n_aug} total '
              f'(capped at {aug_max_per_class}/class, {backend})')
        if cpu_aug:
            aug_tf = _NpAugTransform(img_size)
            thr_feats_by_class = _extract(aug_tf, sample_list=aug_sample_list,
                                          repeat=max(1, aug_views))
        else:
            gpu_aug = _GpuAugBatch(img_size).to(device).eval()
            raw_tf = _build_raw_transform(img_size)
            thr_feats_by_class = _extract(raw_tf, sample_list=aug_sample_list,
                                          repeat=max(1, aug_views), gpu_aug=gpu_aug)
    else:
        thr_feats_by_class = feats_by_class

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

    prototypes, thresholds = {}, {}
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
        thr_feats = torch.stack(thr_feats_by_class[i]) if thr_feats_by_class[i] else feats
        # Also filter aug outliers against the clean prototype
        thr_dists = 1.0 - (thr_feats @ proto)
        thr_feats = thr_feats[thr_dists <= OUTLIER_THR]
        if len(thr_feats) == 0:
            thr_feats = feats
        dists = 1.0 - (thr_feats @ proto)                    # distances used for the threshold
        # PER-CLASS threshold: this class's own distance spread, not a single global value
        # (a global threshold is dominated by diffuse catch-all classes like 'others').
        thr = float(torch.quantile(dists, threshold_quantile))
        thresholds[cls] = thr
        print(f'{cls:12s}: {len(feats):4d} imgs, mean={dists.mean():.4f}, '
              f'max={dists.max():.4f}, thr={thr:.4f}')

    # keep class_names aligned with classes that produced a prototype
    class_names = [c for c in class_names if c in prototypes]
    if outlier_rows:
        csv_path = outlier_root / 'outlier.txt'
        _write_aligned_csv(
            csv_path, ('classtype/file', 'bad_dist', 'nearest_type', 'nearest_dist'), outlier_rows)
        print(f'outlier list: {len(outlier_rows)} rows → {_short_path(csv_path, data_dir)}')
    if copy_inliers:
        print(f'inliers copied: {n_inliers_copied} images → {_short_path(inlier_root, data_dir)}')
        if inlier_rows:
            _write_aligned_csv(inlier_root / 'inlier.txt', ('classtype/file', 'distance'), inlier_rows)
    print(f'build_prototypes: {len(class_names)} classes, {len(samples)} images '
          f'in {time.time() - t_start:.1f}s')
    return prototypes, class_names, thresholds


class _OpenSetWrapper(torch.nn.Module):
    """Backbone + prototype constants + cosine-distance head, ready for ONNX/NCNN export.

    Uses a PER-CLASS threshold vector (each class judged against its own distance spread).

    Two output modes (selected by ``no_argmin``):

    - ``no_argmin=False`` (default) — IN-GRAPH argmin: the nearest-prototype selection and
      the reject decision are baked into the graph, so the model emits scalars directly. This needs
      the ncnn runtime to support ArgMin + the dynamic-index Crop (gather); a stock pip ncnn wheel
      does not (ncnn has no ``ArgMin`` layer at all; its ``ArgMax`` layer exists but is OFF by
      default and is a different op), so deploy with a custom layer.
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
            thresh_vec: torch.Tensor,
            no_argmin: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.no_argmin = no_argmin
        self.register_buffer('proto_mat', proto_mat)   # [C, D], L2-normalized
        self.register_buffer('thresh', thresh_vec.to(torch.float32))  # [C] per-class thresholds

    def forward(self, x: torch.Tensor) -> tuple:
        feat = self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)  # [1, D]
        feat = F.normalize(feat, dim=1)                     # [1, D] unit vector

        if self.no_argmin:
            # Client-side argmin: emit per-class vectors only (no argmax/gather in the graph).
            sims = feat @ self.proto_mat.t()                # [1, C] cosine similarity
            dists = 1.0 - sims                              # [1, C] cosine distance
            margins = dists - self.thresh                   # [1, C] per-class reject margin (>0 unknown)
            return dists, margins

        # In-graph argmin: nearest prototype is the smallest cosine distance.
        feat = feat.squeeze(0)                              # [D]
        sims = self.proto_mat @ feat                        # [C] cosine similarities
        dists = 1.0 - sims                                  # [C] cosine distances
        best_idx = dists.argmin()
        best_dist = dists[best_idx]                         # cosine distance of the nearest prototype
        margin = best_dist - self.thresh[best_idx]           # per-class; >0 → unknown, <0 → known
        return best_idx, best_dist, margin


@torch.no_grad()
def export_openset_pt(model, prototypes, class_names, thresholds, output_path, img_size=224,
                      no_argmin=False):
    """Trace the open-set model (_OpenSetWrapper) to a TorchScript .pt for PNNX → ncnn.

    Uses the SAME _OpenSetWrapper as export_openset_onnx, so the ONNX and TorchScript exports are
    the identical model. Per-class thresholds are baked in. Also writes a sidecar
    <output_path>.meta.json with the class order, per-class thresholds (index→name), and the
    ``no_argmin`` flag so the predict/load paths know the output format.

    Args:
        no_argmin: if True, export the client-side-argmin variant (per-class ``dists``/``margins``
            vectors, no argmax/gather in the graph); if False, the in-graph argmin scalar outputs.
    """
    import json
    proto_mat = torch.stack([prototypes[c] for c in class_names]).cpu()  # [C, D]
    thresh_vec = torch.tensor([thresholds[c] for c in class_names])      # [C]
    # trace on CPU so the .pt is portable and PNNX/ncnn-friendly
    net = _OpenSetWrapper(model.cpu(), proto_mat, thresh_vec, no_argmin=no_argmin).eval()
    example = torch.zeros(1, 3, img_size, img_size)
    ts = torch.jit.trace(net, example)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ts.save(output_path)
    meta = {'class_names': list(class_names),
            'thresholds': [float(thresholds[c]) for c in class_names],
            'img_size': img_size, 'mean': list(MEAN), 'std': list(STD), 'no_argmin': no_argmin}
    with open(output_path + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    outputs = 'dists, margins (client argmin)' if no_argmin else 'best_class_idx, best_dist, margin'
    print(f'TorchScript open-set model → {output_path}  [outputs: {outputs}]')
    print(f'  meta (classes + per-class thresholds) → {output_path}.meta.json')
    print(f'  next: pnnx {output_path} inputshape=[1,3,{img_size},{img_size}]')


@torch.no_grad()
def export_openset_onnx(
        model: torch.nn.Module,
        prototypes: dict,
        class_names: list,
        thresholds: dict,
        output_path: str,
        img_size: int = 224,
        no_argmin: bool = False,
) -> None:
    """Export the full open-set pipeline as a single ONNX graph.

    The exported model embeds the class prototypes and per-class thresholds as constants,
    so at inference time only the raw image is needed as input.

    Args:
        model: Trained backbone (already on the target device, eval mode).
        prototypes: Dict mapping class name → L2-normalized prototype tensor [D].
        class_names: Ordered list of class names (determines output index mapping).
        thresholds: Dict mapping class name → per-class cosine-distance reject threshold.
        output_path: Destination .onnx file path.
        img_size: Square input size for the exported graph.
        no_argmin: if True, export the client-side-argmin variant (per-class ``dists``/``margins``
            vectors, no argmax/gather in the graph); if False, the in-graph argmin scalar outputs.
    """
    proto_mat = torch.stack([prototypes[c] for c in class_names])  # [C, D]
    device = next(model.parameters()).device
    thresh_vec = torch.tensor([thresholds[c] for c in class_names], device=device)  # [C]

    wrapper = _OpenSetWrapper(model, proto_mat.to(device), thresh_vec, no_argmin=no_argmin).eval()

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
            'thresholds': [float(thresholds[c]) for c in class_names],
            'img_size': img_size, 'mean': list(MEAN), 'std': list(STD), 'no_argmin': no_argmin}
    with open(output_path + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'ONNX exported → {output_path}')
    print(f'  meta (classes + per-class thresholds) → {output_path}.meta.json')


@torch.no_grad()
def predict_openset(model, transform, prototypes, class_names, thresholds, img_path, device):
    feat = extract_feature(model, transform, img_path, device)   # [1280]
    proto_mat = torch.stack([prototypes[c] for c in class_names])  # [C, 1280]
    dists = 1.0 - (proto_mat @ feat)                              # cosine distance to each class
    best_idx = int(dists.argmin())
    best_dist = float(dists[best_idx])
    nearest = class_names[best_idx]
    # reject if the nearest distance exceeds THAT class's own threshold
    if best_dist > thresholds[nearest]:
        return 'unknown', best_dist, nearest
    return nearest, best_dist, nearest


@torch.no_grad()
def predict_openset_many(model, transform, prototypes, class_names, thresholds, img_paths, device,
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
                label = 'unknown' if bd > thresholds[nearest] else nearest
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
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).numpy()           # [1, 3, H, W] float32
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
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)         # [1, 3, H, W] float32
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
        '--quantile', default=0.97, type=float, metavar='Q',
        help='reject threshold quantile (default: 0.97)',
    )
    parser.add_argument(
        '--aug', action=argparse.BooleanOptionalAction, default=True,
        help='use npaug augmentation to calibrate thresholds (prototype stays clean); --no-aug to disable',
    )
    parser.add_argument('--aug-views', default=2, type=int, help='augmented views per image for thresholds (default: 2)')
    parser.add_argument('--aug-max-per-class', default=100, type=int,
                        help='max images per class for aug threshold pass (default: 100); 0 = use all')
    parser.add_argument('--cpu-aug', action='store_true', default=False,
                        help='use CPU npaug (albumentations) instead of GPU torchvision aug for thresholds')
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
        if requested_img_size is not None and requested_img_size != img_size:
            print(f'WARNING — ignoring --img-size {requested_img_size}; the model was exported at '
                  f'{img_size} (from {meta_path})')
        transform = build_transform(img_size, mean=mean, std=std)
        print(f'ONNX model : {args.load_onnx}')
        print(f'Classes    : {len(class_names)} (from {meta_path})')
        print(f'Preprocess : {img_size}x{img_size}, mean={tuple(mean)}, std={tuple(std)} (from {meta_path})')
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

    print('Building class prototypes from training set...')
    prototypes, class_names, thresholds = build_prototypes(
        model, transform, device, args.data_dir, args.img_size, class_map_names,
        threshold_quantile=args.quantile, aug=args.aug, aug_views=args.aug_views,
        aug_max_per_class=args.aug_max_per_class or 0, cpu_aug=args.cpu_aug,
        copy_inliers=args.copy_inliers)
    print(f'\nPer-class thresholds computed ({len(class_names)} classes, quantile={args.quantile}, '
          f'aug={args.aug})\n')

    onnx_path = pt_path = None
    if args.export_onnx is not None:
        onnx_path = args.export_onnx or os.path.splitext(checkpoint)[0] + '_openset.onnx'
        export_openset_onnx(model, prototypes, class_names, thresholds, onnx_path,
                            img_size=args.img_size, no_argmin=args.no_argmin)

    if args.export_pt is not None:
        pt_path = args.export_pt or os.path.splitext(checkpoint)[0] + '_openset.pt'
        export_openset_pt(model, prototypes, class_names, thresholds, pt_path,
                          img_size=args.img_size, no_argmin=args.no_argmin)

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
            model, transform, prototypes, class_names, thresholds, paths, predict_device,
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
