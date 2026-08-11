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


def build_transform(img_size: int) -> transforms.Compose:
    # matches inference: squash (resize whole image to square) + normalize
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
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
        return self.norm(self.aug(x))


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


def build_prototypes(model, transform, device: str, data_dir: str, img_size: int,
                     class_map_names, batch_size: int = 64, num_workers: int = 8,
                     threshold_quantile: float = 0.97, aug: bool = True, aug_views: int = 2,
                     aug_max_per_class: int = 100, cpu_aug: bool = False):
    """Compute an L2-normalized mean feature per class from the training split.

    Only the classes listed in `class_map_names` are used, in that order (folders not in the
    class-map are ignored). A class-map entry whose folder is missing or has no images is warned
    about and skipped. Parallelized: decode across `num_workers` workers, backbone on batches.
    `class_names` (return) stays aligned with the prototype matrix used for export/inference.
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
                  f'(dist>{OUTLIER_THR}) → {outlier_cls_dir}')
            for j, is_out in enumerate(outlier_mask.tolist()):
                if is_out:
                    src = Path(paths[j])
                    dst = outlier_cls_dir / src.name
                    shutil.copy2(src, dst)
                    print(f'               dist={dists0[j]:.4f}  {paths[j]}')
            feats = feats[~outlier_mask]
            if len(feats) == 0:
                print(f'{cls:12s}: WARNING — all images are outliers, class skipped')
                continue

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


def load_openset_onnx(onnx_path: str) -> ort.InferenceSession:
    """Load an exported open-set ONNX model for CPU inference.

    Args:
        onnx_path: Path to the .onnx file produced by export_openset_onnx.

    Returns:
        An onnxruntime InferenceSession ready for inference.
    """
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
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
def predict_pt(ts_model, transform, class_names, img_path, no_argmin=False):
    """Run open-set inference on one image via a TorchScript open-set model (CPU).

    Returns (label, best_dist, nearest) — 'unknown' when the baked-in per-class margin > 0.
    ``no_argmin`` mirrors the export flag: True → the model returns per-class ``dists``/``margins``
    vectors and the argmin is done here; False → the model returns the scalar decision directly.
    """
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0)                    # [1, 3, H, W] float32
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
    parser.add_argument('--img-size', default=_DEFAULT_IMG_SIZE, type=int,
                        help='square input size')
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
        help='run inference via ONNX (CPU); skips PyTorch model and prototype building. '
             'Class names are read from <PATH>.meta.json written at export.',
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
    args = parser.parse_args()

    transform = build_transform(args.img_size)

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
        print(f'ONNX model : {args.load_onnx}')
        print(f'Classes    : {len(class_names)} (from {meta_path})')
        print(f'Output     : {"dists/margins (client argmin)" if no_argmin else "scalar (in-graph argmin)"}\n')
        session = load_openset_onnx(args.load_onnx)
        img_paths = [args.image] if args.image else _scan_eval_images(args.data_dir)
        print(f'  prediction via ONNX model')
        for img_path in img_paths:
            label, dist, nearest = predict_onnx(session, transform, class_names, img_path,
                                                no_argmin=no_argmin)
            print(f'{img_path}')
            print(f'  → prediction: {label}  (nearest known: {nearest}, dist={dist:.4f})')
        return

    # --- PyTorch path ---
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
        aug_max_per_class=args.aug_max_per_class or 0, cpu_aug=args.cpu_aug)
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

    # If a model was exported, verify by predicting through THAT model; otherwise use the
    # in-memory PyTorch prototypes.
    if onnx_path is not None:
        session = load_openset_onnx(onnx_path)
        print(f'Predicting via exported ONNX model: {onnx_path}\n')
        predict = lambda p: predict_onnx(session, transform, class_names, p, no_argmin=args.no_argmin)
    elif pt_path is not None:
        ts_model = torch.jit.load(pt_path).eval()
        print(f'Predicting via exported TorchScript model: {pt_path}\n')
        predict = lambda p: predict_pt(ts_model, transform, class_names, p, no_argmin=args.no_argmin)
    else:
        print('Predicting via in-memory PyTorch prototypes\n')
        predict = lambda p: predict_openset(
            model, transform, prototypes, class_names, thresholds, p, device)

    for img_path in img_paths:
        label, dist, nearest = predict(img_path)
        print(f'{img_path}')
        print(f'  → prediction: {label}  (nearest: {nearest}, dist={dist:.4f})')

    # Predict on outlier folder (outliers flagged during prototype building).
    outlier_root = Path(args.data_dir) / 'outlier'
    if outlier_root.exists():
        outlier_imgs = sorted(
            p for p in outlier_root.rglob('*')
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )
        if outlier_imgs:
            print(f'\n--- outlier predictions ({len(outlier_imgs)} files) ---')
            for p in outlier_imgs:
                true_cls = p.parent.name          # folder name = original class
                label, dist, nearest = predict(str(p))
                marker = '' if label == true_cls else f'  ← expected {true_cls}'
                print(f'{p}')
                print(f'  → {label}  (nearest: {nearest}, dist={dist:.4f}){marker}')

def dump_layer():
    model = onnx.load('nptools/model_openset.onnx')
    for node in model.graph.node:
        inputs = ', '.join(node.input)
        outputs = ', '.join(node.output)
        print(f'{node.op_type:20s} [{node.name}]  in=({inputs})  out=({outputs})')

if __name__ == '__main__':
    main()
