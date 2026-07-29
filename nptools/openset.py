"""Open-set recognition on top of a trained closed-set classifier.

Uses the pre-logits feature (pooled vector before the FC layer) and cosine distance
to per-class mean prototypes. A test image whose nearest prototype is farther than a
threshold is rejected as 'unknown'.
"""
import os
import glob
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


def build_prototypes(model, transform, device: str, data_dir: str, img_size: int,
                     class_map_names, batch_size: int = 64, num_workers: int = 8,
                     threshold_quantile: float = 0.95):
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
                if p.lower().endswith(_IMG_EXTS)]
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

    loader = torch.utils.data.DataLoader(
        _FeatDataset(samples, transform, img_size),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(str(device) != 'cpu'),
    )

    # batched feature extraction, grouped by class
    feats_by_class = [[] for _ in class_names]
    model.eval()
    with torch.inference_mode():
        for x, cls in loader:
            x = x.to(device, non_blocking=True)
            f = model.forward_head(model.forward_features(x), pre_logits=True)  # [B, 1280]
            f = F.normalize(f, dim=1).cpu()
            for fi, ci in zip(f, cls.tolist()):
                if ci >= 0:                                   # skip unreadable (ci == -1)
                    feats_by_class[ci].append(fi)

    prototypes, thresholds = {}, {}
    for i, cls in enumerate(class_names):
        if not feats_by_class[i]:
            continue
        feats = torch.stack(feats_by_class[i])               # [N, 1280]
        proto = F.normalize(feats.mean(dim=0), dim=0)        # normalized class mean
        prototypes[cls] = proto
        dists = 1.0 - (feats @ proto)                        # in-distribution cosine distances
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

    Outputs:
        best_class_idx  – int64 scalar: index into class_names
        best_dist       – float32 scalar: cosine distance to nearest prototype
        margin          – float32 scalar: best_dist - threshold[best_class]
                          negative → known class, positive → unknown
    """

    def __init__(self, backbone: torch.nn.Module, proto_mat: torch.Tensor, thresh_vec: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('proto_mat', proto_mat)   # [C, D], L2-normalized
        self.register_buffer('thresh', thresh_vec.to(torch.float32))  # [C] per-class thresholds

    def forward(self, x: torch.Tensor) -> tuple:
        feat = self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)
        feat = F.normalize(feat, dim=1).squeeze(0)          # [D]
        dists = 1.0 - (self.proto_mat @ feat)               # [C] cosine distances
        best_idx = dists.argmin()
        best_dist = dists[best_idx]
        margin = best_dist - self.thresh[best_idx]           # per-class; >0 → unknown, <0 → known
        return best_idx, best_dist, margin


@torch.no_grad()
def export_openset_pt(model, prototypes, class_names, thresholds, output_path, img_size=224):
    """Trace the open-set model (_OpenSetWrapper) to a TorchScript .pt for PNNX → ncnn.

    Uses the SAME _OpenSetWrapper as export_openset_onnx, so the ONNX and TorchScript exports are
    the identical model. Per-class thresholds are baked in. Also writes a sidecar
    <output_path>.meta.json with the class order and per-class thresholds (index→name).
    """
    import json
    proto_mat = torch.stack([prototypes[c] for c in class_names]).cpu()  # [C, D]
    thresh_vec = torch.tensor([thresholds[c] for c in class_names])      # [C]
    # trace on CPU so the .pt is portable and PNNX/ncnn-friendly
    net = _OpenSetWrapper(model.cpu(), proto_mat, thresh_vec).eval()
    example = torch.zeros(1, 3, img_size, img_size)
    ts = torch.jit.trace(net, example)
    ts.save(output_path)
    meta = {'class_names': list(class_names),
            'thresholds': [float(thresholds[c]) for c in class_names],
            'img_size': img_size, 'mean': list(MEAN), 'std': list(STD)}
    with open(output_path + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'TorchScript open-set model → {output_path}')
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
    """
    proto_mat = torch.stack([prototypes[c] for c in class_names])  # [C, D]
    device = next(model.parameters()).device
    thresh_vec = torch.tensor([thresholds[c] for c in class_names], device=device)  # [C]

    wrapper = _OpenSetWrapper(model, proto_mat.to(device), thresh_vec).eval()

    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=['image'],
        output_names=['best_class_idx', 'best_dist', 'margin'],
        opset_version=17,
        dynamic_axes={'image': {0: 'batch_size'}},
        dynamo=False,
    )
    import json
    meta = {'class_names': list(class_names),
            'thresholds': [float(thresholds[c]) for c in class_names],
            'img_size': img_size, 'mean': list(MEAN), 'std': list(STD)}
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
) -> tuple:
    """Run open-set inference on a single image using an ONNX session.

    Args:
        session: InferenceSession from load_openset_onnx.
        transform: Same torchvision transform used during training.
        class_names: Ordered class names matching the exported prototype matrix.
        img_path: Path to the input image.

    Returns:
        Tuple of (label, best_dist, nearest_class) where label is 'unknown'
        when the cosine distance exceeds the embedded threshold.
    """
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).numpy()           # [1, 3, H, W] float32
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
def predict_pt(ts_model, transform, class_names, img_path):
    """Run open-set inference on one image via a TorchScript open-set model (CPU).

    Returns (label, best_dist, nearest) — 'unknown' when the baked-in per-class margin > 0.
    """
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0)                    # [1, 3, H, W] float32
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
    parser.add_argument('--checkpoint', default='', type=str)
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
        class_names = json.load(open(meta_path))['class_names']
        print(f'ONNX model : {args.load_onnx}')
        print(f'Classes    : {len(class_names)} (from {meta_path})\n')
        session = load_openset_onnx(args.load_onnx)
        img_paths = [args.image] if args.image else _scan_eval_images(args.data_dir)
        print(f'  prediction via ONNX model')
        for img_path in img_paths:
            label, dist, nearest = predict_onnx(session, transform, class_names, img_path)
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
        model, transform, device, args.data_dir, args.img_size, class_map_names)
    print(f'\nPer-class thresholds computed ({len(class_names)} classes)\n')

    onnx_path = pt_path = None
    if args.export_onnx is not None:
        onnx_path = args.export_onnx or os.path.splitext(checkpoint)[0] + '_openset.onnx'
        export_openset_onnx(model, prototypes, class_names, thresholds, onnx_path,
                            img_size=args.img_size)

    if args.export_pt is not None:
        pt_path = args.export_pt or os.path.splitext(checkpoint)[0] + '_openset.pt'
        export_openset_pt(model, prototypes, class_names, thresholds, pt_path,
                          img_size=args.img_size)

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
        predict = lambda p: predict_onnx(session, transform, class_names, p)
    elif pt_path is not None:
        ts_model = torch.jit.load(pt_path).eval()
        print(f'Predicting via exported TorchScript model: {pt_path}\n')
        predict = lambda p: predict_pt(ts_model, transform, class_names, p)
    else:
        print('Predicting via in-memory PyTorch prototypes\n')
        predict = lambda p: predict_openset(
            model, transform, prototypes, class_names, thresholds, p, device)

    for img_path in img_paths:
        label, dist, nearest = predict(img_path)
        print(f'{img_path}')
        print(f'  → prediction: {label}  (nearest: {nearest}, dist={dist:.4f})')

def dump_layer():
    model = onnx.load('nptools/model_openset.onnx')
    for node in model.graph.node:
        inputs = ', '.join(node.input)
        outputs = ', '.join(node.output)
        print(f'{node.op_type:20s} [{node.name}]  in=({inputs})  out=({outputs})')

if __name__ == '__main__':
    main()
