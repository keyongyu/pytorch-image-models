"""Open-set recognition on top of a trained closed-set classifier.

Uses the pre-logits feature (pooled vector before the FC layer) and cosine distance
to per-class mean prototypes. A test image whose nearest prototype is farther than a
threshold is rejected as 'unknown'.
"""
import os
import glob
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
#DATA_DIR = os.path.join(_HERE, '..', 'pepsi_merge')
DATA_DIR = os.path.join(_HERE, '..', 'posmlv')
OUTPUT_DIR = os.path.join(_HERE, '..', 'posmlv', 'output')
MODEL_NAME = 'tf_efficientnet_lite0.in1k'
#NUM_CLASSES = 3
NUM_CLASSES =84 
IMG_SIZE = 224
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def find_best_checkpoint(output_dir: str) -> str:
    runs = sorted(glob.glob(os.path.join(output_dir, '20*')), reverse=True)
    for run in runs:
        best = os.path.join(run, 'model_best.pth.tar')
        if os.path.isfile(best):
            return best
    raise FileNotFoundError(f'No model_best.pth.tar found under {output_dir}')


def build_transform() -> transforms.Compose:
    # matches inference: squash (resize whole image to square) + normalize
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
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

    def __init__(self, samples, transform):
        self.samples = samples          # list of (path, class_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cls = self.samples[i]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), cls
        except (IOError, OSError):
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1


def build_prototypes(model, transform, device: str, batch_size: int = 64, num_workers: int = 8):
    """Compute an L2-normalized mean feature per class from the training split.

    Parallelized: image decode/transform runs across `num_workers` DataLoader processes and the
    backbone runs on batches of `batch_size` (one GPU forward per batch instead of per image).
    Classes with no readable images are skipped; `class_names` stays aligned with the prototype
    matrix used for export/inference.
    """
    train_root = os.path.join(DATA_DIR, 'train')

    # collect (path, class_idx) for classes that actually have images
    class_names, samples = [], []
    for cls in sorted(os.listdir(train_root)):
        cls_dir = os.path.join(train_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        imgs = [p for p in sorted(glob.glob(os.path.join(cls_dir, '*')))
                if p.lower().endswith(_IMG_EXTS)]
        if not imgs:
            print(f'{cls:12s}: 0 imgs — skipped (no prototype)')
            continue
        idx = len(class_names)
        class_names.append(cls)
        samples += [(p, idx) for p in imgs]

    if not class_names:
        raise RuntimeError(f'No class had any images under {train_root}')

    loader = torch.utils.data.DataLoader(
        _FeatDataset(samples, transform),
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

    prototypes, all_train_dists = {}, []
    for i, cls in enumerate(class_names):
        if not feats_by_class[i]:
            continue
        feats = torch.stack(feats_by_class[i])               # [N, 1280]
        proto = F.normalize(feats.mean(dim=0), dim=0)        # normalized class mean
        prototypes[cls] = proto
        dists = 1.0 - (feats @ proto)                        # in-distribution cosine distances
        all_train_dists.append(dists)
        print(f'{cls:12s}: {len(feats):4d} imgs, mean dist={dists.mean():.4f}, max dist={dists.max():.4f}')

    # keep class_names aligned with classes that produced a prototype
    class_names = [c for c in class_names if c in prototypes]
    all_train_dists = torch.cat(all_train_dists)
    threshold = torch.quantile(all_train_dists, 0.95).item()  # 95th pct of in-distribution distances
    return prototypes, class_names, threshold


class _OpenSetWrapper(torch.nn.Module):
    """Backbone + prototype constants + cosine-distance head, ready for ONNX/NCNN export.

    Outputs:
        best_class_idx  – int64 scalar: index into class_names
        best_dist       – float32 scalar: cosine distance to nearest prototype
        margin          – float32 scalar: best_dist - threshold
                          negative → known class, positive → unknown
    """

    def __init__(self, backbone: torch.nn.Module, proto_mat: torch.Tensor, threshold: float):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('proto_mat', proto_mat)   # [C, D], L2-normalized
        self.register_buffer('thresh', torch.tensor(threshold, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple:
        feat = self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)
        feat = F.normalize(feat, dim=1).squeeze(0)          # [D]
        dists = 1.0 - (self.proto_mat @ feat)               # [C] cosine distances
        best_idx = dists.argmin()
        best_dist = dists[best_idx]
        margin = best_dist - self.thresh                     # >0 → unknown, <0 → known
        return best_idx, best_dist, margin


@torch.no_grad()
def export_openset_pt(model, prototypes, class_names, threshold, output_path, img_size=224):
    """Trace the open-set model (_OpenSetWrapper) to a TorchScript .pt for PNNX → ncnn.

    Uses the SAME _OpenSetWrapper as export_openset_onnx, so the ONNX and TorchScript exports are
    the identical model (same forward: backbone → prototype cosine-distance → argmin/dist/margin).
    Also writes a sidecar <output_path>.meta.json with the class order and threshold (index→name).
    """
    import json
    proto_mat = torch.stack([prototypes[c] for c in class_names]).cpu()  # [C, D]
    # trace on CPU so the .pt is portable and PNNX/ncnn-friendly
    net = _OpenSetWrapper(model.cpu(), proto_mat, threshold).eval()
    example = torch.zeros(1, 3, img_size, img_size)
    ts = torch.jit.trace(net, example)
    ts.save(output_path)
    meta = {'class_names': list(class_names), 'threshold': float(threshold),
            'img_size': img_size, 'mean': list(MEAN), 'std': list(STD)}
    with open(output_path + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'TorchScript open-set model → {output_path}')
    print(f'  meta (classes + threshold) → {output_path}.meta.json')
    print(f'  next: pnnx {output_path} inputshape=[1,3,{img_size},{img_size}]')


@torch.no_grad()
def export_openset_onnx(
        model: torch.nn.Module,
        prototypes: dict,
        class_names: list,
        threshold: float,
        output_path: str,
) -> None:
    """Export the full open-set pipeline as a single ONNX graph.

    The exported model embeds the class prototypes and threshold as constants,
    so at inference time only the raw image is needed as input.

    Args:
        model: Trained backbone (already on the target device, eval mode).
        prototypes: Dict mapping class name → L2-normalized prototype tensor [D].
        class_names: Ordered list of class names (determines output index mapping).
        threshold: Cosine-distance threshold for open-set rejection.
        output_path: Destination .onnx file path.
    """
    proto_mat = torch.stack([prototypes[c] for c in class_names])  # [C, D]
    device = next(model.parameters()).device

    wrapper = _OpenSetWrapper(model, proto_mat.to(device), threshold).eval()

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
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
    print(f'ONNX exported → {output_path}')
    print(f'  classes   : {class_names}')
    print(f'  threshold : {threshold:.4f}')


@torch.no_grad()
def predict_openset(model, transform, prototypes, class_names, threshold, img_path, device):
    feat = extract_feature(model, transform, img_path, device)   # [1280]
    proto_mat = torch.stack([prototypes[c] for c in class_names])  # [C, 1280]
    dists = 1.0 - (proto_mat @ feat)                              # cosine distance to each class
    best_idx = int(dists.argmin())
    best_dist = float(dists[best_idx])
    if best_dist > threshold:
        return 'unknown', best_dist, class_names[best_idx]
    return class_names[best_idx], best_dist, class_names[best_idx]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--image', default='', type=str, help='single image to classify (open-set)')
    parser.add_argument('--threshold', default=None, type=float, help='override auto threshold')
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
        help='run inference via ONNX (CPU); skips PyTorch model and prototype building',
    )
    parser.add_argument(
        '--class-names', default='', type=str,
        help='comma-separated class names for --load-onnx (e.g. CRM060109,CRM060118,known)',
    )
    args = parser.parse_args()

    transform = build_transform()

    # --- ONNX-only inference path ---
    if args.load_onnx:
        if not args.class_names:
            # fall back: infer class names from training folder order
            class_names = sorted(os.listdir(os.path.join(DATA_DIR, 'train')))
        else:
            class_names = [c.strip() for c in args.class_names.split(',')]
        print(f'ONNX model : {args.load_onnx}')
        print(f'Classes    : {class_names}\n')
        session = load_openset_onnx(args.load_onnx)
        img_paths = [args.image] if args.image else sorted(
            p for p in glob.glob(os.path.join(DATA_DIR, 'val', '**', '*'), recursive=True)
            if os.path.isfile(p) and p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        )
        if not args.image:
            print(f'Scanning val: {len(img_paths)} images found\n')
        print(f'  prediction via ONNX model')
        for img_path in img_paths:
            label, dist, nearest = predict_onnx(session, transform, class_names, img_path)
            print(f'{img_path}')
            print(f'  → prediction: {label}  (nearest known: {nearest}, dist={dist:.4f})')
        return

    # --- PyTorch path ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = args.checkpoint or find_best_checkpoint(OUTPUT_DIR)
    print(f'Checkpoint: {checkpoint}')
    print(f'Device: {device}\n')

    model = timm.create_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)
    load_checkpoint(model, checkpoint, weights_only=False)
    model.eval().to(device)

    print('Building class prototypes from training set...')
    prototypes, class_names, threshold = build_prototypes(model, transform, device)
    if args.threshold is not None:
        threshold = args.threshold
    print(f'\nOpen-set threshold (cosine dist): {threshold:.4f}\n')

    if args.export_onnx is not None:
        onnx_path = args.export_onnx or os.path.splitext(checkpoint)[0] + '_openset.onnx'
        export_openset_onnx(model, prototypes, class_names, threshold, onnx_path)

    if args.export_pt is not None:
        pt_path = args.export_pt or os.path.splitext(checkpoint)[0] + '_openset.pt'
        export_openset_pt(model, prototypes, class_names, threshold, pt_path, img_size=IMG_SIZE)

    if args.image:
        img_paths = [args.image]
    else:
        val_dir = os.path.join(DATA_DIR, 'val')
        img_paths = sorted(
            p for p in glob.glob(os.path.join(val_dir, '**', '*'), recursive=True)
            if os.path.isfile(p) and p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        )
        print(f'Scanning {val_dir}: {len(img_paths)} images found\n')

    for img_path in img_paths:
        label, dist, nearest = predict_openset(
            model, transform, prototypes, class_names, threshold, img_path, device)
        print(f'{img_path}')
        print(f'  → prediction: {label}  (nearest known: {nearest}, dist={dist:.4f})')

def dump_layer():
    model = onnx.load('kytest/model_openset.onnx')
    for node in model.graph.node:
        inputs = ', '.join(node.input)
        outputs = ', '.join(node.output)
        print(f'{node.op_type:20s} [{node.name}]  in=({inputs})  out=({outputs})')

if __name__ == '__main__':
    main()
