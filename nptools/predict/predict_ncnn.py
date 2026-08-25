"""Classify images with an exported open-set ncnn model — the reference client.

This is what a device-side client has to reproduce, in the smallest form that still gets every
detail of the preprocessing contract right. Those details are where "correct in PyTorch, wrong in
ncnn" comes from, and none of them fail loudly:

  * **BGR**, i.e. cv2.imread's own order — do NOT cvtColor to RGB. The model is trained on BGR.
  * **squash** resize to img_size x img_size, ignoring aspect ratio (training used
    --crop-mode=squash), with **INTER_AREA**: it averages every source pixel, while cv2's default
    INTER_LINEAR reads a fixed 2x2 kernel and aliases badly on a large downscale.
  * mean/std 0.5 on 0..1, which on 0..255 pixels is substract_mean_normalize([127.5]*3, [1/127.5]*3).

Everything above is read from the sidecar .meta.json rather than hardcoded, and asserted where a
mismatch would otherwise be silent.

Assumes the --no-argmin export: the graph emits per-class `dists` and `margins` vectors (blobs
out0/out1) and the decision is made here. That is the variant a stock ncnn wheel can run.

Usage:
    python -m nptools.predict.predict_ncnn --model <run>/openset.ncnn --image photo.jpg
    python -m nptools.predict.predict_ncnn --model <run>/openset.ncnn --dir <data>/val --top 3
"""
import argparse
import glob
import json
import os
import sys

import cv2
import ncnn
import numpy as np

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def load_meta(model_stem: str) -> dict:
    """Read the sidecar written at export time, and check it describes a model we can run."""
    for cand in (f'{model_stem}.pt.meta.json', f'{model_stem}.onnx.meta.json',
                 f'{model_stem}.meta.json'):
        if os.path.isfile(cand):
            meta = json.load(open(cand))
            break
    else:
        sys.exit(f'no sidecar .meta.json next to {model_stem}; it carries the class order and the '
                 f'preprocessing contract, and nothing works without it')
    if not meta.get('no_argmin', False):
        sys.exit('this model was exported WITHOUT --no-argmin, so the decision is inside the graph '
                 '(ArgMin + gather) and a stock ncnn wheel cannot run it. Re-export with '
                 '--no-argmin, or use a custom ncnn layer.')
    if meta.get('channel_order', 'rgb') != 'bgr':
        sys.exit(f"sidecar says channel_order={meta.get('channel_order')!r}; this client feeds BGR. "
                 f"An RGB-era model needs re-exporting (see openset.md).")
    return meta


def preprocess(path: str, meta: dict) -> ncnn.Mat:
    """Decode and normalize one image exactly as the prototypes were built."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)          # BGR, alpha dropped
    if bgr is None:
        raise OSError(f'could not decode image: {path}')
    size = int(meta['img_size'])
    interp = cv2.INTER_AREA if meta.get('resize', 'area') == 'area' else cv2.INTER_LINEAR
    bgr = cv2.resize(bgr, (size, size), interpolation=interp)   # squash, not letterbox
    mat = ncnn.Mat.from_pixels(bgr, ncnn.Mat.PixelType.PIXEL_BGR, size, size)
    # mean/std are expressed on 0..1; ncnn works on 0..255 pixels, hence the 255x rescale:
    #   (x/255 - m) / s  ==  (x - 255m) * (1 / (255s))
    mean = [255.0 * v for v in meta['mean']]
    norm = [1.0 / (255.0 * v) for v in meta['std']]
    mat.substract_mean_normalize(mean, norm)
    return mat


def predict(net: ncnn.Net, mat: ncnn.Mat, meta: dict, top: int = 1) -> tuple:
    """Return (label, best_dist, ranked) — label is 'unknown' when the nearest prototype is too far."""
    with net.create_extractor() as ex:
        ex.input('in0', mat)
        dists = np.array(ex.extract('out0')[1])       # [C] cosine distance to every prototype
        margins = np.array(ex.extract('out1')[1])     # [C] dists - threshold; >0 means reject
    names = meta['class_names']
    order = np.argsort(dists)[:max(1, top)]
    best = int(order[0])
    # margins already carries the threshold, so the client only checks its sign. Comparing
    # dists[best] > meta['threshold'] is equivalent; margins keeps the two in lockstep.
    label = 'unknown' if float(margins[best]) > 0 else names[best]
    ranked = [(names[i], float(dists[i]), float(margins[i])) for i in order]
    return label, float(dists[best]), ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, metavar='STEM',
                        help='path stem of the ncnn pair, e.g. <run>/openset.ncnn '
                             '(reads STEM.param and STEM.bin)')
    parser.add_argument('--image', default='', help='one image to classify')
    parser.add_argument('--dir', default='', help='directory to scan recursively instead')
    parser.add_argument('--top', type=int, default=1, help='also show the N nearest classes')
    args = parser.parse_args()

    stem = args.model[:-len('.param')] if args.model.endswith('.param') else args.model
    stem = stem[:-len('.bin')] if stem.endswith('.bin') else stem
    param, binf = f'{stem}.param', f'{stem}.bin'
    for f in (param, binf):
        if not os.path.isfile(f):
            sys.exit(f'missing {f}')
    # strip the trailing '.ncnn' to find the sidecar, which is named after the exported model
    meta = load_meta(stem[:-len('.ncnn')] if stem.endswith('.ncnn') else stem)

    if args.image:
        paths = [args.image]
    elif args.dir:
        paths = sorted(p for p in glob.glob(os.path.join(args.dir, '**', '*'), recursive=True)
                       if os.path.isfile(p) and p.lower().endswith(_IMG_EXTS))
    else:
        sys.exit('need --image or --dir')

    empty = set(meta.get('empty_classes', []))
    print(f'model      : {param}')
    print(f'preprocess : {meta["img_size"]}x{meta["img_size"]} {meta["channel_order"].upper()}, '
          f'{meta.get("resize", "area")} resize, mean={tuple(meta["mean"])} std={tuple(meta["std"])}')
    print(f'classes    : {len(meta["class_names"])}'
          + (f' ({len(empty)} with no training data)' if empty else '')
          + f'   reject beyond dist {meta["threshold"]:.4f}'
          + (f' ({meta["threshold_deg"]:g} deg)' if 'threshold_deg' in meta else '') + '\n')

    net = ncnn.Net()
    net.load_param(param)
    net.load_model(binf)
    n_unknown = 0
    try:
        for p in paths:
            try:
                label, dist, ranked = predict(net, preprocess(p, meta), meta, args.top)
            except OSError as e:
                print(f'{p}\n  skipped: {e}')
                continue
            n_unknown += label == 'unknown'
            note = '  [no training data for this class]' if label in empty else ''
            print(f'{p}\n  detected: {label}   dist={dist:.4f}{note}')
            if args.top > 1:
                for name, d, m in ranked:
                    print(f'    {name:14s} dist={d:.4f}  margin={m:+.4f}'
                          + ('  <- rejected' if m > 0 else ''))
    finally:
        del net           # release the ncnn allocator before exit
    if len(paths) > 1:
        print(f'\n{len(paths)} images, {n_unknown} rejected as unknown')


if __name__ == '__main__':
    main()
