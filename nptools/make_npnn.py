"""Pack a box detector and an open-set classifier into one two-section npnn file.

An npnn ("Newspage ncnn") file is a text header terminated by a ``Content:`` line, followed by a
raw ncnn payload -- the ``.param`` text with the ``.bin`` concatenated straight onto it. No length
prefix and no separator: ncnn's param parser reads the layer count from line 2 and stops after that
many lines, so the reader knows where the weights begin. (Note the detector files are named ``.pb``
for historical reasons; there is no TensorFlow GraphDef involved.)

The combined layout (see nptools/npnn_openset_format.txt) chains two of those sections:

    Version: ...                  <- the detector's own header, kept verbatim
    ...
    Content: OpenSet              <- marker announces that a second section follows
    <detector param+bin>
    <newline>                     <- so Model: starts on its own line, not glued to the
                                     detector's last weight byte
    Model: OpenSet
    SkuNum: <classes>
    InputWidth: <w>
    InputHeight: <h>
    Content:
    <open-set param+bin>

The detector header is copied as-is, because its StepScale/AspectRatio/Input* fields describe the
weights that follow it -- rewriting them from the format document (which is a template, with values
from a different detector) would leave the header describing a model that is not there.

Usage:
    python -m nptools.make_npnn \\
        --box   nptools/pbfiles/NPBox_20260816.pb \\
        --param <run>/openset.ncnn.param \\
        --bin   <run>/openset.ncnn.bin \\
        --meta  <run>/openset.pt.meta.json \\
        --out   <run>/NPBox_openset.npnn
"""
import argparse
import json
import os
import sys

_CONTENT = b'Content:'
_NCNN_MAGIC = b'7767517'


def _split_npnn(blob: bytes) -> tuple:
    """Split an npnn file into (header_lines_without_content, content_suffix, payload).

    ``content_suffix`` is whatever followed ``Content:`` on its line -- empty for a plain
    single-section file, ``OpenSet`` for one that already carries a classifier.
    """
    i = blob.find(_CONTENT)
    if i < 0:
        raise ValueError('no "Content:" line: not an npnn file')
    line_end = blob.index(b'\n', i)
    suffix = blob[i + len(_CONTENT):line_end].strip()
    return blob[:i], suffix, blob[line_end + 1:]


def _param_text_len(payload: bytes) -> int:
    """Byte length of the ncnn .param text at the head of `payload`, validating it as we go."""
    if not payload.startswith(_NCNN_MAGIC):
        raise ValueError(f'payload does not start with the ncnn magic {_NCNN_MAGIC.decode()}')
    nl1 = payload.index(b'\n')
    nl2 = payload.index(b'\n', nl1 + 1)
    layers, _blobs = (int(x) for x in payload[nl1 + 1:nl2].split())
    pos = nl2 + 1
    for _ in range(layers):
        pos = payload.index(b'\n', pos) + 1        # IndexError/ValueError if the text is truncated
    return pos


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--box', required=True, help='detector npnn file (e.g. NPBox_*.pb)')
    parser.add_argument('--param', required=True, help='open-set openset.ncnn.param')
    parser.add_argument('--bin', required=True, dest='bin_', help='open-set openset.ncnn.bin')
    parser.add_argument('--meta', default='', help='openset.pt.meta.json; supplies SkuNum and the '
                                                   'input size unless overridden below')
    parser.add_argument('--sku-num', type=int, default=None, help='override the class count')
    parser.add_argument('--input-size', type=int, default=None,
                        help='override the square input size')
    parser.add_argument('--out', required=True, help='destination .npnn')
    args = parser.parse_args()

    box = open(args.box, 'rb').read()
    header, suffix, box_payload = _split_npnn(box)
    if suffix or b'\nModel: OpenSet\n' in box:
        sys.exit(f'--box already has a "Content: {suffix.decode()}" section; pass the original '
                 f'single-section detector file instead of a combined one')
    _param_text_len(box_payload)          # validate the detector payload before building on it

    # SkuNum / input size describe the OPEN-SET model, so they come from its own sidecar -- the
    # detector header above keeps its own (different) Input* values for its own weights.
    sku_num, input_size = args.sku_num, args.input_size
    if args.meta:
        meta = json.load(open(args.meta))
        sku_num = sku_num if sku_num is not None else len(meta['class_names'])
        input_size = input_size if input_size is not None else int(meta['img_size'])
    if sku_num is None or input_size is None:
        sys.exit('need --meta, or both --sku-num and --input-size')

    param = open(args.param, 'rb').read()
    if not param.endswith(b'\n'):
        param += b'\n'                    # the .bin must start on its own byte boundary
    binw = open(args.bin_, 'rb').read()
    os_payload = param + binw
    _param_text_len(os_payload)           # same validation for what we are about to append

    out = (header + _CONTENT + b' OpenSet\n' + box_payload
           # Leading \n: the detector's .bin ends on an arbitrary byte, so without it the key
           # would continue that byte's line and no line-oriented reader would find it.
           + b'\nModel: OpenSet\n'
           + f'SkuNum: {sku_num}\n'.encode()
           + f'InputWidth: {input_size}\n'.encode()
           + f'InputHeight: {input_size}\n'.encode()
           + _CONTENT + b'\n' + os_payload)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        f.write(out)

    print(f'npnn written -> {args.out}  ({len(out) / 1e6:.2f} MB)')
    print(f'  [1] detector : header {len(header)} B + payload {len(box_payload) / 1e6:.2f} MB '
          f'(from {os.path.basename(args.box)})')
    print(f'  [2] open-set : SkuNum={sku_num} {input_size}x{input_size}, '
          f'param {len(param) / 1e3:.1f} kB + bin {len(binw) / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
