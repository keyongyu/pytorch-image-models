#!/bin/sh
 
 # background images for nobg bg-swap augmentation (read by nptools/npaug.py)
export NOBG_BG_DIR="${NOBG_BG_DIR:-/home/keyong/cls6/bg_photos}"
#export AUG_DUMP_DIR=kytest/after_aug

# Keep native math/threading libs single-threaded inside each DataLoader worker; parallelism
# comes from --workers below (avoids thread oversubscription — see npaug.py cv2.setNumThreads).
export OMP_NUM_THREADS=1

# Defaults (override via the script-level flags below).
DATA_DIR=/home/keyong/cls2/code/posmlv
OUTPUT_DIR=""
CLASS_MAP=""
# ON by default: the open-set pipeline (nptools/openset.py) needs ArcFace embeddings to build
# usable prototypes, and per-class accuracy is how look-alike classes get spotted. train.py
# declares both as plain store_true with no --no- counterpart, so switching them off means not
# emitting the flag at all -- hence the wrapper owns it, via --no-arcface / --no-per-class-acc.
ARCFACE=1
PER_CLASS_ACC=1
# THE size knob, set by --resize. It is the model input size (forwarded to train.py --img-size,
# and to openset.py / pnnx in the export hint), and it also fixes the pre-resize target at 2x it:
# npaug's build_aug_pipeline normalises every sample to a 2*img_size square before cropping back,
# so 2*IMG_SIZE -- not IMG_SIZE -- is the most detail the pipeline can use and therefore the right
# size to store on disk. Storing IMG_SIZE instead would just make the pipeline upsample again.
IMG_SIZE=224
# ON by default too, same store_true reasoning: mixed precision is the only lever left on the GPU
# side. After the draft() decode fix the loader delivers ~1870 img/s while the 3080 Ti saturates
# near ~1000, so the GPU -- not the data pipeline -- is what caps the epoch (measured sm 91%).
# fp16 keeps fp32 master weights and loss-scales the gradients, so accuracy is unaffected, and
# it is invisible to export: openset.py rebuilds the model and loads fp32 weights (verified --
# ONNX graph and onnxruntime outputs are bit-identical with or without it).
# If ArcFace's margin logits ever produce NaN losses, pass --amp-dtype bfloat16 through.
AMP=1
# The pre-resize is unconditional: shrinking 10 MP JPEGs on every epoch cost 10.5ms of the 14.3ms
# per-sample budget -- more than the whole augmentation -- and npaug's PIL draft() cannot help the
# elongated shelf crops that dominate this dataset (JPEG DCT scaling is uniform, so it only reduces
# when BOTH axes exceed the budget). Doing it once, offline, is axis-independent. It writes a
# sibling <data-dir>_<2*IMG_SIZE> tree and trains from that; the originals are never touched, and
# re-runs are idempotent (mtime-checked), so the cost after the first launch is a directory walk.

 # Consume script-level flags here (not forwarded to train.py); anything else is
 # collected in EXTRA_ARGS and passed through. Must run BEFORE `set --` below
 # repurposes $@ as the python interpreter prefix. Both "--flag value" and
 # "--flag=value" forms are accepted.
 #   --new              fresh run from the pretrained backbone, ignore last.pth.tar
 #   --data-dir  PATH   dataset root        (default: $DATA_DIR)
 #   --output-dir PATH  output/checkpoints  (default: $DATA_DIR/output)
 #   --class-map PATH   class-map file      (default: $DATA_DIR/class_84.txt)
 #   --resize    N      model input size    (default: 224; split always pre-resized to 2*N)
 #   --no-arcface       disable --arcface        (on by default)
 #   --no-per-class-acc disable --per-class-acc  (on by default)
 #   --no-amp           disable --amp            (on by default)
 usage() {
     cat <<EOF
Usage: sh $0 [options] [-- train.py args...]

Wrapper around train.py for the posm dataset. Script-level options (consumed here,
not forwarded); any other argument is passed straight through to train.py.

Options:
  --new              Fresh run from the pretrained backbone; ignore last.pth.tar.
  --data-dir  PATH   Dataset root.        (default: $DATA_DIR)
  --output-dir PATH  Output/checkpoints.  (default: \$DATA_DIR/output)
  --class-map PATH   Class-map file.      (default: \$DATA_DIR/class_84.txt)
  --resize N         Model input size. (default: ${IMG_SIZE}) Passed to train.py as --img-size,
                     and used in the export commands printed at the end. The split is ALWAYS
                     pre-resized once into <data-dir>_<2N> and trained from there, because npaug
                     normalises every sample to a 2*img_size canvas -- 2N is exactly the detail
                     the pipeline can use. Originals are never modified.
  --test-npaug-dir PATH  DRY-RUN: no training; dump augmented images (grouped by class)
                     to PATH for <=4 epochs to visually test nptools/npaug.py.
  --test-npaug-class CLS  Restrict that dry-run to class folder(s) only (comma-separated).
  --no-arcface       Disable --arcface        (ON by default; needed by nptools/openset.py).
  --no-per-class-acc Disable --per-class-acc  (ON by default).
  --no-amp           Disable --amp (fp16 mixed precision, ON by default; the GPU is the
                     bottleneck once npaug's draft() decode is in place).
  -h, --help         Show this help and exit.

--num-classes is derived automatically from the class-map (non-empty line count).
--workers is derived as 2/3 of the PHYSICAL cores (SMT siblings do not help this aug); override
with WORKERS=N in the environment, or pass --workers=N, since pass-through args come last.
--arcface, --per-class-acc and --amp are passed to train.py by DEFAULT; pass them explicitly if
you like (it changes nothing), or use the --no- forms above to turn them off.
Both "--flag value" and "--flag=value" forms are accepted.

Examples:
  sh $0
  sh $0 --new
  sh $0 --data-dir /path/to/ds --class-map /path/to/ds/classes.txt   # arcface on by default
  sh $0 --no-arcface --no-per-class-acc                              # plain softmax run
  sh $0 --test-npaug-dir /tmp/aug_check --test-npaug-class posm_18   # aug dry-run, posm_18 only
EOF
 }

 EXTRA_ARGS=""
 NEW=0
 TEST_NPAUG_DIR=""
 TEST_NPAUG_CLASS=""
 while [ $# -gt 0 ]; do
     case "$1" in
         -h|--help)      usage; exit 0 ;;
         --new)          NEW=1 ;;
         --data-dir)     DATA_DIR="$2"; shift ;;
         --data-dir=*)   DATA_DIR="${1#*=}" ;;
         --output-dir)   OUTPUT_DIR="$2"; shift ;;
         --output-dir=*) OUTPUT_DIR="${1#*=}" ;;
         --class-map)    CLASS_MAP="$2"; shift ;;
         --class-map=*)  CLASS_MAP="${1#*=}" ;;
         # Consumed, not forwarded verbatim: the wrapper re-emits it as train.py --img-size AND
         # derives the pre-resize target (2x) and the export hint's sizes from it.
         --resize)       IMG_SIZE="$2"; shift ;;
         --resize=*)     IMG_SIZE="${1#*=}" ;;
         --test-npaug-dir)     TEST_NPAUG_DIR="$2"; shift ;;
         --test-npaug-dir=*)   TEST_NPAUG_DIR="${1#*=}" ;;
         --test-npaug-class)   TEST_NPAUG_CLASS="$2"; shift ;;
         --test-npaug-class=*) TEST_NPAUG_CLASS="${1#*=}" ;;
         # Consumed, not forwarded, so passing --arcface explicitly cannot emit it twice.
         # ("--arcface-s"/"--arcface-m" take values and fall through to EXTRA_ARGS below.)
         --arcface)            ARCFACE=1 ;;
         --no-arcface)         ARCFACE=0 ;;
         --per-class-acc)      PER_CLASS_ACC=1 ;;
         --no-per-class-acc)   PER_CLASS_ACC=0 ;;
         # ("--amp-dtype bfloat16" takes a value and falls through to EXTRA_ARGS below.)
         --amp)                AMP=1 ;;
         --no-amp)             AMP=0 ;;
         *)              EXTRA_ARGS="$EXTRA_ARGS $1" ;;
     esac
     shift
 done

 # output-dir and class-map default relative to the (possibly overridden) data dir.
 [ -n "$OUTPUT_DIR" ] || OUTPUT_DIR="${DATA_DIR}/output"
 [ -n "$CLASS_MAP" ] || CLASS_MAP="${DATA_DIR}/classmap.txt"

 # Derive --num-classes from the class-map (count non-empty lines) instead of hardcoding.
NUM_CLASSES=$(grep -cve '^[[:space:]]*$' "${CLASS_MAP}")

 # IMG_SIZE feeds shell arithmetic (2*IMG_SIZE) and a directory name, so reject junk here rather
 # than letting $(( )) treat it as 0 and silently build a <data-dir>_0 tree.
case "$IMG_SIZE" in
    ''|*[!0-9]*) echo "--resize must be a positive integer, got '${IMG_SIZE}'" >&2; exit 2 ;;
esac
[ "$IMG_SIZE" -gt 0 ] || { echo "--resize must be > 0" >&2; exit 2; }

 # Scale DataLoader workers with the box instead of hardcoding: this run is data-bound, not
 # GPU-bound -- npaug.py's per-sample albumentations costs ~25ms of pure CPU, so throughput is
 # (workers / 25ms) until the GPU saturates. 2/3 of the cores leaves headroom for the main
 # process (which feeds the GPU and does its own share of Python work) and for other users.
 #
 # PHYSICAL cores, not nproc: this aug is compute-bound, so SMT siblings do not add throughput.
 # Measured on the 3960X (24C/48T): 16 workers -> 576 img/s, 32 workers -> ~509 img/s, i.e.
 # oversubscribing the real cores made per-worker cost rise from 28ms to 63ms per image.
 # Override for a quick experiment with `WORKERS=22 sh nptools/train_posm.sh ...`.
if [ -z "${WORKERS}" ]; then
    # -p lists one row per logical CPU; distinct (core,socket) pairs are the physical cores.
    CORES=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
    [ "${CORES:-0}" -gt 0 ] 2>/dev/null || CORES=$(nproc 2>/dev/null || echo 8)
    WORKERS=$(( CORES * 2 / 3 ))
    [ "$WORKERS" -ge 1 ] || WORKERS=1
fi

 # All three are store_true in train.py, so "off" means omitting the flag entirely.
ARCFACE_FLAGS=""
[ "$ARCFACE" = "1" ]       && ARCFACE_FLAGS="--arcface"
[ "$PER_CLASS_ACC" = "1" ] && ARCFACE_FLAGS="${ARCFACE_FLAGS} --per-class-acc"
AMP_FLAGS=""
[ "$AMP" = "1" ]           && AMP_FLAGS="--amp --channels-last"

 # ── Pre-resize the training split (see RESIZE above) ────────────────────────────────────────────
 # Deliberately AFTER the OUTPUT_DIR / CLASS_MAP / NUM_CLASSES derivations above: those must stay
 # anchored to the ORIGINAL data dir, so checkpoints and --resume keep living next to the source
 # dataset instead of migrating into the resized copy on the first run that enables this.
 # Only DATA_DIR (what train.py reads images from) is redirected. Idempotent -- a second launch
 # re-checks mtimes and rewrites nothing, so this costs a directory walk once the tree exists.
ORIG_DATA_DIR="${DATA_DIR}"
RESIZE_SIZE=$((2 * IMG_SIZE))            # npaug's canvas; see IMG_SIZE above
RESIZED_DIR="${DATA_DIR%/}_${RESIZE_SIZE}"
if uv run python -m nptools.resize_dataset \
        --src "${DATA_DIR%/}" --out "${RESIZED_DIR}" --size "${RESIZE_SIZE}"; then
    DATA_DIR="${RESIZED_DIR}"
    echo "training from the resized split: ${DATA_DIR}"
else
    echo "resize_dataset failed -- falling back to the original images at ${DATA_DIR}"
fi

 if [ "${DEBUGPY:-0}" = "1" ]; then
     set -- -m debugpy \
         --listen 127.0.0.1:5678 \
         --wait-for-client \
         train.py
 else
     set -- train.py
 fi


 # CROP MODE (edit this one variable to switch; passed to the command below to avoid '\' edits):
 #   squash -> stretch to square (aspect discarded), fills the frame
 #   border -> aspect-preserving letterbox (keeps object shape/proportions)
CROP_FLAGS="--crop-mode=squash --train-crop-mode=squash"
#CROP_FLAGS="--crop-mode=border"
if [ "$NEW" = "1" ]; then
    # train from scratch: fresh run from the pretrained backbone, don't resume
    RESUME_OR_NEW="--pretrained-path=last.pth.tar --pretrained"
else
    LAST=$(ls -td "${OUTPUT_DIR}"/20*/last.pth.tar 2>/dev/null | head -1)
    if [ -n "$LAST" ]; then
        RESUME_OR_NEW="--resume=${LAST}"
    else
        RESUME_OR_NEW="--pretrained-path=last.pth.tar  --pretrained"
    fi
fi


 # # #"$PYTHON"
 uv run python  "$@" \
 --model=tf_efficientnet_lite0.in1k \
    --lr 0.001 \
    --warmup-epochs 2 \
    --aa=originalr  \
    --color-jitter 0.4 --reprob 0.2 \
    --nobg \
     --workers="${WORKERS}" \
     --img-size="${IMG_SIZE}" \
     --crop-pct=1 --batch-size=48 \
     --validation-batch-size=256 \
     --val-interval=5 \
     ${CROP_FLAGS} \
     --checkpoint-hist=10 \
     --epochs=60 \
     --data-dir="${DATA_DIR}" \
     --class-map="${CLASS_MAP}" \
     --output="${OUTPUT_DIR}" \
     --num-classes="${NUM_CLASSES}" \
     ${TEST_NPAUG_DIR:+--test-npaug-dir="${TEST_NPAUG_DIR}"} \
     ${TEST_NPAUG_CLASS:+--test-npaug-class="${TEST_NPAUG_CLASS}"} \
     ${ARCFACE_FLAGS} \
     ${AMP_FLAGS} \
     ${RESUME_OR_NEW} \
     ${EXTRA_ARGS}
     #--bg-dir="${DATA_DIR}/bg_photos" \
TRAIN_STATUS=$?


 # ── After training: PRINT (do not run) the open-set export command ──────────────────────────────
 # The follow-up step needs three paths that only this script knows (the timestamped run dir, the
 # class-map, and openset.py's data-dir), so reconstructing it by hand is where mistakes happen.
 #
 # openset.py wants the directory that CONTAINS train/, which is NOT necessarily train.py's
 # --data-dir: this wrapper is usually pointed straight at the class-folder root (timm falls back to
 # the root when <data-dir>/train is absent), in which case openset.py wants its PARENT.
 # ORIG_DATA_DIR, not DATA_DIR: prototypes and the calibrated threshold must come from the images a
 # deployed client will actually see, i.e. the originals -- not from a copy pre-shrunk for training.
if [ -d "${ORIG_DATA_DIR}/train" ]; then
    OPENSET_DATA_DIR="${ORIG_DATA_DIR}"
else
    OPENSET_DATA_DIR=$(dirname "${ORIG_DATA_DIR}")
fi
BEST=$(ls -td "${OUTPUT_DIR}"/20*/model_best.pth.tar 2>/dev/null | head -1)

if [ "$TRAIN_STATUS" != "0" ]; then
    echo
    echo "train.py exited with status ${TRAIN_STATUS} -- not suggesting an export."
elif [ -z "$BEST" ]; then
    echo
    echo "No model_best.pth.tar under ${OUTPUT_DIR}/20*/ -- nothing to export."
else
 # --no-argmin keeps the graph free of ArgMin/Gather so a stock ncnn wheel can run it; the client
 # does argmin over the emitted per-class dists. Class order comes from the .meta.json sidecar.
 # --img-size must match what was TRAINED, or prototypes are extracted at one scale and the client
 # feeds another: openset.py defaults to 224 independently of this script, so pass it explicitly.
 # It is baked into the exported graph and recorded in the sidecar as part of the preprocessing
 # contract, which makes a mismatch silent -- accuracy just quietly drops.
 # Both formats go in ONE run: prototype extraction + threshold calibration is the expensive part
 # and is shared, and it guarantees the two artifacts carry the same prototypes and threshold.
cat <<EOF

Next step -- export the open-set model (prototypes + calibrated threshold baked in).
TorchScript (for pnnx -> ncnn) and ONNX in one run:

  uv run --no-sync python nptools/openset.py \\
      --data-dir ${OPENSET_DATA_DIR} \\
      --class-map ${CLASS_MAP} \\
      --checkpoint ${BEST} \\
      --img-size ${IMG_SIZE} \\
      --export-pt $(dirname "${BEST}")/openset.pt \\
      --export-onnx $(dirname "${BEST}")/openset.onnx \\
      --no-argmin

Drop either --export-* to emit only that format. Each artifact gets a .meta.json sidecar
(class order, threshold, preprocessing) that the client must read.

Then convert the TorchScript file to ncnn (writes openset.ncnn.param/.bin next to the .pt,
alongside .pnnx.* intermediates you can delete):

  uv run --no-sync pnnx $(dirname "${BEST}")/openset.pt inputshape=[1,3,${IMG_SIZE},${IMG_SIZE}]

inputshape must match --img-size above: pnnx traces at that shape and constant-folds the
SAME-padding arithmetic against it, so a wrong value bakes wrong padding into the .param.
pnnx is not in pyproject.toml -- 'uv pip install pnnx ncnn' if the command is missing.
EOF
fi



 # uv run python  "$@" \
 # --model=tf_efficientnet_lite0.in1k \
 #     --crop-pct=1 --aa=originalr --batch-size=48 \
 #     --checkpoint-hist=10 --hflip=0 --scale 0.7 1.0 \
 #     --data-dir="${DATA_DIR}/stanford_cars" \
 #     --output="${DATA_DIR}/stanford_cars_output" \
 #     --num-classes=196 \
 #     --class-map="${DATA_DIR}/stanford_cars/class_map.txt" \
 #     --pretrained-path=last.pth.tar \
 #     --pretrained