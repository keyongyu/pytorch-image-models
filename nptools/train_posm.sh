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
# THE size knob, set by --resize. It is the model input size (forwarded to train.py --img-size) and
# it also fixes the pre-resize target at 2x it: npaug's build_aug_pipeline normalises every sample
# to a 2*img_size square before cropping back, so 2*IMG_SIZE -- not IMG_SIZE -- is the most detail
# the pipeline can use and therefore the right size to store on disk. Storing IMG_SIZE instead would
# just make the pipeline upsample again.
IMG_SIZE=224

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
  --resize N         Model input size. (default: ${IMG_SIZE}) Passed to train.py as --img-size.
                     The split is ALWAYS pre-resized once into <data-dir>_<2N> and trained from
                     there, because npaug normalises every sample to a 2*img_size canvas -- 2N is
                     exactly the detail the pipeline can use. Originals are never modified.
  --test-npaug-dir PATH  DRY-RUN: no training; dump augmented images (grouped by class)
                     to PATH for <=4 epochs to visually test nptools/npaug.py.
  --test-npaug-class CLS  Restrict that dry-run to class folder(s) only (comma-separated).
  --no-arcface       Disable --arcface        (ON by default; needed by nptools/openset.py).
  --no-per-class-acc Disable --per-class-acc  (ON by default).
  -h, --help         Show this help and exit.

--num-classes is derived automatically from the class-map (non-empty line count).
--arcface and --per-class-acc are passed to train.py by DEFAULT; pass them explicitly if you
like (it changes nothing), or use the --no- forms above to turn them off.
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
         # derives the pre-resize target (2x) from it.
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

 # ── Pre-resize the training split, unconditionally ──────────────────────────────────────────────
 # Shrinking 10 MP JPEGs on every epoch cost 10.5ms of the 14.3ms per-sample budget -- more than the
 # whole augmentation -- and it is pure waste: npaug fits every sample to a 2*img_size canvas, so
 # nothing above that survives. Doing it once, offline, also handles the elongated shelf crops that
 # a JPEG-decoder-side shrink cannot (DCT scaling is uniform, so it only reduces when BOTH axes
 # exceed the budget, and 1737x371 never does). Measured: the 16-worker loader goes 941.8 ->
 # 1820.0 img/s, and the split shrinks 1705 MB -> 879 MB.
 #
 # Deliberately AFTER the OUTPUT_DIR / CLASS_MAP / NUM_CLASSES derivations above: those stay
 # anchored to the ORIGINAL data dir, so checkpoints and --resume keep living next to the source
 # dataset instead of migrating into the resized copy. Only DATA_DIR is redirected. Idempotent
 # (mtime-checked), so the cost after the first build is a directory walk.
RESIZE_SIZE=$((2 * IMG_SIZE))
RESIZED_DIR="${DATA_DIR%/}_${RESIZE_SIZE}"
if uv run python -m nptools.resize_dataset \
        --src "${DATA_DIR%/}" --out "${RESIZED_DIR}" --size "${RESIZE_SIZE}"; then
    DATA_DIR="${RESIZED_DIR}"
    echo "training from the resized split: ${DATA_DIR}"
else
    echo "resize_dataset failed -- falling back to the original images at ${DATA_DIR}"
fi

 # Both are store_true in train.py, so "off" means omitting the flag entirely.
ARCFACE_FLAGS=""
[ "$ARCFACE" = "1" ]       && ARCFACE_FLAGS="--arcface"
[ "$PER_CLASS_ACC" = "1" ] && ARCFACE_FLAGS="${ARCFACE_FLAGS} --per-class-acc"

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
     --workers=16 \
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
     ${RESUME_OR_NEW} \
     ${EXTRA_ARGS}
     #--bg-dir="${DATA_DIR}/bg_photos" \



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