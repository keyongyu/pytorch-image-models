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

 # Consume script-level flags here (not forwarded to train.py); anything else is
 # collected in EXTRA_ARGS and passed through. Must run BEFORE `set --` below
 # repurposes $@ as the python interpreter prefix. Both "--flag value" and
 # "--flag=value" forms are accepted.
 #   --new              fresh run from the pretrained backbone, ignore last.pth.tar
 #   --data-dir  PATH   dataset root        (default: $DATA_DIR)
 #   --output-dir PATH  output/checkpoints  (default: $DATA_DIR/output)
 #   --class-map PATH   class-map file      (default: $DATA_DIR/class_84.txt)
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