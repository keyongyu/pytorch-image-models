#!/bin/sh
 
 # Background images for the nobg bg-swap augmentation (read by nptools/npaug.py). train.py fails
 # fast when --nobg is set and this is missing or empty, so the default has to be a path that
 # actually exists -- the previous one did not, and every run from a shell without NOBG_BG_DIR set
 # died before epoch 0. PASCAL VOC2012 works well here: 17k varied indoor/outdoor scenes, which is
 # what the composite needs (general backgrounds, not more product shots).
 #   wget https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
 #   tar -xf VOCtrainval_11-May-2012.tar          # -> VOCdevkit/VOC2012/JPEGImages
export NOBG_BG_DIR="${NOBG_BG_DIR:-/home/keyong/datasets/VOCdevkit/VOC2012/JPEGImages}"
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
# THE size knob, set by --img-size. It is the model input size (forwarded to train.py --img-size,
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
# ON by default: after training, RUN the export script written into the run dir (openset.py export
# + pnnx -> ncnn). The script is always written either way; this only decides whether it is also
# executed. Costs a couple of minutes (the prototype pass and threshold calibration dominate), and
# a training run whose artifacts were never exported is the more common annoyance. --no-ncnn skips.
NCNN=1
# The pre-resize is unconditional: shrinking 10 MP JPEGs on every epoch cost 10.5ms of the 14.3ms
# per-sample budget -- more than the whole augmentation -- and npaug's PIL draft() cannot help the
# elongated shelf crops that dominate this dataset (JPEG DCT scaling is uniform, so it only reduces
# when BOTH axes exceed the budget). Doing it once, offline, is axis-independent. It writes a
# sibling <data-dir>/train_<2*IMG_SIZE> split and trains from that via --train-split; the originals
# are never touched, and re-runs are idempotent (mtime-checked), so the cost after the first
# launch is a directory walk.

 # Consume script-level flags here (not forwarded to train.py); anything else is
 # collected in EXTRA_ARGS and passed through. Must run BEFORE `set --` below
 # repurposes $@ as the python interpreter prefix. Both "--flag value" and
 # "--flag=value" forms are accepted.
 #   --new              fresh run from the pretrained backbone, ignore last.pth.tar
 #   --data-dir  PATH   dataset root        (default: $DATA_DIR)
 #   --output-dir PATH  output/checkpoints  (default: $DATA_DIR/output)
 #   --class-map PATH   class-map file      (default: $DATA_DIR/class_84.txt)
 #   --img-size  N      model input size    (default: 224; split always pre-resized to 2*N)
 #   --no-arcface       disable --arcface        (on by default)
 #   --no-per-class-acc disable --per-class-acc  (on by default)
 #   --no-amp           disable --amp            (on by default)
 #   --no-ncnn          write the export script but do not run it (running is the default)
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
  --img-size N       Model input size. (default: ${IMG_SIZE}) Passed straight to train.py,
                     and used in the export commands printed at the end. The split is ALWAYS
                     pre-resized once into <data-dir>/train_<2N> and trained from there, because npaug
                     normalises every sample to a 2*img_size canvas -- 2N is exactly the detail
                     the pipeline can use. Originals are never modified.
  --test-npaug-dir PATH  DRY-RUN: no training; dump augmented images (grouped by class)
                     to PATH for <=4 epochs to visually test nptools/npaug.py.
  --test-npaug-class CLS  Restrict that dry-run to class folder(s) only (comma-separated).
  --no-arcface       Disable --arcface        (ON by default; needed by nptools/openset.py).
  --no-per-class-acc Disable --per-class-acc  (ON by default).
  --no-amp           Disable --amp (fp16 mixed precision, ON by default; the GPU is the
                     bottleneck once npaug's draft() decode is in place).
  --no-ncnn          Write <run>/export_ncnn.sh but do not run it. By default it IS run after
                     training: openset.py exports the open-set model (.pt + .onnx, prototypes
                     and calibrated threshold baked in) and pnnx converts it to ncnn.
  -h, --help         Show this help and exit.

--num-classes is derived automatically from the class-map (non-empty line count).
--workers is derived as 2/3 of the PHYSICAL cores (SMT siblings do not help this aug); override
with WORKERS=N in the environment, or pass --workers=N, since pass-through args come last.
The export/ncnn step runs by DEFAULT once training finishes (see --no-ncnn).
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
         --img-size)     IMG_SIZE="$2"; shift ;;
         --img-size=*)   IMG_SIZE="${1#*=}" ;;
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
         --ncnn)               NCNN=1 ;;
         --no-ncnn)            NCNN=0 ;;
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
 # than letting $(( )) treat it as 0 and silently build a train_0 split.
case "$IMG_SIZE" in
    ''|*[!0-9]*) echo "--img-size must be a positive integer, got '${IMG_SIZE}'" >&2; exit 2 ;;
esac
[ "$IMG_SIZE" -gt 0 ] || { echo "--img-size must be > 0" >&2; exit 2; }

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

 # ── Pre-resize the training split (see IMG_SIZE above) ──────────────────────────────────────────
 # --data-dir is the DATASET ROOT and must contain a train/ subfolder. Requiring it explicitly
 # matters because timm does not: create_dataset() calls _search_split(), which silently falls back
 # to the root when the named split is absent, so a wrong --data-dir trains on whatever happens to
 # be underneath it instead of failing.
DATA_DIR="${DATA_DIR%/}"
TRAIN_SPLIT="train"
if [ ! -d "${DATA_DIR}/${TRAIN_SPLIT}" ]; then
    echo "error: no '${TRAIN_SPLIT}' subfolder under --data-dir '${DATA_DIR}'." >&2
    echo "       --data-dir is the dataset ROOT; class folders belong in" >&2
    echo "       '${DATA_DIR}/${TRAIN_SPLIT}/<class>/'." >&2
    exit 2
fi

 # Pick the validation split by looking for it, and say which one won. train.py's default is
 # "validation", and timm resolves a MISSING split by falling back to the dataset root -- which now
 # holds train/ AND the resized train_<2N>/, so a missing val split would quietly validate on every
 # training image twice plus anything else under the root (outlier/, inlier/, ...). Resolving it
 # here makes the choice explicit and keeps that fallback from ever triggering.
VAL_SPLIT=""
for cand in val validation; do
    if [ -d "${DATA_DIR}/${cand}" ]; then
        VAL_SPLIT="${cand}"
        break
    fi
done

 # Resize train/ into a SIBLING SPLIT inside the same root, and point --train-split at it, so the
 # dataset root (and therefore --output, --class-map and any --val-split) stays put. Idempotent:
 # a second launch re-checks mtimes and rewrites nothing, so this costs a directory walk.
RESIZE_SIZE=$((2 * IMG_SIZE))            # npaug's canvas; see IMG_SIZE above
RESIZED_SPLIT="${TRAIN_SPLIT}_${RESIZE_SIZE}"
if uv run python -m nptools.resize_dataset \
        --src "${DATA_DIR}/${TRAIN_SPLIT}" --out "${DATA_DIR}/${RESIZED_SPLIT}" \
        --size "${RESIZE_SIZE}"; then
    TRAIN_SPLIT="${RESIZED_SPLIT}"
    echo "training from the resized split: ${DATA_DIR}/${TRAIN_SPLIT}"
else
    echo "resize_dataset failed -- training from the originals at ${DATA_DIR}/${TRAIN_SPLIT}"
fi

 # Resolved after the resize, so the no-held-out-data fallback reuses the split actually trained on
 # rather than the originals -- eval then at least shares the training preprocessing.
if [ -n "${VAL_SPLIT}" ]; then
    echo "validating on: ${DATA_DIR}/${VAL_SPLIT}"
else
    VAL_SPLIT="${TRAIN_SPLIT}"
    echo "note: no val/ or validation/ under ${DATA_DIR} -- validating on the TRAINING split" \
         "(${VAL_SPLIT}). eval_top1 then measures fit, not generalization, and model_best is" \
         "selected on it. Add ${DATA_DIR}/val/<class>/ for a real held-out metric."
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
MODEL="tf_efficientnet_lite0.in1k"

 # Signature of everything that defines WHAT is being trained, so a resume can tell "continuing the
 # same run" from "a different experiment that happens to start from these weights". Deliberately
 # excludes --resume/--pretrained (they differ by definition), --output and --experiment (where the
 # run lands), and --workers (a throughput knob with no effect on the model).
TRAIN_SIG="model=${MODEL} img_size=${IMG_SIZE} crop=${CROP_FLAGS} data_dir=${DATA_DIR}\
 train_split=${TRAIN_SPLIT} val_split=${VAL_SPLIT} class_map=${CLASS_MAP}\
 num_classes=${NUM_CLASSES} arcface=${ARCFACE_FLAGS} amp=${AMP_FLAGS} extra=${EXTRA_ARGS}"

 # Reuse the resumed run's folder when the signature matches, instead of scattering a fresh
 # timestamped directory per launch: continuing a run belongs in the run it continues, so
 # summary.csv, the checkpoint history and the export script stay in one place. A CHANGED signature
 # means this is a different experiment that merely starts from those weights, so it gets its own
 # folder -- keeping the old one honest about what produced it.
 #
 # Naming the folder ourselves (--experiment) rather than letting train.py timestamp it is what
 # makes this possible; the format matches what train.py would have generated.
 # Read one scalar out of a run's args.yaml (train.py writes it), tolerating quotes.
yaml_get() { sed -n "s/^$1: *//p" "$2" 2>/dev/null | head -1 | sed "s/^['\"]//; s/['\"]\$//"; }

 # Does that run's args.yaml agree with what we are about to launch? Used for run dirs written
 # before train_sig.txt existed: without this the no-signature case can never match, so every
 # launch forks a new folder and the next one resumes from the same old checkpoint and forks
 # again -- a loop that never self-heals. args.yaml is train.py's own record of the run, so it is
 # a better authority than our signature anyway; the signature just makes the check cheap.
args_yaml_matches() {
    _y="$1"
    [ -f "${_y}" ] || return 1
    [ "$(yaml_get model "${_y}")"       = "${MODEL}" ]        || return 1
    [ "$(yaml_get img_size "${_y}")"    = "${IMG_SIZE}" ]     || return 1
    [ "$(yaml_get num_classes "${_y}")" = "${NUM_CLASSES}" ]  || return 1
    [ "$(yaml_get data_dir "${_y}")"    = "${DATA_DIR}" ]     || return 1
    [ "$(yaml_get train_split "${_y}")" = "${TRAIN_SPLIT}" ]  || return 1
    [ "$(yaml_get val_split "${_y}")"   = "${VAL_SPLIT}" ]    || return 1
    [ "$(yaml_get class_map "${_y}")"   = "${CLASS_MAP}" ]    || return 1
    [ "$(yaml_get arcface "${_y}")"       = "$([ "$ARCFACE" = 1 ] && echo true || echo false)" ]       || return 1
    [ "$(yaml_get per_class_acc "${_y}")" = "$([ "$PER_CLASS_ACC" = 1 ] && echo true || echo false)" ] || return 1
    [ "$(yaml_get amp "${_y}")"           = "$([ "$AMP" = 1 ] && echo true || echo false)" ]           || return 1
    return 0
}

NEW_EXP_NAME="$(date +%Y%m%d-%H%M%S)-$(echo "${MODEL}" | tr '.' '_')-${IMG_SIZE}"
REUSED=0
if [ "$NEW" = "1" ]; then
    # train from scratch: fresh run from the pretrained backbone, don't resume
    RESUME_OR_NEW="--pretrained-path=last.pth.tar --pretrained"
    EXP_NAME="${NEW_EXP_NAME}"
else
    LAST=$(ls -td "${OUTPUT_DIR}"/*/last.pth.tar 2>/dev/null | head -1)
    if [ -n "$LAST" ]; then
        RESUME_OR_NEW="--resume=${LAST}"
        PREV_RUN=$(dirname "${LAST}")
        if [ -f "${PREV_RUN}/train_sig.txt" ]; then
            [ "$(cat "${PREV_RUN}/train_sig.txt")" = "${TRAIN_SIG}" ] && REUSED=1
        else
            args_yaml_matches "${PREV_RUN}/args.yaml" && REUSED=1
        fi
        if [ "$REUSED" = "1" ]; then
            EXP_NAME=$(basename "${PREV_RUN}")
            echo "resuming into the same run dir: ${PREV_RUN}"
        else
            EXP_NAME="${NEW_EXP_NAME}"
            echo "resuming from ${LAST}, but that run's settings differ --"
            echo "  starting a new run dir: ${OUTPUT_DIR}/${EXP_NAME}"
            if [ -f "${PREV_RUN}/train_sig.txt" ]; then
                diff -u "${PREV_RUN}/train_sig.txt" - <<EOF | sed -n '4,$p' | sed 's/^/    /'
${TRAIN_SIG}
EOF
            fi
        fi
    else
        RESUME_OR_NEW="--pretrained-path=last.pth.tar  --pretrained"
        EXP_NAME="${NEW_EXP_NAME}"
    fi
fi
RUN_DIR="${OUTPUT_DIR}/${EXP_NAME}"
 # Deliberately NOT created here: train.py makes it, and a run that dies before writing a
 # checkpoint should not leave an empty directory behind (that is how five of them piled up).
 # train_sig.txt is written after training instead, for the same reason.


 # # #"$PYTHON"
 uv run python  "$@" \
 --model="${MODEL}" \
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
     --train-split="${TRAIN_SPLIT}" \
     --val-split="${VAL_SPLIT}" \
     --class-map="${CLASS_MAP}" \
     --output="${OUTPUT_DIR}" \
     --experiment="${EXP_NAME}" \
     --num-classes="${NUM_CLASSES}" \
     ${TEST_NPAUG_DIR:+--test-npaug-dir="${TEST_NPAUG_DIR}"} \
     ${TEST_NPAUG_CLASS:+--test-npaug-class="${TEST_NPAUG_CLASS}"} \
     ${ARCFACE_FLAGS} \
     ${AMP_FLAGS} \
     ${RESUME_OR_NEW} \
     ${EXTRA_ARGS}
     #--bg-dir="${DATA_DIR}/bg_photos" \
TRAIN_STATUS=$?


 # Record the signature only now: a run that never got off the ground leaves nothing to resume, so
 # marking its directory would just teach the next launch to reuse an empty one.
if [ -d "${RUN_DIR}" ] && [ "$TRAIN_STATUS" = "0" ]; then
    printf '%s\n' "${TRAIN_SIG}" > "${RUN_DIR}/train_sig.txt"
fi

 # A failed launch that produced no checkpoint leaves a directory holding nothing but args.yaml.
 # Prune it -- but only if THIS launch created it (never a reused one), and only when it really is
 # checkpoint-free, so an interrupted run that did save something is always kept.
if [ "$TRAIN_STATUS" != "0" ] && [ "$REUSED" != "1" ] && [ -d "${RUN_DIR}" ]; then
    if [ -z "$(ls "${RUN_DIR}"/*.pth.tar 2>/dev/null)" ]; then
        rm -rf "${RUN_DIR}"
        echo "removed the empty run dir ${RUN_DIR} (training produced no checkpoint)"
    fi
fi

 # Same idea for the reused-folder case: timm's update_summary() appends and passes
 # write_header=(best_metric is None), which is true at the start of every run, so resuming into an
 # existing dir leaves a second header row mid-file and breaks anything reading it as a CSV.
 # Repaired here rather than in train.py, so timm stays untouched and this wrapper owns the
 # consequences of its own folder reuse. Idempotent, and it also cleans up older files.
SUMMARY="${RUN_DIR}/summary.csv"
if [ -s "${SUMMARY}" ]; then
    awk 'NR==1 {hdr = $0; print; next} $0 != hdr' "${SUMMARY}" > "${SUMMARY}.tmp"
    if [ "$(wc -l < "${SUMMARY}")" != "$(wc -l < "${SUMMARY}.tmp")" ]; then
        mv "${SUMMARY}.tmp" "${SUMMARY}"
        echo "summary.csv: dropped repeated header row(s) left by resuming into this dir"
    else
        rm -f "${SUMMARY}.tmp"
    fi
fi


 # ── After training: WRITE (do not run) an export script into the run dir ────────────────────────
 # The follow-up steps need paths and sizes only this script knows (the timestamped run dir, the
 # class-map, openset.py's data-dir, the input size), so reconstructing them by hand is where the
 # mistakes happen. Emitting a runnable script rather than text to copy also means the run dir
 # carries the exact command that produced its artifacts -- reproducible months later, when the
 # shell history is gone and the wrapper's defaults may have moved on.
 #
 # openset.py takes the same dataset root and reads <root>/train itself, i.e. the ORIGINAL images --
 # never the pre-shrunk training split. That is deliberate: prototypes and the calibrated threshold
 # have to come from the images a deployed client will actually see.
OPENSET_DATA_DIR="${DATA_DIR}"
 # THIS run's dir, not the newest one on disk: with a reused folder they are the same, but with a
 # concurrent run in the same --output they would not be, and exporting another run's checkpoint
 # is the kind of mistake nothing downstream would reveal.
BEST="${RUN_DIR}/model_best.pth.tar"
[ -f "${BEST}" ] || BEST=""

if [ "$TRAIN_STATUS" != "0" ]; then
    echo
    echo "train.py exited with status ${TRAIN_STATUS} -- no export script written."
elif [ -z "$BEST" ]; then
    echo
    echo "No model_best.pth.tar in ${RUN_DIR} -- nothing to export."
else
EXPORT_SH="${RUN_DIR}/export_ncnn.sh"
 # Quoted heredoc delimiter ('EOF'): everything below is written VERBATIM except the few values
 # interpolated by hand via ${...} below -- so $0/$@/$(...) inside the generated script survive to
 # be evaluated when IT runs, not now.
{
cat <<EOF
#!/bin/sh
# Generated by nptools/train_posm.sh on $(date '+%Y-%m-%d %H:%M:%S') -- edit freely.
# Builds the open-set model from this run's checkpoint and converts it to ncnn.
#
#   openset.pt / openset.onnx     the exported model (+ .meta.json preprocessing contract)
#   openset.ncnn.param / .bin     what you ship; pnnx writes them next to the .pt
#   eval_rounds.csv               per-class recall/precision, all crops vs inliers only
set -e
cd "$(pwd)"

CKPT="${BEST}"
DATA_DIR="${OPENSET_DATA_DIR}"
CLASS_MAP="${CLASS_MAP}"
IMG_SIZE=${IMG_SIZE}
OUT_DIR="${RUN_DIR}"
EOF
cat <<'EOF'

# Both formats in ONE run: prototype extraction + threshold calibration is the expensive part and
# is shared, and it guarantees the .pt and the .onnx carry the same prototypes and threshold (a
# second run would recalibrate).
#
# --img-size must match what was TRAINED. openset.py defaults to 224 independently of the training
# wrapper, and a mismatch does not error -- the size is baked into the graph and recorded in the
# sidecar, so prototypes get built at one scale, the client feeds another, and accuracy quietly
# drops. It is pinned above from the value training actually used.
#
# --no-argmin keeps ArgMin/Gather out of the graph so a STOCK ncnn wheel can run it; the client does
# argmin over the emitted per-class dists and compares margins[best] to 0. Dropping it bakes the
# decision into the model but then needs a custom ncnn layer.
#
# --data-dir is the dataset ROOT: openset.py reads <root>/train itself, i.e. the ORIGINAL images,
# never a pre-shrunk training split -- prototypes and the threshold must come from what a deployed
# client will actually see.
# --eval-csv scores the training crops per class TWICE -- over everything, and again with the
# outlier crops dropped -- into (class type, recall rate, precision rate, eval type). Same
# prototypes and threshold both rounds, so the delta is exactly what those crops cost. Free: it
# reuses the feature pass the export already ran.
uv run python nptools/openset.py \
    --data-dir "${DATA_DIR}" \
    --class-map "${CLASS_MAP}" \
    --checkpoint "${CKPT}" \
    --img-size "${IMG_SIZE}" \
    --eval-csv "${OUT_DIR}/eval_rounds.csv" \
    --export-pt "${OUT_DIR}/openset.pt" \
    --export-onnx "${OUT_DIR}/openset.onnx" \
    --no-argmin

# pnnx is not a project dependency; install on demand rather than failing three minutes in.
command -v pnnx >/dev/null 2>&1 || uv run pnnx --help >/dev/null 2>&1 || uv pip install pnnx ncnn

# inputshape must match --img-size: pnnx traces at that shape and constant-folds Conv2dSame's
# dynamic padding against it, so a wrong value bakes WRONG PADDING into the .param instead of
# failing. Outputs land next to the .pt, alongside .pnnx.* intermediates that are safe to delete.
uv run pnnx "${OUT_DIR}/openset.pt" "inputshape=[1,3,${IMG_SIZE},${IMG_SIZE}]"

echo
echo "ncnn files:"
ls -1 "${OUT_DIR}"/openset.ncnn.* 2>/dev/null || echo "  (none -- check the pnnx output above)"
echo "class order + threshold + preprocessing: ${OUT_DIR}/openset.pt.meta.json"
echo "per-class recall/precision (2 rounds):      ${OUT_DIR}/eval_rounds.csv"
EOF
} > "${EXPORT_SH}"
chmod +x "${EXPORT_SH}"
echo
echo "Wrote ${EXPORT_SH}"
echo "  (pins this run's checkpoint, --img-size ${IMG_SIZE} and --no-argmin, then pnnx with the"
echo "   matching inputshape -- edit and re-run it any time)"

if [ "$NCNN" = "1" ]; then
    echo
    echo "Running it now (--no-ncnn to skip) ..."
    if sh "${EXPORT_SH}"; then
        echo
        echo "export + ncnn conversion done: ${RUN_DIR}"
    else
        # Do not fail the whole run: training succeeded and its checkpoint is safe on disk, so the
        # useful outcome is preserved and the export is one re-runnable command away.
        echo
        echo "export failed -- the checkpoint is intact; fix the cause and re-run:" >&2
        echo "  sh ${EXPORT_SH}" >&2
    fi
else
    echo
    echo "Skipping the export (--no-ncnn). To produce the ncnn files:"
    echo "  sh ${EXPORT_SH}"
fi
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