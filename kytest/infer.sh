#!/bin/sh

DATA_DIR=/home/keyong/cls2/code
OUTPUT_DIR="${DATA_DIR}/posmlv/output"

# Auto-find the latest training run directory
LATEST_RUN=$(ls -td "${OUTPUT_DIR}"/20*-tf_efficientnet_lite0_in1k-* 2>/dev/null | head -1)
if [ -z "$LATEST_RUN" ]; then
    echo "No training output found in ${OUTPUT_DIR}"
    exit 1
fi

CHECKPOINT="${LATEST_RUN}/model_best.pth.tar"
if [ ! -f "$CHECKPOINT" ]; then
    CHECKPOINT="${LATEST_RUN}/last.pth.tar"
fi
echo "Using checkpoint: ${CHECKPOINT}"

# # Default inference on training data; override with: INFER_DIR=/path/to/images ./infer.sh
INFER_DIR="${INFER_DIR:-${DATA_DIR}/posmlv/test}"

uv run python inference.py \
    --model=tf_efficientnet_lite0.in1k \
    --checkpoint="${CHECKPOINT}" \
    --data-dir="${INFER_DIR}" \
    --split=test\
    --num-classes=84 \
    --class-map="${DATA_DIR}/posmlv/class_84.txt" \
    --crop-mode=squash \
    --batch-size=48 \
    --topk=3 \
    --results-dir="${OUTPUT_DIR}/inference_results" \
    --results-file="results.csv"

# uv run python validate.py \
#     --model=tf_efficientnet_lite0.in1k \
#     --checkpoint="${CHECKPOINT}" \
#     --data-dir="${DATA_DIR}"/"${DATASET_NAME}" \
#     --num-classes=3 \
#     --batch-size=48 \
#     --class-map="${DATA_DIR}"/"${DATASET_NAME}"/class_3.txt \
#     --results-dir="${OUTPUT_DIR}/inference_results" \
#     --results-file="results.csv"