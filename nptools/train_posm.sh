#!/bin/sh
 
 # background images for nobg bg-swap augmentation (read by timm/data/npaug.py)
export NOBG_BG_DIR=/home/keyong/cls6/bg_photos
#export AUG_DUMP_DIR=kytest/after_aug

DATA_DIR=/home/keyong/cls2/code/posmlv
OUTPUT_DIR=/home/keyong/cls2/code/posmlv/output

 # Preserve args passed to this script (e.g. --per-class-acc) so they reach train.py.
 # Must capture BEFORE `set --` below repurposes $@ as the python interpreter prefix.
 EXTRA_ARGS="$*"

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
LAST=$(ls -td "${OUTPUT_DIR}"/20*/last.pth.tar | head -1)
if [ -n "$LAST" ]; then
    RESUME_OR_NEW="--resume=${LAST}"
else
    RESUME_OR_NEW="--pretrained-path=last.pth.tar  --pretrained"
fi


 # # #"$PYTHON"
 uv run python  "$@" \
 --model=tf_efficientnet_lite0.in1k \
    --lr 0.001 \
    --warmup-epochs 2 \
    --aa=originalr  \
    --color-jitter 0.4 --reprob 0.2 \
    --nobg --heavy-aug \
     --crop-pct=1 --batch-size=48 \
     ${CROP_FLAGS} \
     --checkpoint-hist=10 \
     --epochs=60 \
     --data-dir="${DATA_DIR}" \
     --class-map="${DATA_DIR}/class_84.txt" \
     --output="${OUTPUT_DIR}" \
     --num-classes=83 \
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