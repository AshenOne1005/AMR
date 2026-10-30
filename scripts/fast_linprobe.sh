#!/usr/bin/env bash
# Single-process fast linprobe: cache features once, then train the classifier.
# No data augmentation; usually slightly lower accuracy than regular linprobe,
# but substantially faster. GPUS selects the visible device; no torchrun needed.
set -e

CFG=${CFG:-config/linprobe.yaml}
NAME=${NAME:-ntu60_xsub}
GPUS=${GPUS:-0}
export CUDA_VISIBLE_DEVICES=$GPUS
CKPT=${CKPT:?Set CKPT to a pretrained encoder checkpoint}
OUT=${OUT:-./output/fast_linprobe_${NAME}}

python main_fast_linprobe.py \
    --config "$CFG" \
    --cfg-name "$NAME" \
    --finetune "$CKPT" \
    --output_dir "$OUT" \
    --log_dir "$OUT" \
    "$@"
