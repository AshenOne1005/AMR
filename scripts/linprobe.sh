#!/usr/bin/env bash
# Linear probing on frozen pretrained encoder.
# Pass the pretrained checkpoint with FINETUNE (or --finetune directly).
set -e

CFG=${CFG:-config/linprobe.yaml}
NAME=${NAME:-ntu60_xsub}
GPUS=${GPUS:-0}   # comma-separated device ids, e.g. GPUS=0,2,3
export CUDA_VISIBLE_DEVICES=$GPUS
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
CKPT=${CKPT:?Set CKPT to the pretrained checkpoint, e.g. output/pretrain_ntu60_xsub/checkpoint-399.pth}
OUT=${OUT:-./output/linprobe_${NAME}}

# Pick a free rendezvous port per run, so concurrent jobs don't fight over 29500.
PORT=${PORT:-$(python -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')}

torchrun --nproc_per_node=$NGPU --master-port=$PORT main_linprobe.py \
    --config $CFG \
    --cfg-name $NAME \
    --finetune $CKPT \
    --output_dir $OUT \
    --log_dir $OUT \
    "$@"
