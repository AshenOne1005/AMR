#!/usr/bin/env bash
# Pretrain on NTU60 xsub. Select dataset with --cfg-name (see config/pretrain.yaml).
set -e

CFG=${CFG:-config/pretrain.yaml}
NAME=${NAME:-ntu60_xsub}
GPUS=${GPUS:-0}   # comma-separated device ids, e.g. GPUS=0,2,3
export CUDA_VISIBLE_DEVICES=$GPUS
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
OUT=${OUT:-./output/pretrain_${NAME}}

# Pick a free rendezvous port per run, so concurrent jobs don't fight over 29500.
PORT=${PORT:-$(python -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')}

torchrun --nproc_per_node=$NGPU --master-port=$PORT main_pretrain.py \
    --config $CFG \
    --cfg-name $NAME \
    --output_dir "$OUT" \
    --log_dir "$OUT" \
    "$@"
