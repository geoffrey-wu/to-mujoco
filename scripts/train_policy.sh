#!/bin/bash
GPUS=$1
SEED=$2
EXP_NAME=$3
TASK=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python3 train.py task=${TASK} headless=True seed=${SEED} \
train.algo=HANDEM \
train.handem.output_name="${EXP_NAME}" \
${EXTRA_ARGS}
