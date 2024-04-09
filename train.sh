#!/bin/bash

GPU=$1
ENV_NAME=$2
EXPERIMENT_NAME=$3

# if any of the arguments are empty
if [ -z "$GPU" ] || [ -z "$ENV_NAME" ] || [ -z "$EXPERIMENT_NAME" ]; then
    echo "Usage: $0 <gpu> <env_name> <experiment_name>"
    exit 1
fi

MUJOCO_GL=egl \
PATH=/usr/local/cuda-11.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64 \
CUDA_VISIBLE_DEVICES=$GPU \
# TF_CPP_MIN_LOG_LEVEL=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=.13 \
python train.py --env_name $ENV_NAME --experiment_name $EXPERIMENT_NAME
