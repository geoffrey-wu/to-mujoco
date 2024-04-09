#!/bin/bash

GPU=$1

if [ -z "$GPU" ]; then
    echo "Usage: $0 <gpu>"
    exit 1
fi

MUJOCO_GL=egl \
PATH=/usr/local/cuda-11.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64 \
CUDA_VISIBLE_DEVICES=$GPU \
# TF_CPP_MIN_LOG_LEVEL=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=.25 \
python train.py --env_name $ENV_NAME --experiment_name $EXPERIMENT_NAME
