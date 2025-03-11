#!/bin/bash

# Fixed values for server configuration

PORT=8100
API_KEY="abc"

MODEL_PATH="/ephemeral/home/xiong/data/hf_cache/Qwen/Qwen2.5-7B-Instruct"
SERVED_MODEL_NAME="Qwen2.5-7B-Instruct"
GPU_DEVICE="0,1,2,3"
TP_SIZE=4
# DP_SIZE=4

# Set GPU device
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

# Start SGLang server
python -m sglang.launch_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --dtype bfloat16 \
    --api-key "$API_KEY" \
    --context-length 4096 \
    --served-model-name "$SERVED_MODEL_NAME" \
    --allow-auto-truncate \
    --constrained-json-whitespace-pattern "[\n\t ]*" \
    --tp $TP_SIZE
 #   --dp-size $DP_SIZE

## some issues reported : https://github.com/sgl-project/sglang/issues/2216