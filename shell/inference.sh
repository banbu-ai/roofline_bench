#!/bin/bash

MODEL_PATH="models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf"
LOG_DIR="log"
LOG_FILE="$LOG_DIR/inference.json"
RUNTIME_FILE="$LOG_DIR/runtime.txt"

P_TOKENS=512
N_TOKENS=128
#  -p, --n-prompt <n>                        (default: 512)
#  -n, --n-gen <n>                           (default: 128)

echo "Running llama.cpp inference..."

START=$(date +%s%N)
llama-bench -m $MODEL_PATH -t 1 -p $P_TOKENS -n $N_TOKENS -o json > $LOG_FILE
END=$(date +%s%N)

RUNTIME_NS=$((END - START))
RUNTIME_SEC=$(echo "scale=6; $RUNTIME_NS/1000000000" | bc)

echo "Inference time: $RUNTIME_SEC seconds"
echo $RUNTIME_SEC > $RUNTIME_FILE