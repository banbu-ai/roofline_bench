#!/bin/bash

MODEL_PATH="./models/qwen1.5-0.5B.gguf"
PROMPT="你好，今天的天气如何？"
N_TOKENS=32

echo "Running llama.cpp inference..."

START=$(date +%s%N)
./build/bin/llama-bench -m $MODEL_PATH -t 1 -p "$PROMPT" -n $N_TOKENS --log-disable > output.log
END=$(date +%s%N)

RUNTIME_NS=$((END - START))
RUNTIME_SEC=$(echo "scale=6; $RUNTIME_NS/1000000000" | bc)

echo "Inference time: $RUNTIME_SEC seconds"
echo $RUNTIME_SEC > runtime.txt
