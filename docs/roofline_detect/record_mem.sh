#!/bin/bash

echo "Recording tegrastats..."
sudo tegrastats --interval 100 --logfile tegrastats.log &
PID=$!

# run inference
bash run_llama.sh

# kill tegrastats
kill $PID
sleep 1
