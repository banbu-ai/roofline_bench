#!/bin/bash

LOG_DIR="log"
LOG_FILE="$LOG_DIR/record_mem.log" # 内存记录 单位KB
INFERENCE_SCRIPT="inference.sh"
INTERVAL=0.01 # 监控间隔，单位为秒

# 如果LOG目录不存在自动创建
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# 运行推理脚本，并在后台获取其PID
bash "$INFERENCE_SCRIPT" &
INFERENCE_PID=$!

# 循环监控该PID的内存使用，并写入日志
echo "Recording peak memory usage for PID $INFERENCE_PID..."
echo "" > "$LOG_FILE" # 清空日志文件

while kill -0 "$INFERENCE_PID" 2>/dev/null; do
    # -o rss: 输出 Resident Set Size (RSS) 进程实际使用的物理内存量 单位KB
    # -p: 指定 PID
    # | grep -v 'RSS' 是为了排除表头
    ps -o rss -p "$INFERENCE_PID" | grep -v 'RSS' >> "$LOG_FILE"

    sleep "$INTERVAL"
done

echo "Monitoring complete. Log saved to $LOG_FILE."