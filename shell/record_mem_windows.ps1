# record_mem_macos.ps1

$LOG_DIR = "log"
$LOG_FILE = "log/record_mem.log"
$INFERENCE_SCRIPT = "inference.ps1"
$INTERVAL = 0.01

# 如果LOG目录不存在自动创建
if (-not (Test-Path -Path $LOG_DIR -PathType Container)) {
    New-Item -Path $LOG_DIR -ItemType Directory | Out-Null
}

# 运行推理脚本，并在后台获取其PID
$inferenceProcess = Start-Process -FilePath "pwsh.exe" -ArgumentList "-File", $INFERENCE_SCRIPT -WindowStyle Hidden -PassThru
$INFERENCE_PID = $inferenceProcess.Id

# 循环监控该PID的内存使用，并写入日志
Write-Host "Recording peak memory usage for PID $INFERENCE_PID..."
# 清空日志文件
if (-not (Test-Path -Path $LOG_FILE)) {
    New-Item -Path $LOG_FILE -ItemType File | Out-Null
}
Clear-Content -Path $LOG_FILE

while (Get-Process -Id $INFERENCE_PID -ErrorAction SilentlyContinue) {
    # Get-Process 获取进程对象
    # .WorkingSet / 1KB 将单位转换为 KB
    $memoryUsage = (Get-Process -Id $INFERENCE_PID).WorkingSet / 1KB
    $memoryUsage | Out-File -FilePath $LOG_FILE -Append

    Start-Sleep -Seconds $INTERVAL
}

Write-Host "Monitoring complete. Log saved to $LOG_FILE."