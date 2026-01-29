# inference.ps1

$MODEL_PATH = "models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf"
$LOG_DIR = "log"
$LOG_FILE = "$LOG_DIR/inference.log"
$RUNTIME_FILE = "$LOG_DIR/runtime.txt"

$N_TOKENS = 128
$P_TOKENS = 512
#  -p, --n-prompt <n>                        (default: 512)
#  -n, --n-gen <n>                           (default: 128)

Write-Host "Running llama.cpp inference..."

$START = Get-Date
llama-bench -m $MODEL_PATH -t 1 -p $P_TOKENS -n $N_TOKENS -o json | Out-File -FilePath $LOG_FILE -Encoding GBK
$END = Get-Date

$RUNTIME_S = New-TimeSpan -Start $START -End $END
$RUNTIME_MS = $RUNTIME_S.TotalMilliseconds / 1000

Write-Host "Inference time: $($RUNTIME_MS) seconds"
$RUNTIME_MS | Out-File -FilePath $RUNTIME_FILE