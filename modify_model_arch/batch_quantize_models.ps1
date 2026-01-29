# ==========================================================
# 脚本配置区：请根据你的实际情况修改以下变量
# ==========================================================

# 1. 模型的基路径和名称
$MODEL_BASE_PATH = "E:\Code\llm_inference_roofline_detect\models\modified"
#$MODEL_NAME = "Qwen2.5-1.5B-Instruct"
#$MODEL_NAME = "Qwen3-0.6B"
#$MODEL_NAME = "Llama-3.2-1B-Instruct"
#$MODEL_NAME = "PLM-1.8B-Instruct"
#$MODEL_NAME = "Fox-1-1.6B"
$MODEL_NAME = "SmolLM2-1.7B-Instruct"

# 2. 要量化的层数范围和步长
$START_LAYERS = 2
$END_LAYERS = 64
$STEP_LAYERS = 2

# 3. 输入 GGUF 文件的类型后缀
$INPUT_GGUF_TYPE = "f16"

# 4. 量化类型 (例如 q4_0, q5_k_m 等)
$QUANT_TYPE = "q4_k_m"

# ==========================================================
# 脚本执行区：通常无需修改以下内容
# ==========================================================

Write-Host "====================================================="
Write-Host "开始批量量化 GGUF 模型..."
Write-Host "-----------------------------------------------------"
Write-Host "模型路径基目录: $MODEL_BASE_PATH"
Write-Host "模型名称基目录: $MODEL_NAME"
Write-Host "量化层数范围: $START_LAYERS 到 $END_LAYERS，步长为 $STEP_LAYERS"
Write-Host "输入模型类型: $INPUT_GGUF_TYPE"
Write-Host "量化类型: $QUANT_TYPE"
Write-Host "====================================================="

# 检查量化工具是否可用
if (-not (Get-Command llama-quantize -ErrorAction SilentlyContinue)) {
    Write-Host "错误：未找到 llama-quantize 命令。请确保已通过 winget 正确安装 llama.cpp。"
    exit 1
}

# 循环遍历指定范围的层数，并使用步长
for ($i = $START_LAYERS; $i -le $END_LAYERS; $i += $STEP_LAYERS) {
    # 构造输入和输出的模型路径
    $INPUT_FILE = "${MODEL_BASE_PATH}\${MODEL_NAME}-${i}-layers-${INPUT_GGUF_TYPE}.gguf"
    $OUTPUT_FILE = "${MODEL_BASE_PATH}\${MODEL_NAME}-${i}-layers-${QUANT_TYPE}.gguf"

    # 检查输入文件是否存在
    if (-not (Test-Path $INPUT_FILE)) {
        Write-Host "⚠️ 警告：跳过，未找到输入文件：$INPUT_FILE"
        continue
    }

    Write-Host "" # 添加空行以提高可读性
    Write-Host "-----------------------------------------------------"
    Write-Host "正在量化模型：层数 $i"
    Write-Host "输入文件：$INPUT_FILE"
    Write-Host "输出文件：$OUTPUT_FILE"
    Write-Host "-----------------------------------------------------"

    # 打印即将执行的命令
    Write-Host "执行命令：llama-quantize '$INPUT_FILE' '$OUTPUT_FILE' '$QUANT_TYPE'"

    # 执行量化命令
    llama-quantize $INPUT_FILE $OUTPUT_FILE $QUANT_TYPE

    # 检查量化命令是否成功
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ 量化成功！"
    } else {
        Write-Host "❌ 量化失败，请检查错误信息。"
    }
}

Write-Host ""
Write-Host "所有指定范围的模型量化已完成。"
Write-Host "脚本执行完毕。"