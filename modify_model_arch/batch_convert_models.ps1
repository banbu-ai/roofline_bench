# ==========================================================
# 脚本配置区：请根据你的实际情况修改以下变量
# ==========================================================

# 1. llama.cpp 目录的路径
$LLAMA_CPP_DIR = "E:\Code\llama.cpp"

# 2. conda 环境的名称
$CONDA_ENV_NAME = "llama.cpp"

# 3. 模型的基路径和名称
$MODEL_BASE_PATH = "E:\Code\llm_inference_roofline_detect\models\modified"
#$MODEL_NAME = "Qwen2.5-1.5B-Instruct"
#$MODEL_NAME = "Qwen3-0.6B"
#$MODEL_NAME = "Llama-3.2-1B-Instruct"
$MODEL_NAME = "PLM-1.8B-Instruct"
#$MODEL_NAME = "Fox-1-1.6B"
#$MODEL_NAME = "SmolLM2-1.7B-Instruct"

# 4. 要转换的层数范围和步长
$START_LAYERS = 2
$END_LAYERS = 64
$STEP_LAYERS = 2

# 5. GGUF 的输出类型 (例如 f16, q8_0 等)
#  --outtype {f32,f16,bf16,q8_0,tq1_0,tq2_0,auto}
#  output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, tq1_0 or tq2_0 for ternary, and auto for the highest-fidelity 16-bit float type depending on the first loaded tensor type
$GGUF_OUTTYPE = "f16"
#$GGUF_OUTTYPE = "q8_0"

# ==========================================================
# 脚本执行区：通常无需修改以下内容
# ==========================================================

Write-Host "执行命令：Set-Location -Path '$LLAMA_CPP_DIR'"
Set-Location -Path $LLAMA_CPP_DIR -ErrorAction Stop
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误：无法进入目录 $LLAMA_CPP_DIR。请检查路径。"
    exit 1
}

Write-Host "正在激活 conda 环境 $CONDA_ENV_NAME..."
# 检查 conda 是否可用
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "错误：未找到 conda 命令。请确保已安装 conda 并已添加到 PATH。"
    exit 1
}

Write-Host "执行命令：conda activate '$CONDA_ENV_NAME'"
conda activate $CONDA_ENV_NAME

# 检查环境是否成功激活
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误：无法激活 conda 环境 $CONDA_ENV_NAME。请检查环境名称。"
    exit 1
}

Write-Host "====================================================="
Write-Host "开始批量转换模型..."
Write-Host "-----------------------------------------------------"
Write-Host "模型路径基目录: $MODEL_BASE_PATH"
Write-Host "模型名称基目录: $MODEL_NAME"
Write-Host "转换层数范围: $START_LAYERS 到 $END_LAYERS，步长为 $STEP_LAYERS"
Write-Host "输出类型: $GGUF_OUTTYPE"
Write-Host "====================================================="

# 循环遍历指定范围的层数，并使用步长
for ($i = $START_LAYERS; $i -le $END_LAYERS; $i += $STEP_LAYERS) {
    # 构造输入和输出的模型路径
    $INPUT_MODEL_PATH = "${MODEL_BASE_PATH}\${MODEL_NAME}-${i}-layers"
    $OUTPUT_FILE = "${INPUT_MODEL_PATH}-${GGUF_OUTTYPE}.gguf"

    Write-Host "" # 添加空行以提高可读性
    Write-Host "-----------------------------------------------------"
    Write-Host "正在转换模型：层数 $i"
    Write-Host "输入路径：$INPUT_MODEL_PATH"
    Write-Host "输出文件：$OUTPUT_FILE"
    Write-Host "-----------------------------------------------------"

    # 打印即将执行的命令
    Write-Host "执行命令：python convert_hf_to_gguf.py '$INPUT_MODEL_PATH' --outfile '$OUTPUT_FILE' --outtype '$GGUF_OUTTYPE' --verbose"

    # 执行转换命令
    python convert_hf_to_gguf.py "$INPUT_MODEL_PATH" `
      --outfile "$OUTPUT_FILE" `
      --outtype "$GGUF_OUTTYPE" `
      --verbose

    # 检查转换命令是否成功
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ 转换成功！"
    } else {
        Write-Host "❌ 转换失败，请检查错误信息。"
    }
}

Write-Host ""
Write-Host "所有指定范围的模型转换已完成。"
Write-Host "执行命令：conda deactivate"
conda deactivate
Write-Host "脚本执行完毕。"