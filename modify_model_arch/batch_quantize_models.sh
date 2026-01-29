#!/bin/bash

# ==========================================================
# 脚本配置区：请根据你的实际情况修改以下变量
# ==========================================================

# 1. 模型的基路径和名称
MODEL_BASE_PATH="/Users/saber/Code/llm_inference_roofline_detect/models/modified"
#MODEL_NAME="Qwen2.5-1.5B-Instruct"
#MODEL_NAME="Qwen3-0.6B"
#MODEL_NAME="Llama-3.2-1B-Instruct"
#MODEL_NAME="PLM-1.8B-Instruct"
#MODEL_NAME="Fox-1-1.6B"
MODEL_NAME="SmolLM2-1.7B-Instruct"

# 2. 要量化的层数范围和步长
START_LAYERS=2
END_LAYERS=64
STEP_LAYERS=2

# 3. 输入 GGUF 文件的类型后缀
INPUT_GGUF_TYPE="f16"

# 4. 量化类型 (例如 q4_0, q5_k_m 等)
QUANT_TYPE="q4_k_m"

# ==========================================================
# 脚本执行区：通常无需修改以下内容
# ==========================================================

echo "====================================================="
echo "开始批量量化 GGUF 模型..."
echo "-----------------------------------------------------"
echo "模型路径基目录: ${MODEL_BASE_PATH}"
echo "模型名称基目录: ${MODEL_NAME}"
echo "量化层数范围: ${START_LAYERS} 到 ${END_LAYERS}，步长为 ${STEP_LAYERS}"
echo "输入模型类型: ${INPUT_GGUF_TYPE}"
echo "量化类型: ${QUANT_TYPE}"
echo "====================================================="

# 检查量化工具是否可用
if ! command -v llama-quantize &> /dev/null; then
    echo "错误：未找到 llama-quantize 命令。请确保已通过 Homebrew 正确安装 llama.cpp。"
    exit 1
fi

# 循环遍历指定范围的层数，并使用步长
for ((i=$START_LAYERS; i<=$END_LAYERS; i+=STEP_LAYERS))
do
    # 构造输入和输出的模型路径
    INPUT_FILE="${MODEL_BASE_PATH}/${MODEL_NAME}-${i}-layers-${INPUT_GGUF_TYPE}.gguf"
    OUTPUT_FILE="${MODEL_BASE_PATH}/${MODEL_NAME}-${i}-layers-${QUANT_TYPE}.gguf"

    # 检查输入文件是否存在
    if [ ! -f "${INPUT_FILE}" ]; then
        echo "⚠️ 警告：跳过，未找到输入文件：${INPUT_FILE}"
        continue
    fi

    echo "" # 添加空行以提高可读性
    echo "-----------------------------------------------------"
    echo "正在量化模型：层数 ${i}"
    echo "输入文件：${INPUT_FILE}"
    echo "输出文件：${OUTPUT_FILE}"
    echo "-----------------------------------------------------"

    # 打印即将执行的命令
    echo "执行命令：llama-quantize \"${INPUT_FILE}\" \"${OUTPUT_FILE}\" \"${QUANT_TYPE}\""

    # 执行量化命令
    llama-quantize "${INPUT_FILE}" "${OUTPUT_FILE}" "${QUANT_TYPE}"

    # 检查量化命令是否成功
    if [ $? -eq 0 ]; then
        echo "✅ 量化成功！"
    else
        echo "❌ 量化失败，请检查错误信息。"
    fi
done

echo ""
echo "所有指定范围的模型量化已完成。"
echo "脚本执行完毕。"