import json
import os

from utils.common_utils import CommonUtils

model_name_list = ["Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "PLM-1.8B-Instruct", "Qwen3-0.6B", "Fox-1-1.6B", "SmolLM2-1.7B-Instruct"]
analysis_dir = os.path.expanduser("~/Code/llm_inference_roofline_detect/analysis/analysis_M1Pro")
# analysis_dir = os.path.expanduser("~/Code/llm_inference_roofline_detect/analysis/analysis_RTX 3070Ti")
# analysis_dir = os.path.expanduser("~/Code/llm_inference_roofline_detect/analysis/analysis_Jetson")

def merge_analysis_json(model_name, analysis_dir, prec):
    """
    合并指定目录下特定模型名称特定精度的所有 analysis 文件
    Args:
        model_name: 想要查找的模型名称，例如 "Qwen3-0.6B"
        analysis_dir: analysis_file 存放目录
        prec: f16, q8_0, q4_k_m
    """
    file_list = CommonUtils.find_file_in_directory(directory=analysis_dir,
                                                   strings_to_find=[model_name, prec],
                                                   extension=".json")
    file_list.sort()
    merged_json_dir = os.path.join(analysis_dir, "merged")
    os.makedirs(merged_json_dir, exist_ok=True)
    # 用于存储所有 JSON 数据的列表
    merged_data = []
    merged_json_path = os.path.join(merged_json_dir, model_name + "_" + prec + "_inference_analysis.json")
    # 遍历目录中的所有文件
    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 假设每个文件都是一个 JSON 列表
                data = json.load(f)
                # 如果读取成功，将数据添加到总列表中
                merged_data.extend(data)
        except json.JSONDecodeError as e:
            print(f"警告：跳过文件 {file_path}，因为它不是一个有效的 JSON 文件。错误: {e}")
        except Exception as e:
            print(f"处理文件 {file_path} 时发生错误: {e}")
        with open(merged_json_path, 'w+', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
    if len(file_list) != 0:
        print(f"{model_name} {prec} 共有 {len(file_list)} 个 analysis 文件")
        print(f"所有 JSON 文件已成功合并到 {merged_json_path}")


if __name__ == "__main__":
    for model_name in model_name_list:
        merge_analysis_json(model_name=model_name, analysis_dir=analysis_dir, prec="f16")
        merge_analysis_json(model_name=model_name, analysis_dir=analysis_dir, prec="q8_0")
        merge_analysis_json(model_name=model_name, analysis_dir=analysis_dir, prec="q4_k_m")