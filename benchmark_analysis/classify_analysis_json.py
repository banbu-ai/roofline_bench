import os
import shutil

source_dir = os.path.expanduser("~/Code/llm_inference_roofline_detect/analysis/analysis_M1Pro")
# source_dir = os.path.expanduser("~/Code/llm_inference_roofline_detect/analysis/analysis_RTX 3070Ti")
# source_dir = os.path.expanduser("~/Code/llm_inference_roofline_detect/analysis/analysis_Jetson")
target_dir = os.path.join(source_dir, "classified")


# 遍历源文件夹中的所有文件
for filename in os.listdir(source_dir):
    if filename.endswith(".json"):
        # 完整的源文件路径
        source_path = os.path.join(source_dir, filename)

        # 移除文件后缀，方便后续解析
        base_name = filename.replace('.gguf_inference_analysis.json', '')

        # 构造目标文件夹名称
        target_folder_name = ""

        # 尝试通过 '-layers-' 来分割文件名
        if '-layers-' in base_name:
            # 文件名格式: <模型名>-<层数>-layers-<精度>
            parts = base_name.split('-layers-')
            precision = parts[1]

            model_parts = parts[0].split('-')
            # 找到层数的位置（通常是倒数第二个部分）
            # 例如 "Qwen2.5-1.5B-Instruct-20"
            model_name_without_layers = '-'.join(model_parts[:-1])
            target_folder_name = f"{model_name_without_layers}_{precision}"

        if not target_folder_name:
            print(f"警告：无法解析文件名 {filename}，已跳过。")
            continue

        # 完整的目录路径
        destination_folder = os.path.join(target_dir, target_folder_name)

        # 检查目标文件夹是否存在，如果不存在则创建
        if not os.path.exists(destination_folder):
            print(f"目标文件夹不存在，创建中: {destination_folder}")
            os.makedirs(destination_folder)

        # 完整的复制目标路径
        destination_path = os.path.join(destination_folder, filename)

        # 复制文件
        try:
            shutil.copy(source_path, destination_path)
            print(f"成功复制: {filename} -> {destination_folder}")
        except Exception as e:
            print(f"复制 {filename} 失败: {e}")

print("所有文件处理完毕。")