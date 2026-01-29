import os
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import json
from utils.common_utils import CommonUtils
from utils.bandwidth_test_utils import BandwidthTestUtils


chip = "Apple M1 Pro"
model_name_list = ["Qwen2.5-1.5B-Instruct-28-layers", "Llama-3.2-1B-Instruct-21-layers", "PLM-1.8B-Instruct-26-layers", "Qwen3-1.7B-24-layers", "Fox-1-1.6B-28-layers", "SmolLM2-1.7B-Instruct-21-layers"]
analysis_dir = os.path.expanduser("~/Code/llm_inference_roofline_detect/analysis/analysis_size_matched_test_M1Pro")
merged_analysis_dir = os.path.join(analysis_dir, "merged")
output_dir = os.path.join(analysis_dir, "pic")


def create_plot_settings(model_files_dict: Dict[str, List[str]], remove_model_size=True):
    """
    根据模型-文件字典动态生成绘图设置，为每个独特的模型-精度组合分配颜色和标签。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。

    Returns:
        Dict[str, Dict[str, str]]:
            一个字典，键是 "model_name_precision"，值是包含 'color' 和 'label' 的字典。
    """
    settings = {}
    # 定义一个颜色列表，用于循环分配
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'lime', 'brown', 'pink']

    # 使用一个字典来跟踪已分配的唯一组合
    unique_combinations = {}
    color_index = 0

    for model_name, file_list in model_files_dict.items():
        for file_path in file_list:
            base_name = os.path.basename(file_path)
            # 提取精度信息
            if '_f16_' in base_name:
                precision = 'FP16'
            elif '_q8_0_' in base_name:
                precision = 'Q8_0'
            elif '_q4_k_m_' in base_name:
                precision = 'Q4_K_M'
            else:
                precision = 'Unknown'
            # 创建一个唯一的组合键，作为返回字典的键
            combination_key = f"{model_name}_{precision}"

            # 如果这个组合是新的，分配一个新颜色
            if combination_key not in unique_combinations:
                unique_combinations[combination_key] = colors[color_index % len(colors)]
                color_index += 1

            settings[combination_key] = {
                "color": unique_combinations[combination_key],
                "label": f"{CommonUtils.remove_model_size(model_name)} {precision}" if remove_model_size else f"{model_name} {precision}",
                "file_path": file_path  # 添加文件路径，方便后续绘图
            }

    return settings

def create_plot_settings_by_model(model_files_dict: Dict[str, List[str]], remove_model_size=True):
    """
    根据模型-文件字典动态生成绘图设置，为每个模型分配颜色。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。

    Returns:
        Dict[str, Dict[str, str]]:
            一个字典，键是 "model_name_precision"，值是包含 'color', 'label' 和 'file_path' 的字典。
    """
    settings = {}
    colors = ['#E65239',  # 珊瑚红 (Red)
              '#1E6F8C',  # 深蓝 (Dark Blue)
              '#5ABF49',  # 鲜绿 (Vivid Green)
              '#F0B138',  # 金黄色 (Golden Yellow)
              '#8A2BE2',  # 蓝紫色 (Blue Violet)
              '#00CED1',  # 深青色 (Dark Cyan)
              '#FF69B4',  # 亮粉色 (Hot Pink)
              '#8B4513']  # 马鞍棕 (Saddle Brown)

    model_colors = {}
    color_index = 0

    for model_name, file_list in model_files_dict.items():
        if model_name not in model_colors:
            model_colors[model_name] = colors[color_index % len(colors)]
            color_index += 1

        for file_path in file_list:
            base_name = os.path.basename(file_path)

            if '_f16_' in base_name:
                precision = 'FP16'
            elif '_q8_0_' in base_name:
                precision = 'Q8_0'
            elif '_q4_k_m_' in base_name:
                precision = 'Q4_K_M'
            else:
                precision = 'Unknown'

            combination_key = f"{model_name}_{precision}"

            settings[combination_key] = {
                "color": model_colors[model_name],
                "label": f"{CommonUtils.remove_model_size(model_name)} {precision}" if remove_model_size else f"{model_name} {precision}",
                "file_path": file_path
            }
    return settings

def create_plot_settings_by_model_attn_type(model_files_dict: Dict[str, List[str]], remove_model_size=True):
    """
    根据模型-文件字典动态生成绘图设置，通过关键字匹配（in）为模型添加 Attention 类型前缀。
    """
    settings = {}
    # 定义颜色列表
    colors = ['#E65239', '#1E6F8C', '#5ABF49', '#F0B138', '#8A2BE2', '#00CED1', '#FF69B4', '#A52A2A']
    unique_combinations = {}
    color_index = 0

    for model_name, file_list in model_files_dict.items():
        # 使用 in 关键字匹配模型类型，忽略 -28-layers 等后缀
        if any(kw in model_name for kw in ["Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "Qwen3-1.7B", "Fox-1-1.6B"]):
            attn_prefix = "GQA"
        elif "PLM-1.8B-Instruct" in model_name:
            attn_prefix = "MLA"
        elif "SmolLM2-1.7B-Instruct" in model_name:
            attn_prefix = "MHA"
        else:
            attn_prefix = ""

        for file_path in file_list:
            base_name = os.path.basename(file_path)
            # 提取精度信息
            if '_f16_' in base_name:
                precision = 'FP16'
            elif '_q8_0_' in base_name:
                precision = 'Q8_0'
            elif '_q4_k_m_' in base_name:
                precision = 'Q4_K_M'
            else:
                precision = 'Unknown'

            # 创建唯一组合键
            combination_key = f"{model_name}_{precision}"

            # 分配颜色
            if combination_key not in unique_combinations:
                unique_combinations[combination_key] = colors[color_index % len(colors)]
                color_index += 1

            # 生成 Label
            display_name = CommonUtils.remove_model_size(model_name) if remove_model_size else model_name
            label = f"{attn_prefix}: {display_name} {precision}" if attn_prefix else f"{display_name} {precision}"

            settings[combination_key] = {
                "color": unique_combinations[combination_key],
                "label": label,
                "file_path": file_path
            }

    return settings

def create_plot_settings_by_precision(model_files_dict: Dict[str, List[str]], remove_model_size=True):
    """
    根据模型-文件字典动态生成绘图设置，为每种精度分配颜色。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。

    Returns:
        Dict[str, Dict[str, str]]:
            一个字典，键是 "model_name_precision"，值是包含 'color', 'label' 和 'file_path' 的字典。
    """
    settings = {}
    colors = ['#E65239',  # 珊瑚红
              '#1E6F8C',  # 深蓝
              '#5ABF49',  # 鲜绿
              '#F0B138']  # 金黄色

    precision_colors = {}
    color_index = 0

    for model_name, file_list in model_files_dict.items():
        for file_path in file_list:
            base_name = os.path.basename(file_path)

            if '_f16_' in base_name:
                precision = 'FP16'
            elif '_q8_0_' in base_name:
                precision = 'Q8_0'
            elif '_q4_k_m_' in base_name:
                precision = 'Q4_K_M'
            else:
                precision = 'Unknown'

            if precision not in precision_colors:
                precision_colors[precision] = colors[color_index % len(colors)]
                color_index += 1

            combination_key = f"{model_name}_{precision}"

            settings[combination_key] = {
                "color": precision_colors[precision],
                "label": f"{CommonUtils.remove_model_size(model_name)} {precision}" if remove_model_size else f"{model_name} {precision}",
                "file_path": file_path
            }
    return settings

def _plot_roofline_base(ax, chip: str, plot_roofline_curves: bool):
    """
    绘制屋顶线图的公共部分（理论和实测的屋顶线）。

    Args:
        ax (matplotlib.axes.Axes): 绘图的 Axes 对象。
        chip (str): 芯片名称，用于图表标题。
        plot_roofline_curves (bool): 是否绘制屋顶线曲线。
    """
    # 绘制屋顶线所需的基础数据
    theo_bandwidth, theo_peak_flops = BandwidthTestUtils.get_theo_bandwidth_peak_flops_by_chip_name(chip)
    x_range = [(2 ** i) / 3.0 for i in range(-10, 16)]

    if chip == "Apple M1 Pro":
        # M1 Pro
        real_bandwidth = 120 * 10 ** 9
        real_flops = 4.3 * 10 ** 12
    elif chip == "NVIDIA GeForce RTX 3090":
        real_bandwidth = 560.00 * 10 ** 9
        real_flops = 24.28 * 10 ** 12
    elif chip == "NVIDIA GeForce RTX 3070 Ti Laptop GPU":
        # RTX 3070Ti
        real_bandwidth = 220 * 10 ** 9
        real_flops = 9.5 * 10 ** 12
    elif chip == "Jetson Orin Nano Super 8G":
        # Jetson Orin Nano Super 8G
        real_bandwidth = 59.4 * 10 ** 9
        real_flops = 1.34 * 10 ** 12
    elif chip == "Raspberry Pi 5":
        # Raspberry Pi 5
        real_bandwidth = 3.90 * 10 ** 9
        real_flops = 78.50 * 10 ** 9
    else:
        # 默认情况下绘制 M1Pro Roofline
        real_bandwidth = 120 * 10 ** 9
        real_flops = 4.3 * 10 ** 12

    real_ridge_x, _ = BandwidthTestUtils.calculate_ridge_point(real_flops, real_bandwidth)

    if plot_roofline_curves:
        memory_bound_line = [theo_bandwidth * ai for ai in x_range]
        computation_bound_line = [theo_peak_flops for _ in x_range]
        real_memory_bound_line = [real_bandwidth * ai for ai in x_range]
        real_computation_bound_line = [real_flops for _ in x_range]
        # 绘制屋顶线
        ax.loglog(x_range, memory_bound_line, "-", linewidth=2, label="Theoretical Memory Access Bound")
        ax.loglog(x_range, computation_bound_line, "-", linewidth=2, label="Theoretical Computation Bound")
        ax.loglog(x_range, real_memory_bound_line, "--", linewidth=2, label="Measured Memory Access Bound")
        ax.loglog(x_range, real_computation_bound_line, "--", linewidth=2, label="Measured Computation Bound")
        ax.axvline(x=real_ridge_x, linestyle='--', linewidth=1, label=f"Measured Ridge: {real_ridge_x:.2f} FLOPs/Byte")

    # 设置图表标签和标题
    ax.set_xlabel("Operational Intensity (FLOPs/Byte)")
    ax.set_ylabel("Performance (FLOPs/sec)")
    ax.set_title(f"Inference Roofline Test on {chip}")
    ax.grid(True, which="both", ls="--", linewidth=0.5)

    # 调整坐标轴范围
    ax.set_xlim(min(x_range), max(x_range))
    if plot_roofline_curves:
        ax.set_ylim(bottom=1e6, top=theo_peak_flops * 2)

def _categorize_data(data_points: List[Dict]) -> Dict[str, List[Dict]]:
    """
    内部方法，根据推理场景对数据点进行分类。
    """
    categorized_data = {
        'SISO': [], 'SILO': [], 'LISO': [], 'LILO': []
    }
    short_in_threshold = 200
    short_out_threshold = 4000
    for d in data_points:
        n_prompt = d.get('inference_info', {}).get('n_prompt', 0)
        n_gen = d.get('inference_info', {}).get('n_gen', 0)

        if n_prompt <= short_in_threshold and n_gen <= short_out_threshold:
            categorized_data['SISO'].append(d)
        elif n_prompt <= short_in_threshold and n_gen > short_out_threshold:
            categorized_data['SILO'].append(d)
        elif n_prompt > short_in_threshold and n_gen <= short_out_threshold:
            categorized_data['LISO'].append(d)
        elif n_prompt > short_in_threshold and n_gen > short_out_threshold:
            categorized_data['LILO'].append(d)
    return categorized_data

def plot_inference_roofline(model_files_dict: Dict[str, List[str]], output_dir: str, color_type: str, plot_style: str = "full", chip: str = "Apple M1 Pro"):
    """
    为所有模型和精度绘制屋顶线图。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir:
            输出目录，用于保存生成的图像。
        color_type:
            着色类型，'model', 'prec' 或其他。
        plot_style:
            绘图风格，'full'（绘制屋顶线和数据点）或 'points'（仅绘制数据点）。
    """
    os.makedirs(output_dir, exist_ok=True)

    remove_model_size = False if chip == "Raspberry Pi 5" else True

    if color_type == 'model':
        plot_settings = create_plot_settings_by_model(model_files_dict, remove_model_size=remove_model_size)
    elif color_type == 'prec':
        plot_settings = create_plot_settings_by_precision(model_files_dict, remove_model_size=remove_model_size)
    else:
        plot_settings = create_plot_settings(model_files_dict, remove_model_size=remove_model_size)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_roofline_curves = (plot_style == "full")
    _plot_roofline_base(ax, chip, plot_roofline_curves)

    all_oi_values = []
    all_perf_values = []
    for combination_key, settings in plot_settings.items():
        file_path = settings['file_path']
        try:
            with open(file_path, 'r') as f:
                data_points = json.load(f)
            oi_values = [d['operational_intensity'] for d in data_points]
            perf_values = [d['performance'] for d in data_points]
            all_oi_values.extend(oi_values)
            all_perf_values.extend(perf_values)
            ax.loglog(oi_values, perf_values, 'o', markersize=4, color=settings['color'], label=settings['label'])
        except FileNotFoundError:
            print(f"Warning: analysis file {file_path} not found. Skipping data point plotting.")

    # 调整Y轴范围以适应所有数据点
    if not plot_roofline_curves and all_perf_values:
        min_y = min(all_perf_values) * 0.5 if all_perf_values else 1e7
        max_y = max(all_perf_values) * 1.5 if all_perf_values else 1e12
        ax.set_ylim(bottom=min_y, top=max_y)

    # 设置图表标题和图例
    title = f"Benchmark on {chip}"
    # title = f"{chip}"
    ax.set_title(title, fontsize=28)
    ax.legend(loc='lower right')

    # 根据plot_style修改文件名
    file_path = os.path.join(output_dir, title.replace(' ', '_') + ("_roofline" if plot_roofline_curves else "") + ".png")
    plt.savefig(file_path)
    print(f"Save to {file_path}.")
    plt.close(fig)

def plot_inference_roofline_different_attn_type(model_files_dict: Dict[str, List[str]], output_dir: str, color_type: str, plot_style: str = "full", chip: str = "Apple M1 Pro"):
    """
    为所有模型和精度绘制屋顶线图。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir:
            输出目录，用于保存生成的图像。
        color_type:
            着色类型，'model', 'prec' 或其他。
        plot_style:
            绘图风格，'full'（绘制屋顶线和数据点）或 'points'（仅绘制数据点）。
    """
    os.makedirs(output_dir, exist_ok=True)

    remove_model_size = False if chip == "Raspberry Pi 5" else True

    if color_type == 'model':
        plot_settings = create_plot_settings_by_model_attn_type(model_files_dict, remove_model_size=remove_model_size)
    elif color_type == 'prec':
        plot_settings = create_plot_settings_by_precision(model_files_dict, remove_model_size=remove_model_size)
    else:
        plot_settings = create_plot_settings(model_files_dict, remove_model_size=remove_model_size)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_roofline_curves = (plot_style == "full")
    _plot_roofline_base(ax, chip, plot_roofline_curves)

    all_oi_values = []
    all_perf_values = []
    for combination_key, settings in plot_settings.items():
        file_path = settings['file_path']
        try:
            with open(file_path, 'r') as f:
                data_points = json.load(f)
            oi_values = [d['operational_intensity'] for d in data_points]
            perf_values = [d['performance'] for d in data_points]
            all_oi_values.extend(oi_values)
            all_perf_values.extend(perf_values)
            ax.loglog(oi_values, perf_values, 'o', markersize=4, color=settings['color'], label=settings['label'])
        except FileNotFoundError:
            print(f"Warning: analysis file {file_path} not found. Skipping data point plotting.")

    # 调整Y轴范围以适应所有数据点
    if not plot_roofline_curves and all_perf_values:
        min_y = min(all_perf_values) * 0.5 if all_perf_values else 1e7
        max_y = max(all_perf_values) * 1.5 if all_perf_values else 1e12
        ax.set_ylim(bottom=min_y, top=max_y)

    # 设置图表标题和图例
    title = f"Different Attention Type Model Comparison (1.5B)"
    ax.set_title(title, fontsize=20)
    ax.legend(loc='lower right')

    # 根据plot_style修改文件名
    file_path = os.path.join(output_dir, title.replace(' ', '_') + ("_roofline" if plot_roofline_curves else "") + ".png")
    plt.savefig(file_path)
    print(f"Save to {file_path}.")
    plt.close(fig)

def plot_inference_roofline_different_model(model_files_dict: Dict[str, List[str]], output_dir: str, plot_style: str = "full", chip: str = "Apple M1 Pro"):
    """
    为每种模型绘制一个独立的屋顶线图，并在图上展示所有精度的相应数据。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir:
            输出目录，用于保存生成的图像。
        plot_style:
            绘图风格，'full'（绘制屋顶线和数据点）或 'points'（仅绘制数据点）。
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_roofline_curves = (plot_style == "full")

    for model_name, file_list in model_files_dict.items():
        single_model_dict = {model_name: file_list}
        plot_settings = create_plot_settings_by_precision(single_model_dict)

        fig, ax = plt.subplots(figsize=(10, 8))
        _plot_roofline_base(ax, chip, plot_roofline_curves)

        all_perf_values = []
        for combination_key, settings in plot_settings.items():
            file_path = settings['file_path']
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
                oi_values = [d['operational_intensity'] for d in data_points]
                perf_values = [d['performance'] for d in data_points]
                all_perf_values.extend(perf_values)
                ax.loglog(oi_values, perf_values, 'o', markersize=4, color=settings['color'], label=settings['label'])
            except FileNotFoundError:
                print(f"Warning: analysis file {file_path} not found. Skipping data point plotting.")

        # 调整Y轴范围以适应所有数据点
        if not plot_roofline_curves and all_perf_values:
            min_y = min(all_perf_values) * 0.5 if all_perf_values else 1e7
            max_y = max(all_perf_values) * 1.5 if all_perf_values else 1e12
            ax.set_ylim(bottom=min_y, top=max_y)

        # 设置图表标题和图例
        title = f"{CommonUtils.remove_model_size(model_name)} Inference Roofline Test on {chip}"
        ax.set_title(title)
        ax.legend(loc='lower right')

        # 根据plot_style修改文件名
        file_path = os.path.join(output_dir, title.replace(' ', '_') + ("_roofline" if plot_roofline_curves else "") + ".png")
        plt.savefig(file_path)
        print(f"Save to {file_path}.")
        plt.close(fig)

def plot_inference_roofline_different_precision(model_files_dict: Dict[str, List[str]], output_dir: str, plot_style: str = "full", chip: str = "Apple M1 Pro"):
    """
    为每种精度绘制一个独立的屋顶线图，并在图上展示所有模型的相应数据。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir:
            输出目录，用于保存生成的图像。
        plot_style:
            绘图风格，'full'（绘制屋顶线和数据点）或 'points'（仅绘制数据点）。
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_roofline_curves = (plot_style == "full")

    precision_files_dict = {}
    precision_order = {'_f16_': 'FP16', '_q8_0_': 'Q8_0', '_q4_k_m_': 'Q4_K_M'}

    for model_name, file_list in model_files_dict.items():
        for file_path in file_list:
            base_name = os.path.basename(file_path)
            for key, precision in precision_order.items():
                if key in base_name:
                    if precision not in precision_files_dict:
                        precision_files_dict[precision] = {}
                    precision_files_dict[precision][model_name] = file_path
                    break

    for precision, model_data in precision_files_dict.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        _plot_roofline_base(ax, chip, plot_roofline_curves)

        colors = ['#E65239', '#1E6F8C', '#5ABF49', '#F0B138', '#8A2BE2', '#00CED1']
        color_index = 0
        all_perf_values = []

        for model_name, file_path in model_data.items():
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)

                oi_values = [d['operational_intensity'] for d in data_points]
                perf_values = [d['performance'] for d in data_points]
                all_perf_values.extend(perf_values)

                color = colors[color_index % len(colors)]
                label = f"{CommonUtils.remove_model_size(model_name)} {precision} Inference"
                ax.loglog(oi_values, perf_values, 'o', markersize=4, color=color, label=label)
                color_index += 1
            except FileNotFoundError:
                print(f"Warning: analysis file {file_path} not found. Skipping data point plotting.")

        # 调整Y轴范围以适应所有数据点
        if not plot_roofline_curves and all_perf_values:
            min_y = min(all_perf_values) * 0.5 if all_perf_values else 1e7
            max_y = max(all_perf_values) * 1.5 if all_perf_values else 1e12
            ax.set_ylim(bottom=min_y, top=max_y)

        # 设置图表标题和图例
        title = f"Inference Roofline Test - {precision} on {chip}"
        ax.set_title(title)
        ax.legend(loc='lower right')

        # 根据plot_style修改文件名
        file_path = os.path.join(output_dir, title.replace(' ', '_') + ("_roofline" if plot_roofline_curves else "") + ".png")
        plt.savefig(file_path)
        print(f"Save to {file_path}.")
        plt.close(fig)

def plot_inference_roofline_different_scenario(model_files_dict: Dict[str, List[str]], output_dir: str, plot_style: str = "full", chip: str = "Apple M1 Pro"):
    """
    绘制每个模型文件在所有不同推理场景（SISO, SILO, LISO, LILO）下的屋顶线图。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir (str):
            保存图表的目录。
        plot_style (str):
            绘图风格，'full'（绘制屋顶线和数据点）或 'points'（仅绘制数据点）。
    """
    all_scenario_settings = {
        'SISO': {'color': '#1E6F8C', 'label': 'SISO (Short In, Short Out)'},
        'SILO': {'color': '#E65239', 'label': 'SILO (Short In, Long Out)'},
        'LISO': {'color': '#F0B138', 'label': 'LISO (Long In, Short Out)'},
        'LILO': {'color': '#5ABF49', 'label': 'LILO (Long In, Long Out)'}
    }

    os.makedirs(output_dir, exist_ok=True)
    plot_roofline_curves = (plot_style == "full")

    for model_name, file_list in model_files_dict.items():
        for file_path in file_list:
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
            except FileNotFoundError:
                print(f"Error: Analysis file not found at {file_path}. Skipping.")
                continue

            base_name = os.path.basename(file_path)
            if '_f16_' in base_name:
                precision = 'FP16'
            elif '_q8_0_' in base_name:
                precision = 'Q8_0'
            elif '_q4_k_m_' in base_name:
                precision = 'Q4_K_M'
            else:
                precision = 'Unknown Precision'

            categorized_data = _categorize_data(data_points)

            fig, ax = plt.subplots(figsize=(10, 8))
            _plot_roofline_base(ax, chip, plot_roofline_curves)

            all_perf_values = []
            all_oi_values = []

            for scenario_name, settings in all_scenario_settings.items():
                points_to_plot = categorized_data.get(scenario_name, [])
                if points_to_plot:
                    oi_values = [p['operational_intensity'] for p in points_to_plot]
                    perf_values = [p['performance'] for p in points_to_plot]
                    all_perf_values.extend(perf_values)
                    all_oi_values.extend(oi_values)
                    ax.loglog(oi_values, perf_values, 'o', markersize=4, color=settings['color'],
                              label=settings['label'])

            if not plot_roofline_curves and all_perf_values and all_oi_values:
                min_y = min(all_perf_values) * 0.5
                max_y = max(all_perf_values) * 1.5
                ax.set_ylim(bottom=min_y, top=max_y)
                min_x = min(all_oi_values) * 0.5
                max_x = max(all_oi_values) * 1.5
                ax.set_xlim(left=min_x, right=max_x)

            title = f"{CommonUtils.remove_model_size(model_name)} {precision} Different Inference Scenarios on {chip}"
            ax.set_title(title)
            ax.legend(loc='lower right')
            file_path = os.path.join(output_dir, title.replace(' ', '_') + ("_roofline" if plot_roofline_curves else "") + ".png")
            plt.savefig(file_path)
            print(f"Save to {file_path}")
            plt.close(fig)

def plot_inference_roofline_single_scenario(model_files_dict: Dict[str, List[str]], output_dir: str, scenario: str):
    """
    绘制每个模型文件在单一指定推理场景下的数据点图，颜色根据层数渐变。
    图例按层数排序。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir (str):
            保存图表的目录。
        scenario (str):
            要绘制的单一场景名称，如 'SISO', 'SILO', 'LISO', 'LILO'。
    """
    all_scenario_settings = {
        'SISO': {'color': '#1E6F8C', 'label': 'SISO (Short In, Short Out)'},
        'SILO': {'color': '#E65239', 'label': 'SILO (Short In, Long Out)'},
        'LISO': {'color': '#F0B138', 'label': 'LISO (Long In, Short Out)'},
        'LILO': {'color': '#5ABF49', 'label': 'LILO (Long In, Long Out)'}
    }

    if scenario not in all_scenario_settings:
        print(f"Error: Invalid scenario '{scenario}'. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for model_name, file_list in model_files_dict.items():
        for file_path in file_list:
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
            except FileNotFoundError:
                print(f"Error: Analysis file not found at {file_path}. Skipping.")
                continue

            base_name = os.path.basename(file_path)
            if '_f16_' in base_name:
                precision = 'FP16'
            elif '_q8_0_' in base_name:
                precision = 'Q8_0'
            elif '_q4_k_m_' in base_name:
                precision = 'Q4_K_M'
            else:
                precision = 'Unknown Precision'

            categorized_data = _categorize_data(data_points)
            points_to_plot = categorized_data.get(scenario, [])

            if not points_to_plot:
                print(f"No data points found for scenario '{scenario}' in file '{file_path}'. Skipping.")
                continue

            # 根据 layers 值对数据点进行排序
            points_to_plot.sort(key=lambda p: p.get('model_info', {}).get('num_hidden_layers', 0))

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.grid(True, which="both", ls="--", linewidth=0.5)

            all_perf_values = []
            all_oi_values = []

            # 获取层数范围用于颜色映射
            layers = [p.get('model_info', {}).get('num_hidden_layers', 0) for p in points_to_plot]
            if not layers:
                continue
            min_layers = min(layers)
            max_layers = max(layers)

            # 选择一个颜色映射
            cmap = cm.plasma_r
            # 如果只有一个数据点，则直接使用颜色映射的中间颜色
            if min_layers == max_layers:
                colors = [cmap(0.5)] * len(points_to_plot)
            # 否则，标准化层数以获取颜色
            else:
                norm = plt.Normalize(vmin=min_layers, vmax=max_layers)
                colors = [cmap(norm(l)) for l in layers]

            for i, point in enumerate(points_to_plot):
                oi_value = point['operational_intensity']
                perf_value = point['performance']
                all_perf_values.append(perf_value)
                all_oi_values.append(oi_value)

                num_layers = point.get('model_info', {}).get('num_hidden_layers', 'N/A')
                label = f"Layers: {num_layers}"

                ax.plot([oi_value], [perf_value], 'o', markersize=4, color=colors[i], label=label)

            if all_perf_values and all_oi_values:
                min_y = min(all_perf_values) * 0.1
                max_y = max(all_perf_values) * 1.3
                ax.set_ylim(bottom=min_y, top=max_y)
                min_x = min(all_oi_values) * 0.5
                max_x = max(all_oi_values) * 1.5
                ax.set_xlim(left=min_x, right=max_x)

            title = f"{CommonUtils.remove_model_size(model_name)} {precision} {scenario} Scenario on {chip}"
            ax.set_title(title)
            ax.legend(loc='lower right', ncol=2)

            file_path = os.path.join(output_dir, title.replace(' ', '_') + ".png")
            plt.savefig(file_path)
            print(f"Save to {file_path}")
            plt.close(fig)

def plot_inference_roofline_single_scenario_log(model_files_dict: Dict[str, List[str]], output_dir: str, scenario: str):
    """
    绘制每个模型文件在单一指定推理场景下的数据点图，颜色根据层数渐变。
    图例按层数排序。X 轴使用对数刻度（Operational Intensity）。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir (str):
            保存图表的目录。
        scenario (str):
            要绘制的单一场景名称，如 'SISO', 'SILO', 'LISO', 'LILO'。
        plot_style (str):
            绘图风格，'full'（绘制屋顶线和数据点）或 'points'（仅绘制数据点）。
    """
    all_scenario_settings = {
        'SISO': {'color': '#1E6F8C', 'label': 'SISO (Short In, Short Out)'},
        'SILO': {'color': '#E65239', 'label': 'SILO (Short In, Long Out)'},
        'LISO': {'color': '#F0B138', 'label': 'LISO (Long In, Short Out)'},
        'LILO': {'color': '#5ABF49', 'label': 'LILO (Long In, Long Out)'}
    }

    if scenario not in all_scenario_settings:
        print(f"Error: Invalid scenario '{scenario}'. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for model_name, file_list in model_files_dict.items():
        for file_path in file_list:
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
            except FileNotFoundError:
                print(f"Error: Analysis file not found at {file_path}. Skipping.")
                continue

            base_name = os.path.basename(file_path)
            if '_f16_' in base_name:
                precision = 'FP16'
            elif '_q8_0_' in base_name:
                precision = 'Q8_0'
            elif '_q4_k_m_' in base_name:
                precision = 'Q4_K_M'
            else:
                precision = 'Unknown Precision'

            categorized_data = _categorize_data(data_points)
            points_to_plot = categorized_data.get(scenario, [])

            if not points_to_plot:
                print(f"No data points found for scenario '{scenario}' in file '{file_path}'. Skipping.")
                continue

            # 根据 layers 值对数据点进行排序
            points_to_plot.sort(key=lambda p: p.get('model_info', {}).get('num_hidden_layers', 0))

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.grid(True, which="both", ls="--", linewidth=0.5)

            all_perf_values = []
            all_oi_values = []

            # 获取层数范围用于颜色映射
            layers = [p.get('model_info', {}).get('num_hidden_layers', 0) for p in points_to_plot]
            # 过滤掉层数为 0 的点，因为 log 轴不能为 0
            points_to_plot = [p for p in points_to_plot if p.get('model_info', {}).get('num_hidden_layers', 0) > 0]
            layers = [p.get('model_info', {}).get('num_hidden_layers', 0) for p in points_to_plot]

            if not layers:
                continue

            min_layers = min(layers)
            max_layers = max(layers)

            # 选择一个颜色映射
            cmap = cm.plasma_r
            # 如果只有一个数据点，则直接使用颜色映射的中间颜色
            if min_layers == max_layers:
                colors = [cmap(0.5)] * len(points_to_plot)
            # 否则，标准化层数以获取颜色
            else:
                norm = plt.Normalize(vmin=min_layers, vmax=max_layers)
                colors = [cmap(norm(l)) for l in layers]

            for i, point in enumerate(points_to_plot):
                oi_value = point['operational_intensity']
                perf_value = point['performance']
                num_layers = point.get('model_info', {}).get('num_hidden_layers', 'N/A')

                all_perf_values.append(perf_value)
                all_oi_values.append(oi_value)

                label = f"Layers: {num_layers}"

                # 确保层数大于 0 才能在 log 轴上绘图
                if num_layers != 'N/A' and num_layers > 0:
                    ax.plot([oi_value], [perf_value], 'o', markersize=4, color=colors[i], label=label)

            # 设置 X 轴为对数刻度
            # ax.set_xscale('log')
            # 确保 Y 轴为对数刻度
            ax.set_yscale('log')

            if all_perf_values and all_oi_values:
                min_y = min(all_perf_values) * 0.1
                max_y = max(all_perf_values) * 1.3
                ax.set_ylim(bottom=min_y, top=max_y)

                # 调整 X 轴范围以适应所有 OI 数据点
                min_x = min(all_oi_values) * 0.5
                max_x = max(all_oi_values) * 1.5
                ax.set_xlim(left=min_x, right=max_x)

            # --- 修改标题和文件名以保持一致性 ---
            # 标题与 plot_inference_roofline_single_scenario 保持一致
            title = f"{CommonUtils.remove_model_size(model_name)} {precision} {scenario} Scenario on {chip}"
            ax.set_title(title)
            ax.legend(loc='lower right', ncol=2)

            # 文件名与 plot_inference_roofline_single_scenario 保持一致
            file_path = os.path.join(output_dir, title.replace(' ', '_') + ".png")
            plt.savefig(file_path)
            print(f"Save to {file_path}")
            plt.close(fig)

def plot_performance_vs_layers(model_files_dict: Dict[str, List[str]], output_dir: str):
    """
    绘制 Performance (FLOPs/sec) 与 Layers (层数) 的关系图。
    每个图对应一个模型文件（模型+精度），数据点按场景着色。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir (str):
            保存图表的目录。
    """
    all_scenario_settings = {
        'SISO': {'color': '#1E6F8C', 'label': 'SISO (Short In, Short Out)'},
        'SILO': {'color': '#E65239', 'label': 'SILO (Short In, Long Out)'},
        'LISO': {'color': '#F0B138', 'label': 'LISO (Long In, Short Out)'},
        'LILO': {'color': '#5ABF49', 'label': 'LILO (Long In, Long Out)'}
    }

    os.makedirs(output_dir, exist_ok=True)

    for model_name, file_list in model_files_dict.items():
        for file_path in file_list:
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
            except FileNotFoundError:
                print(f"Error: Analysis file not found at {file_path}. Skipping.")
                continue

            base_name = os.path.basename(file_path)
            if '_f16_' in base_name:
                precision = 'FP16'
            elif '_q8_0_' in base_name:
                precision = 'Q8_0'
            elif '_q4_k_m_' in base_name:
                precision = 'Q4_K_M'
            else:
                precision = 'Unknown Precision'

            categorized_data = _categorize_data(data_points)

            fig, ax = plt.subplots(figsize=(10, 8))

            all_perf_values = []
            all_layer_values = []

            for scenario_name, settings in all_scenario_settings.items():
                points_to_plot = categorized_data.get(scenario_name, [])
                if points_to_plot:
                    # 确保数据点按层数排序，以便绘制平滑的折线图 (如果需要)
                    points_to_plot.sort(key=lambda p: p.get('model_info', {}).get('num_hidden_layers', 0))

                    layer_values = [p.get('model_info', {}).get('num_hidden_layers', 0) for p in points_to_plot]
                    perf_values = [p['performance'] for p in points_to_plot]

                    all_layer_values.extend(layer_values)
                    all_perf_values.extend(perf_values)

                    # 使用 plot 绘制带点的线，以便更好地表示趋势
                    ax.plot(layer_values, perf_values, 'o-', markersize=4, color=settings['color'],
                            label=settings['label'])

            if all_perf_values and all_layer_values:
                # 调整Y轴使用对数刻度，因为性能数据通常跨越几个数量级
                ax.set_yscale('log')
                min_y = min(all_perf_values) * 0.5
                max_y = max(all_perf_values) * 1.5
                ax.set_ylim(bottom=min_y, top=max_y)

                # X轴使用线性刻度
                min_x = 0
                max_x = max(all_layer_values) * 1.05 if all_layer_values else 20
                ax.set_xlim(left=min_x, right=max_x)

            # 设置图表标签和标题
            ax.set_xlabel("Number of Layers")
            ax.set_ylabel("Performance (FLOPs/sec)")
            title = f"{CommonUtils.remove_model_size(model_name)} {precision} Performance vs. Layers on {chip}"
            ax.set_title(title)
            ax.grid(True, which="both", ls="--", linewidth=0.5)
            ax.legend(loc='lower right')

            file_path = os.path.join(output_dir, title.replace(' ', '_') + ".png")
            plt.savefig(file_path)
            print(f"Save to {file_path}")
            plt.close(fig)

def plot_oi_vs_layers(model_files_dict: Dict[str, List[str]], output_dir: str):
    """
    绘制 Operational Intensity (FLOPs/Byte) 与 Layers (层数) 的关系图。
    每个图对应一个模型文件（模型+精度），数据点按场景着色。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir (str):
            保存图表的目录。
    """
    all_scenario_settings = {
        'SISO': {'color': '#1E6F8C', 'label': 'SISO (Short In, Short Out)'},
        'SILO': {'color': '#E65239', 'label': 'SILO (Short In, Long Out)'},
        'LISO': {'color': '#F0B138', 'label': 'LISO (Long In, Short Out)'},
        'LILO': {'color': '#5ABF49', 'label': 'LILO (Long In, Long Out)'}
    }

    os.makedirs(output_dir, exist_ok=True)

    for model_name, file_list in model_files_dict.items():
        for file_path in file_list:
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
            except FileNotFoundError:
                print(f"Error: Analysis file not found at {file_path}. Skipping.")
                continue

            base_name = os.path.basename(file_path)
            if '_f16_' in base_name:
                precision = 'FP16'
            elif '_q8_0_' in base_name:
                precision = 'Q8_0'
            elif '_q4_k_m_' in base_name:
                precision = 'Q4_K_M'
            else:
                precision = 'Unknown Precision'

            categorized_data = _categorize_data(data_points)

            fig, ax = plt.subplots(figsize=(10, 8))

            all_oi_values = []
            all_layer_values = []

            for scenario_name, settings in all_scenario_settings.items():
                points_to_plot = categorized_data.get(scenario_name, [])
                if points_to_plot:
                    # 确保数据点按层数排序
                    points_to_plot.sort(key=lambda p: p.get('model_info', {}).get('num_hidden_layers', 0))

                    layer_values = [p.get('model_info', {}).get('num_hidden_layers', 0) for p in points_to_plot]
                    oi_values = [p['operational_intensity'] for p in points_to_plot]

                    all_layer_values.extend(layer_values)
                    all_oi_values.extend(oi_values)

                    # 使用 loglog 绘制带点的线，因为 OI 通常在屋顶线图中是 log 轴
                    ax.loglog(layer_values, oi_values, 'o-', markersize=4, color=settings['color'],
                            label=settings['label'])

            if all_oi_values and all_layer_values:
                # Y轴使用对数刻度 (Operational Intensity)
                min_y = min(all_oi_values) * 0.5 if all_oi_values else 0.1
                max_y = max(all_oi_values) * 1.5 if all_oi_values else 100
                ax.set_ylim(bottom=min_y, top=max_y)

                # X轴使用对数刻度或线性刻度。考虑到层数通常是小整数，这里尝试使用线性刻度
                ax.set_xscale('linear')
                min_x = 0
                max_x = max(all_layer_values) * 1.05 if all_layer_values else 20
                ax.set_xlim(left=min_x, right=max_x)

            # 设置图表标签和标题
            ax.set_xlabel("Number of Layers")
            ax.set_ylabel("Operational Intensity (FLOPs/Byte)")
            title = f"{CommonUtils.remove_model_size(model_name)} {precision} OI vs. Layers on {chip}"
            ax.set_title(title)
            ax.grid(True, which="both", ls="--", linewidth=0.5)
            ax.legend(loc='lower right')

            file_path = os.path.join(output_dir, title.replace(' ', '_') + ".png")
            plt.savefig(file_path)
            print(f"Save to {file_path}")
            plt.close(fig)

def plot_flops_vs_runtime(model_files_dict: Dict[str, List[str]], output_dir: str):
    """
    绘制 Total FLOPs (总浮点运算次数) 与 Runtime (运行时间) 的关系图。
    每个图对应一个模型文件（模型+精度），数据点按场景着色。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir (str):
            保存图表的目录。
    """
    all_scenario_settings = {
        'SISO': {'color': '#1E6F8C', 'label': 'SISO (Short In, Short Out)'},
        'SILO': {'color': '#E65239', 'label': 'SILO (Short In, Long Out)'},
        'LISO': {'color': '#F0B138', 'label': 'LISO (Long In, Short Out)'},
        'LILO': {'color': '#5ABF49', 'label': 'LILO (Long In, Long Out)'}
    }

    os.makedirs(output_dir, exist_ok=True)

    for model_name, file_list in model_files_dict.items():
        for file_path in file_list:
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
            except FileNotFoundError:
                print(f"Error: Analysis file not found at {file_path}. Skipping.")
                continue

            base_name = os.path.basename(file_path)
            if '_f16_' in base_name:
                precision = 'FP16'
            elif '_q8_0_' in base_name:
                precision = 'Q8_0'
            elif '_q4_k_m_' in base_name:
                precision = 'Q4_K_M'
            else:
                precision = 'Unknown Precision'

            categorized_data = _categorize_data(data_points)

            fig, ax = plt.subplots(figsize=(10, 8))

            all_flops_values = []
            all_runtime_values = []

            for scenario_name, settings in all_scenario_settings.items():
                points_to_plot = categorized_data.get(scenario_name, [])
                if points_to_plot:
                    # 确保数据点按 FLOPs 排序，以帮助趋势可视化
                    points_to_plot.sort(key=lambda p: p['total_flops'])

                    flops_values = [p['total_flops'] for p in points_to_plot]
                    runtime_values = [p['runtime'] for p in points_to_plot]

                    all_flops_values.extend(flops_values)
                    all_runtime_values.extend(runtime_values)

                    # 使用 plot 绘制带点的线
                    ax.plot(flops_values, runtime_values, 'o-', markersize=4, color=settings['color'],
                            label=settings['label'])

            if all_flops_values and all_runtime_values:
                # 调整 X 轴 (FLOPs) 和 Y 轴 (Runtime) 使用对数刻度
                ax.set_xscale('log')
                ax.set_yscale('log')

                min_x = min(all_flops_values) * 0.5
                max_x = max(all_flops_values) * 1.5
                ax.set_xlim(left=min_x, right=max_x)

                min_y = min(all_runtime_values) * 0.5
                max_y = max(all_runtime_values) * 1.5
                ax.set_ylim(bottom=min_y, top=max_y)

            # 设置图表标签和标题
            ax.set_xlabel("Total FLOPs")
            ax.set_ylabel("Runtime (seconds)")
            title = f"{CommonUtils.remove_model_size(model_name)} {precision} FLOPs vs. Runtime on {chip}"
            ax.set_title(title)
            ax.grid(True, which="both", ls="--", linewidth=0.5)
            ax.legend(loc='upper left')

            file_path = os.path.join(output_dir, title.replace(' ', '_') + ".png")
            plt.savefig(file_path)
            print(f"Save to {file_path}")
            plt.close(fig)

def plot_layers_vs_runtime(model_files_dict: Dict[str, List[str]], output_dir: str):
    """
    绘制 Layers (层数) 与 Runtime (运行时间) 的关系图。
    每个图对应一个模型文件（模型+精度），数据点按场景着色。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir (str):
            保存图表的目录。
    """
    all_scenario_settings = {
        'SISO': {'color': '#1E6F8C', 'label': 'SISO (Short In, Short Out)'},
        'SILO': {'color': '#E65239', 'label': 'SILO (Short In, Long Out)'},
        'LISO': {'color': '#F0B138', 'label': 'LISO (Long In, Short Out)'},
        'LILO': {'color': '#5ABF49', 'label': 'LILO (Long In, Long Out)'}
    }

    os.makedirs(output_dir, exist_ok=True)

    for model_name, file_list in model_files_dict.items():
        for file_path in file_list:
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
            except FileNotFoundError:
                print(f"Error: Analysis file not found at {file_path}. Skipping.")
                continue

            base_name = os.path.basename(file_path)
            if '_f16_' in base_name:
                precision = 'FP16'
            elif '_q8_0_' in base_name:
                precision = 'Q8_0'
            elif '_q4_k_m_' in base_name:
                precision = 'Q4_K_M'
            else:
                precision = 'Unknown Precision'

            categorized_data = _categorize_data(data_points)

            fig, ax = plt.subplots(figsize=(10, 8))

            all_layer_values = []
            all_runtime_values = []

            for scenario_name, settings in all_scenario_settings.items():
                points_to_plot = categorized_data.get(scenario_name, [])
                if points_to_plot:
                    # 确保数据点按层数排序
                    points_to_plot.sort(key=lambda p: p.get('model_info', {}).get('num_hidden_layers', 0))

                    layer_values = [p.get('model_info', {}).get('num_hidden_layers', 0) for p in points_to_plot]
                    runtime_values = [p['runtime'] for p in points_to_plot]

                    all_layer_values.extend(layer_values)
                    all_runtime_values.extend(runtime_values)

                    # 使用 plot 绘制带点的线
                    ax.plot(layer_values, runtime_values, 'o-', markersize=4, color=settings['color'],
                            label=settings['label'])

            if all_layer_values and all_runtime_values:
                # X 轴 (Layers) 使用线性刻度
                ax.set_xscale('linear')
                min_x = 0
                max_x = max(all_layer_values) * 1.05 if all_layer_values else 20
                ax.set_xlim(left=min_x, right=max_x)

                # Y 轴 (Runtime) 使用对数刻度
                ax.set_yscale('log')
                min_y = min(all_runtime_values) * 0.5
                max_y = max(all_runtime_values) * 1.5
                ax.set_ylim(bottom=min_y, top=max_y)

            # 设置图表标签和标题
            ax.set_xlabel("Number of Layers")
            ax.set_ylabel("Runtime (seconds)")
            title = f"{CommonUtils.remove_model_size(model_name)} {precision} Layers vs. Runtime on {chip}"
            ax.set_title(title)
            ax.grid(True, which="both", ls="--", linewidth=0.5)
            ax.legend(loc='upper left')

            file_path = os.path.join(output_dir, title.replace(' ', '_') + ".png")
            plt.savefig(file_path)
            print(f"Save to {file_path}")
            plt.close(fig)

def plot_flops_vs_layers(model_files_dict: Dict[str, List[str]], output_dir: str):
    """
    绘制 Total FLOPs (总浮点运算次数) 与 Layers (层数) 的关系图。
    每个图对应一个模型文件（模型+精度），数据点按场景着色。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir (str):
            保存图表的目录。
    """
    all_scenario_settings = {
        'SISO': {'color': '#1E6F8C', 'label': 'SISO (Short In, Short Out)'},
        'SILO': {'color': '#E65239', 'label': 'SILO (Short In, Long Out)'},
        'LISO': {'color': '#F0B138', 'label': 'LISO (Long In, Short Out)'},
        'LILO': {'color': '#5ABF49', 'label': 'LILO (Long In, Long Out)'}
    }

    os.makedirs(output_dir, exist_ok=True)

    for model_name, file_list in model_files_dict.items():
        for file_path in file_list:
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
            except FileNotFoundError:
                print(f"Error: Analysis file not found at {file_path}. Skipping.")
                continue

            base_name = os.path.basename(file_path)
            if '_f16_' in base_name:
                precision = 'FP16'
            elif '_q8_0_' in base_name:
                precision = 'Q8_0'
            elif '_q4_k_m_' in base_name:
                precision = 'Q4_K_M'
            else:
                precision = 'Unknown Precision'

            categorized_data = _categorize_data(data_points)

            fig, ax = plt.subplots(figsize=(10, 8))

            all_flops_values = []
            all_layer_values = []

            for scenario_name, settings in all_scenario_settings.items():
                points_to_plot = categorized_data.get(scenario_name, [])
                if points_to_plot:
                    # 确保数据点按层数排序
                    points_to_plot.sort(key=lambda p: p.get('model_info', {}).get('num_hidden_layers', 0))

                    layer_values = [p.get('model_info', {}).get('num_hidden_layers', 0) for p in points_to_plot]
                    flops_values = [p['total_flops'] for p in points_to_plot]

                    all_layer_values.extend(layer_values)
                    all_flops_values.extend(flops_values)

                    # 使用 plot 绘制带点的线
                    ax.plot(layer_values, flops_values, 'o-', markersize=4, color=settings['color'],
                            label=settings['label'])

            if all_flops_values and all_layer_values:
                # X 轴 (Layers) 使用线性刻度
                ax.set_xscale('linear')
                min_x = 0
                max_x = max(all_layer_values) * 1.05 if all_layer_values else 20
                ax.set_xlim(left=min_x, right=max_x)

                # Y 轴 (FLOPs) 使用对数刻度
                ax.set_yscale('log')
                min_y = min(all_flops_values) * 0.5
                max_y = max(all_flops_values) * 1.5
                ax.set_ylim(bottom=min_y, top=max_y)

            # 设置图表标签和标题
            ax.set_xlabel("Number of Layers")
            ax.set_ylabel("Total FLOPs")
            title = f"{CommonUtils.remove_model_size(model_name)} {precision} FLOPs vs. Layers on {chip}"
            ax.set_title(title)
            ax.grid(True, which="both", ls="--", linewidth=0.5)
            ax.legend(loc='lower right')

            file_path = os.path.join(output_dir, title.replace(' ', '_') + ".png")
            plt.savefig(file_path)
            print(f"Save to {file_path}")
            plt.close(fig)

def plot_inference_roofline_subplot(
        model_files_dict: Dict[str, List[str]],
        output_dir: str,
        color_type: str,
        suptitle_title: str = "Different Model Precision Comparison",
        subplot_layout: Tuple[int, int] = (2, 3),
        plot_style: str = "full"
):
    """
    为所有模型和精度绘制屋顶线图，支持单图或子图模式。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir:
            输出目录，用于保存生成的图像。
        color_type:
            着色类型，'model', 'prec' 或其他。
        suptitle_title (str):
            总图的标题 (suptitle)。
        subplot_layout (Tuple[int, int]):
            子图的行数和列数，例如 (2, 2)。仅在 `draw_subplot` 为 True 时有效。
        plot_style:
            绘图风格，'full'（绘制屋顶线和数据点）或 'points'（仅绘制数据点）。
    """
    os.makedirs(output_dir, exist_ok=True)

    if color_type == 'model':
        plot_settings = create_plot_settings_by_model(model_files_dict)
    elif color_type == 'prec':
        plot_settings = create_plot_settings_by_precision(model_files_dict)
    else:
        plot_settings = create_plot_settings(model_files_dict)

    plot_roofline_curves = (plot_style == "full")

    # 子图模式
    n_rows, n_cols = subplot_layout
    unique_models = list(model_files_dict.keys())
    n_subplots = len(unique_models)

    # 调整子图数量以适应布局
    if n_subplots > n_rows * n_cols:
        print(
            f"Warning: Number of models ({n_subplots}) exceeds subplot layout capacity ({n_rows * n_cols}). Truncating to fit layout.")
        unique_models = unique_models[:n_rows * n_cols]
        n_subplots = len(unique_models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    fig.suptitle(suptitle_title, fontsize=40)
    axes = axes.flatten()

    # 为每个子图分配数据点
    model_plots = {model: {'oi': [], 'perf': [], 'settings': []} for model in unique_models}

    for combination_key, settings in plot_settings.items():
        model_name = combination_key.split('_')[0]  # 假设组合键格式为 'model_prec'
        if model_name in model_plots:
            file_path = settings['file_path']
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
                model_plots[model_name]['oi'].extend([d['operational_intensity'] for d in data_points])
                model_plots[model_name]['perf'].extend([d['performance'] for d in data_points])
                model_plots[model_name]['settings'].append(settings)
            except FileNotFoundError:
                print(
                    f"Warning: analysis file {file_path} not found for {model_name}. Skipping data point plotting.")

    all_oi_values = []
    all_perf_values = []

    for i, model_name in enumerate(unique_models):
        ax = axes[i]
        data = model_plots[model_name]
        _plot_roofline_base(ax, chip, plot_roofline_curves)

        # 绘制当前模型的所有精度点
        current_oi = []
        current_perf = []
        for settings in data['settings']:
            # 重新加载数据以确保只绘制属于当前settings的数据（如果 data['oi'] / data['perf'] 是所有数据的集合）
            # 这里我们假设 plot_settings 中的 settings['label'] 足够细致
            file_path = settings['file_path']
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
                oi_values = [d['operational_intensity'] for d in data_points]
                perf_values = [d['performance'] for d in data_points]

                ax.loglog(oi_values, perf_values, 'o', markersize=4, color=settings['color'],
                          label=settings['label'])
                current_oi.extend(oi_values)
                current_perf.extend(perf_values)
            except FileNotFoundError:
                pass  # 警告已在上面打印

        ax.set_title(f"Model: {model_name}")
        ax.legend(loc='lower right', fontsize='small')

        all_oi_values.extend(current_oi)
        all_perf_values.extend(current_perf)

        # 调整Y轴范围以适应当前子图的数据点
        if not plot_roofline_curves and current_perf:
            min_y = min(current_perf) * 0.5
            max_y = max(current_perf) * 1.5
            ax.set_ylim(bottom=min_y, top=max_y)

    # 隐藏多余的子图
    for j in range(n_subplots, n_rows * n_cols):
        fig.delaxes(axes[j])

    fig.tight_layout(rect=(0, 0.03, 1, 0.97))  # 调整布局以留出 suptitle 空间
    file_path = os.path.join(output_dir, suptitle_title.replace(' ', '_') + (
        "_roofline" if plot_roofline_curves else "") + "_subplot" + ".png")

    plt.savefig(file_path)
    print(f"Save to {file_path}.")
    plt.close(fig)

def plot_inference_roofline_different_scenario_subplot(
        model_files_dict: Dict[str, List[str]],
        output_dir: str,
        suptitle_title: str = "Different Scenario Comparison",
        subplot_layout: Tuple[int, int] = (2, 3),
        plot_style: str = "full"
):
    """
    绘制每个模型文件在所有不同推理场景（SISO, SILO, LISO, LILO）下的屋顶线图，
    按精度分组，并在子图中展示不同模型的性能。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir (str):
            保存图表的目录。
        suptitle_title (str):
            总图的标题 (suptitle)。
        subplot_layout (Tuple[int, int]):
            子图的行数和列数，用于在同一精度下绘制不同模型。
        plot_style (str):
            绘图风格，'full'（绘制屋顶线和数据点）或 'points'（仅绘制数据点）。
    """
    all_scenario_settings = {
        'SISO': {'color': '#1E6F8C', 'label': 'SISO (Short In, Short Out)'},
        'SILO': {'color': '#E65239', 'label': 'SILO (Short In, Long Out)'},
        'LISO': {'color': '#F0B138', 'label': 'LISO (Long In, Short Out)'},
        'LILO': {'color': '#5ABF49', 'label': 'LILO (Long In, Long Out)'}
    }

    os.makedirs(output_dir, exist_ok=True)
    plot_roofline_curves = (plot_style == "full")

    # 1. 收集和按精度分组数据
    data_by_precision: Dict[str, List[Dict]] = {}
    precision_mapping = {
        '_f16_': 'FP16',
        '_q8_0_': 'Q8_0',
        '_q4_k_m_': 'Q4_K_M'
    }

    for model_name, file_list in model_files_dict.items():
        for file_path in file_list:
            base_name = os.path.basename(file_path)
            precision = 'Unknown Precision'
            for key, prec_name in precision_mapping.items():
                if key in base_name:
                    precision = prec_name
                    break

            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
            except FileNotFoundError:
                print(f"Error: Analysis file not found at {file_path}. Skipping file {file_path}.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in file {file_path}. Skipping.")
                continue

            if precision not in data_by_precision:
                data_by_precision[precision] = []

            data_by_precision[precision].append({
                'model_name': model_name,
                'file_path': file_path,
                'categorized_data': _categorize_data(data_points)
            })

    # 2. 遍历每个精度组并创建子图
    n_rows, n_cols = subplot_layout

    for precision, model_data_list in data_by_precision.items():
        n_models = len(model_data_list)

        # 确定实际需要的子图数量和布局
        if n_models == 0:
            continue

        n_subplots = min(n_models, n_rows * n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))

        # 如果只有一个子图，plt.subplots 返回单个 Axes 对象
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif isinstance(axes, plt.Axes):
            axes = [axes]  # handles case of 1xN or Nx1 returning a 1D array
        else:
            axes = axes.flatten()

        fig.suptitle(f"{suptitle_title} - Precision: {precision}", fontsize=40)

        global_all_perf_values = []
        global_all_oi_values = []

        # 3. 在子图上绘制每个模型的场景数据
        for i in range(n_subplots):
            data_entry = model_data_list[i]
            model_name = data_entry['model_name']
            categorized_data = data_entry['categorized_data']
            ax = axes[i]

            _plot_roofline_base(ax, chip, plot_roofline_curves)

            # 使用 CommonUtils.remove_model_size 简化子图标题
            simple_model_name = CommonUtils.remove_model_size(model_name)
            ax.set_title(f"Model: {simple_model_name}")

            local_perf_values = []
            local_oi_values = []

            for scenario_name, settings in all_scenario_settings.items():
                points_to_plot = categorized_data.get(scenario_name, [])
                if points_to_plot:
                    oi_values = [p['operational_intensity'] for p in points_to_plot]
                    perf_values = [p['performance'] for p in points_to_plot]
                    local_perf_values.extend(perf_values)
                    local_oi_values.extend(oi_values)

                    # 仅在第一个子图上绘制图例，或在循环结束后统一处理
                    # 这里选择在每个子图上绘制，便于查看
                    ax.loglog(oi_values, perf_values, 'o', markersize=4, color=settings['color'],
                              label=settings['label'])

            global_all_perf_values.extend(local_perf_values)
            global_all_oi_values.extend(local_oi_values)

            # 调整当前子图的坐标轴范围 (如果不需要屋顶线)
            if not plot_roofline_curves and local_perf_values and local_oi_values:
                min_y = min(local_perf_values) * 0.5
                max_y = max(local_perf_values) * 1.5
                ax.set_ylim(bottom=min_y, top=max_y)
                min_x = min(local_oi_values) * 0.5
                max_x = max(local_oi_values) * 1.5
                ax.set_xlim(left=min_x, right=max_x)

            # 设置图例。如果子图过多，可能需要精简图例
            ax.legend(loc='lower right', fontsize='small')

        # 隐藏多余的子图
        for j in range(n_models, n_rows * n_cols):
            if j < len(axes):  # 确保索引不越界
                fig.delaxes(axes[j])

        # 保存整个图
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))  # 调整布局以留出 suptitle 空间
        file_name = f"{suptitle_title.replace(' ', '_')}_{precision}" + (
            "_roofline" if plot_roofline_curves else "" ) + "_subplot" + ".png"
        file_path = os.path.join(output_dir, file_name)
        plt.savefig(file_path)
        print(f"Save to {file_path}")
        plt.close(fig)

def plot_inference_roofline_single_scenario_subplot(
        model_files_dict: Dict[str, List[str]],
        output_dir: str,
        scenario: str,
        suptitle_title: str = "Inference Roofline Analysis",
        subplot_layout: Tuple[int, int] = (2, 2),
        perf_log: bool = False
):
    """
    绘制每个模型文件在指定场景或所有场景下的屋顶线子图，
    子图按精度和/或场景划分。所有子图共享一个总图例。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir (str):
            保存图表的目录。
        scenario (str):
            要绘制的单一场景名称，如 'SISO', 'SILO', 'LISO', 'LILO'，或 'all'。
        suptitle_title (str):
            总图的标题 (suptitle)。
        subplot_layout (Tuple[int, int]):
            子图的行数和列数，用于绘制同一模型下的不同精度/场景组合。
        perf_log (bool):
            是否对性能轴（Y轴）使用对数刻度。
    """
    all_scenario_settings = {
        'SISO': {'color': '#1E6F8C', 'label': 'SISO (Short In, Short Out)'},
        'SILO': {'color': '#E65239', 'label': 'SILO (Short In, Long Out)'},
        'LISO': {'color': '#F0B138', 'label': 'LISO (Long In, Short Out)'},
        'LILO': {'color': '#5ABF49', 'label': 'LILO (Long In, Long Out)'}
    }

    target_scenarios = list(all_scenario_settings.keys()) if scenario == 'all' else [scenario]

    if not all(s in all_scenario_settings or s == 'all' for s in target_scenarios):
        print(
            f"Error: Invalid scenario '{scenario}'. Must be one of {list(all_scenario_settings.keys())} or 'all'. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    cmap = cm.plasma_r

    # 1. 收集和按模型/精度/场景分组数据
    model_data_map: Dict[str, Dict[Tuple[str, str], List[Dict]]] = {}
    precision_mapping = {
        '_f16_': 'FP16',
        '_q8_0_': 'Q8_0',
        '_q4_k_m_': 'Q4_K_M'
    }

    # *** 关键修改 1: 定义目标精度顺序 ***
    # 按照用户要求：第一行是FP16、第二行是Q8_0、第三行是Q4_K_M
    target_precision_order = ['FP16', 'Q8_0', 'Q4_K_M']
    # 场景顺序保持不变，即按字母顺序或手动定义
    target_scenario_order = sorted(target_scenarios)  # 假设按字母顺序排列 S L I O...

    for model_name, file_list in model_files_dict.items():
        if model_name not in model_data_map:
            model_data_map[model_name] = {}

        for file_path in file_list:
            base_name = os.path.basename(file_path)
            precision = 'Unknown Precision'
            for key, prec_name in precision_mapping.items():
                if key in base_name:
                    precision = prec_name
                    break

            try:
                # 假设您的 JSON 文件只包含列表结构
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Error: Analysis file not found or invalid JSON at {file_path}. Skipping.")
                continue

            # 假设 _categorize_data 返回 {'scenario_name': [data_points]}
            categorized_data = _categorize_data(data_points)

            # 存储 (precision, scenario) -> data
            for s in target_scenarios:
                points_to_plot = categorized_data.get(s, [])
                if points_to_plot:
                    model_data_map[model_name][(precision, s)] = points_to_plot

    # 2. 遍历每个模型并创建子图
    # 这里的 subplot_layout 应该适配您的最终布局，例如 (3, 4)
    n_rows, n_cols = subplot_layout

    for model_name, subplot_data in model_data_map.items():

        # *** 关键修改 2: 自定义排序逻辑 ***
        # 1. 获取所有键
        all_keys = list(subplot_data.keys())

        # 2. 创建一个基于目标精度的自定义排序键
        def custom_sort_key(item: Tuple[str, str]):
            precision, scenario = item
            # 精度排序索引
            try:
                prec_index = target_precision_order.index(precision)
            except ValueError:
                # 不在目标列表中的精度放在最后
                prec_index = len(target_precision_order)

                # 场景排序索引 (使用目标顺序的索引)
            try:
                scenario_index = target_scenario_order.index(scenario)
            except ValueError:
                scenario_index = len(target_scenario_order)

            # 排序优先级：先按精度行 (prec_index)，再按场景列 (scenario_index)
            return (prec_index, scenario_index)

        sorted_keys = sorted(all_keys, key=custom_sort_key)

        n_subplots = len(sorted_keys)

        if n_subplots == 0:
            continue

        # 调整布局以适应实际子图数量
        # 这里使用用户传入的 subplot_layout，如果实际子图数量超出，则需要重新计算
        # rows = min(n_rows, math.ceil(n_subplots / n_cols))
        # cols = min(n_cols, n_subplots if rows == 1 else n_cols)
        # 假设 subplot_layout 已经是准确的，或者 n_subplots <= n_rows * n_cols
        rows, cols = subplot_layout

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

        if n_subplots == 1:
            axes = [axes]
        elif isinstance(axes, plt.Axes):
            axes = [axes]
        else:
            axes = axes.flatten()

        simple_model_name = CommonUtils.remove_model_size(model_name)
        # 增大标题字体
        fig.suptitle(f"{suptitle_title} - Model: {simple_model_name}", fontsize=40)

        # 用于收集所有子图的图例句柄和标签
        all_legend_handles = []
        all_legend_labels = []
        unique_layers_plotted = set()  # 记录已经添加图例的层数，避免重复

        # 3. 在子图上绘制每个 (精度, 场景) 组合
        for i, (precision, s) in enumerate(sorted_keys):
            if i >= rows * cols:
                break  # 超过子图布局，跳过

            ax = axes[i]
            ax.grid(True, which="both", ls="--", linewidth=0.5)
            points_to_plot = subplot_data[(precision, s)]

            # 根据 layers 值对数据点进行排序，用于颜色映射
            points_to_plot.sort(key=lambda p: p.get('model_info', {}).get('num_hidden_layers', 0))

            all_perf_values = []
            all_oi_values = []

            layers = [p.get('model_info', {}).get('num_hidden_layers', 0) for p in points_to_plot]
            min_layers = min(layers) if layers else 0
            max_layers = max(layers) if layers else 0

            # 准备颜色
            if min_layers == max_layers:
                colors = [cmap(0.5)] * len(points_to_plot)
            else:
                norm = plt.Normalize(vmin=min_layers, vmax=max_layers)
                colors = [cmap(norm(l)) for l in layers]

            for j, point in enumerate(points_to_plot):
                oi_value = point['operational_intensity']
                perf_value = point['performance']
                all_perf_values.append(perf_value)
                all_oi_values.append(oi_value)

                num_layers = point.get('model_info', {}).get('num_hidden_layers', 'N/A')
                label = f"Layers: {num_layers}"

                # 绘制点
                line, = ax.plot([oi_value], [perf_value], 'o', markersize=4, color=colors[j])  # 不在这里设置 label

                # 收集用于总图例的句柄和标签
                if num_layers not in unique_layers_plotted:
                    all_legend_handles.append(line)
                    all_legend_labels.append(label)
                    unique_layers_plotted.add(num_layers)

            # 调整当前子图的坐标轴范围
            if all_perf_values and all_oi_values:
                # 确保Y轴最低值合理
                min_y = min(all_perf_values) * 0.1
                max_y = max(all_perf_values) * 1.3
                ax.set_ylim(bottom=min_y, top=max_y)
                # 确保X轴最低值合理
                min_x = min(all_oi_values) * 0.5
                max_x = max(all_oi_values) * 1.5
                ax.set_xlim(left=min_x, right=max_x)

            if perf_log:
                ax.set_yscale('log')

            # 设置子图标题
            ax.set_title(f"{precision} | {s}")

            # 添加轴标签
            if i % cols == 0:
                ax.set_ylabel("Performance", fontsize=16)
            if i >= (rows - 1) * cols:
                ax.set_xlabel("Operational Intensity (Flops/Byte)", fontsize=16)

        # 4. 隐藏多余的子图
        for j in range(n_subplots, rows * cols):
            if j < len(axes):
                fig.delaxes(axes[j])

        # 5. 在总图上创建统一图例
        # 对收集到的图例进行排序，确保图例顺序是按层数从小到大
        sorted_legend_items = []
        for handle, label in zip(all_legend_handles, all_legend_labels):
            try:
                # 提取层数作为排序键
                layers_val = int(label.split(':')[1].strip())
                sorted_legend_items.append((layers_val, handle, label))
            except (ValueError, IndexError):
                # 如果标签格式不匹配，则放到最后
                sorted_legend_items.append((float('inf'), handle, label))

        sorted_legend_items.sort(key=lambda x: x[0])
        final_handles = [item[1] for item in sorted_legend_items]
        final_labels = [item[2] for item in sorted_legend_items]

        # 图例放置在右侧
        fig.legend(handles=final_handles, labels=final_labels,
                   loc='center left',  # 将图例的左中央对齐到 bbox_to_anchor
                   ncol=1,  # 垂直排列
                   bbox_to_anchor=(0.95, 0.5),  # 放置在 Figure 右侧中央 (X=1.0)
                   fontsize=20)  # 减小字体以适应空间

        # 6. 保存整个图
        # 为右侧图例腾出空间
        fig.tight_layout(rect=(0, 0.0, 0.95, 0.95))  # 调整 rect 的 right 值为 0.95

        file_suffix = scenario if scenario != 'all' else 'all_scenarios'
        file_name = f"{simple_model_name.replace(' ', '_')}_{file_suffix}_subplot" + ".png"
        file_path = os.path.join(output_dir, file_name)

        plt.savefig(file_path, bbox_inches='tight')  # 使用 bbox_inches='tight' 确保图例不被裁剪
        print(f"Save to {file_path}")
        plt.close(fig)


if __name__ == "__main__":
    model_files_dict = CommonUtils.find_files_by_model(model_name_list, merged_analysis_dir)
    CommonUtils.sort_files_by_precision(model_files_dict)

    # Roofline
    roofline_output_dir = os.path.join(output_dir, "roofline")
    # plot_inference_roofline(model_files_dict=model_files_dict, output_dir=roofline_output_dir, color_type="model", plot_style="full")
    plot_inference_roofline_different_attn_type(model_files_dict=model_files_dict, output_dir=roofline_output_dir, color_type="model", plot_style="full")
    # plot_inference_roofline_different_model(model_files_dict=model_files_dict, output_dir=roofline_output_dir, plot_style="full")
    # plot_inference_roofline_different_precision(model_files_dict=model_files_dict, output_dir=roofline_output_dir, plot_style="full")
    # plot_inference_roofline_different_scenario(model_files_dict=model_files_dict, output_dir=roofline_output_dir, plot_style="full")

    # # Only Points
    # point_output_dir = os.path.join(output_dir, "points")
    # plot_inference_roofline(model_files_dict=model_files_dict, output_dir=point_output_dir, color_type="model", plot_style="points")
    # plot_inference_roofline_different_model(model_files_dict=model_files_dict, output_dir=point_output_dir, plot_style="points")
    # plot_inference_roofline_different_precision(model_files_dict=model_files_dict, output_dir=point_output_dir, plot_style="points")
    # plot_inference_roofline_different_scenario(model_files_dict=model_files_dict, output_dir=point_output_dir, plot_style="points")
    #
    # # Single Scenario
    # single_scenario_output_dir = os.path.join(output_dir, "single_scenario")
    # plot_inference_roofline_single_scenario(model_files_dict=model_files_dict, output_dir=single_scenario_output_dir, scenario="SISO")
    # plot_inference_roofline_single_scenario(model_files_dict=model_files_dict, output_dir=single_scenario_output_dir, scenario="SILO")
    # plot_inference_roofline_single_scenario(model_files_dict=model_files_dict, output_dir=single_scenario_output_dir, scenario="LISO")
    # plot_inference_roofline_single_scenario(model_files_dict=model_files_dict, output_dir=single_scenario_output_dir, scenario="LILO")
    #
    # # Single Scenario log x
    # single_scenario_output_dir = os.path.join(output_dir, "single_scenario_logx")
    # plot_inference_roofline_single_scenario_log(model_files_dict=model_files_dict, output_dir=single_scenario_output_dir, scenario="SISO")
    # plot_inference_roofline_single_scenario_log(model_files_dict=model_files_dict, output_dir=single_scenario_output_dir, scenario="SILO")
    # plot_inference_roofline_single_scenario_log(model_files_dict=model_files_dict, output_dir=single_scenario_output_dir, scenario="LISO")
    # plot_inference_roofline_single_scenario_log(model_files_dict=model_files_dict, output_dir=single_scenario_output_dir, scenario="LILO")
    #
    # # Performance vs. Layers & Operational Intensity vs. Layers
    # perf_vs_layer_output_dir = os.path.join(output_dir, "versus", "perf_vs_layers")
    # oi_vs_layer_output_dir = os.path.join(output_dir, "versus", "oi_vs_layers")
    # plot_performance_vs_layers(model_files_dict=model_files_dict, output_dir=perf_vs_layer_output_dir)
    # plot_oi_vs_layers(model_files_dict=model_files_dict, output_dir=oi_vs_layer_output_dir)
    #
    # # Total FLOPs vs. Runtime & Layers vs. Runtime & Total FLOPs vs. Layers
    # flops_vs_runtime_output_dir = os.path.join(output_dir, "versus", "flops_vs_runtime")
    # layers_vs_runtime_output_dir = os.path.join(output_dir, "versus", "layers_vs_runtime")
    # flops_vs_layers_output_dir = os.path.join(output_dir, "versus", "flops_vs_layers")
    # plot_flops_vs_runtime(model_files_dict=model_files_dict, output_dir=flops_vs_runtime_output_dir)
    # plot_layers_vs_runtime(model_files_dict=model_files_dict, output_dir=layers_vs_runtime_output_dir)
    # plot_flops_vs_layers(model_files_dict=model_files_dict, output_dir=flops_vs_layers_output_dir)
    #
    # # subplot
    # roofline_output_dir = os.path.join(output_dir, "subplot")
    # plot_inference_roofline_subplot(model_files_dict=model_files_dict, output_dir=roofline_output_dir, color_type="prec", plot_style="full",
    #                                 suptitle_title="Different Precision Comparison", subplot_layout=(2, 3))
    # plot_inference_roofline_different_scenario_subplot(model_files_dict=model_files_dict, output_dir=roofline_output_dir, plot_style="full",
    #                                 suptitle_title="Different Scenario Comparison", subplot_layout=(2, 3))
    # plot_inference_roofline_single_scenario_subplot(model_files_dict=model_files_dict, output_dir=roofline_output_dir, scenario="all",
    #                                                 suptitle_title="Layers Analysis", subplot_layout=(3, 4))