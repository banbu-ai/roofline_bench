import math
import os
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import json
from utils.common_utils import CommonUtils
from utils.bandwidth_test_utils import BandwidthTestUtils


analysis_data_list = [
    {
        "chip": "Apple M1 Pro",
        "model_name_list": ["Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "PLM-1.8B-Instruct", "Qwen3-0.6B", "Fox-1-1.6B", "SmolLM2-1.7B-Instruct"],
        "analysis_dir": os.path.expanduser("~/Code/llm_inference_roofline_detect/analysis/analysis_M1Pro")
    },
    {
        "chip": "NVIDIA GeForce RTX 3070 Ti Laptop GPU",
        "model_name_list": ["Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "PLM-1.8B-Instruct", "Qwen3-0.6B", "Fox-1-1.6B", "SmolLM2-1.7B-Instruct"],
        "analysis_dir": os.path.expanduser("~/Code/llm_inference_roofline_detect/analysis/analysis_RTX 3070Ti")
    },
    {
        "chip": "Jetson Orin Nano Super 8G",
        "model_name_list": ["Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "PLM-1.8B-Instruct", "Qwen3-0.6B", "Fox-1-1.6B", "SmolLM2-1.7B-Instruct"],
        "analysis_dir": os.path.expanduser("~/Code/llm_inference_roofline_detect/analysis/analysis_Jetson")
    },
    {
        "chip": "Raspberry Pi 5",
        "model_name_list": ["Qwen2.5-0.5B-Instruct", "SmolLM2-135M-Instruct", "pythia-160m", "SmolLM2-360M-Instruct", "pythia-410m"],
        "analysis_dir": os.path.expanduser("~/Code/llm_inference_roofline_detect/analysis/analysis_Raspberry Pi")
    }
]


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
    x_range = [(2 ** i) / 3.0 for i in range(-10, 25)]

    if chip == "Apple M1 Pro":
        # M1 Pro
        real_bandwidth = 120 * 10 ** 9
        real_flops = 4.3 * 10 ** 12
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
    # chip = DeviceInfoUtils.get_chip_model()

    remove_model_size = False if chip == "Raspberry Pi 5" else True

    if color_type == 'model':
        plot_settings = create_plot_settings_by_model(model_files_dict, remove_model_size=remove_model_size)
    elif color_type == 'prec':
        plot_settings = create_plot_settings_by_precision(model_files_dict, remove_model_size=remove_model_size)
    else:
        plot_settings = create_plot_settings(model_files_dict, remove_model_size=remove_model_size)

    fig, ax = plt.subplots(figsize=(12, 10))
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

def plot_inference_roofline_multi_device_subplot(
        model_files_dict_list: List[Dict[str, List[str]]],
        chip_list: List[str],
        output_dir: str,
        color_type: str,
        suptitle_title: str = "Benchmark Overall",
        subplot_layout: Tuple[int, int] = (2, 2),
        plot_style: str = "full"
):
    """
    为多个设备（子图）绘制屋顶线图，每个子图展示一个设备上所有模型的性能。

    Args:
        model_files_dict_list (List[Dict[str, List[str]]]):
            字典列表，每个字典包含一个设备上的模型文件信息。
        chip_list (List[str]):
            设备名称列表，与 model_files_dict_list 一一对应，用于子图标题。
        output_dir (str):
            输出目录，用于保存生成的图像。
        color_type (str):
            着色类型，'model', 'prec' 或其他。
        suptitle_title (str):
            总图的标题 (suptitle)。
        subplot_layout (Tuple[int, int]):
            子图的行数和列数。
        plot_style (str):
            绘图风格，'full'（绘制屋顶线和数据点）或 'points'（仅绘制数据点）。
    """
    os.makedirs(output_dir, exist_ok=True)

    if len(model_files_dict_list) != len(chip_list):
        print("Error: The length of model_files_dict_list must match the length of chip_list.")
        return

    num_devices = len(chip_list)
    n_rows, n_cols = subplot_layout

    if num_devices == 0:
        print("Error: No devices specified for plotting.")
        return

    # 确定实际布局
    rows = min(n_rows, math.ceil(num_devices / n_cols))
    cols = min(n_cols, num_devices if rows == 1 else n_cols)

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))

    if num_devices == 1:
        axes = [axes]
    elif isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.suptitle(suptitle_title, fontsize=40)
    plot_roofline_curves = (plot_style == "full")

    for i in range(num_devices):
        if i >= rows * cols:
            print(
                f"Warning: Data for device '{chip_list[i]}' exceeds subplot layout capacity ({rows * cols}). Skipping.")
            break

        ax = axes[i]
        model_files_dict = model_files_dict_list[i]
        chip_name = chip_list[i]

        # 1. 准备绘图设置 (颜色和标签)
        remove_model_size = False if chip_name == "Raspberry Pi 5" else True

        if color_type == 'model':
            plot_settings = create_plot_settings_by_model(model_files_dict, remove_model_size=remove_model_size)
        elif color_type == 'prec':
            plot_settings = create_plot_settings_by_precision(model_files_dict, remove_model_size=remove_model_size)
        else:
            plot_settings = create_plot_settings(model_files_dict, remove_model_size=remove_model_size)

        # 2. 绘制屋顶线基准
        _plot_roofline_base(ax, chip_name, plot_roofline_curves)

        all_oi_values = []
        all_perf_values = []

        # 3. 绘制数据点
        for combination_key, settings in plot_settings.items():
            file_path = settings['file_path']
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
                oi_values = [d['operational_intensity'] for d in data_points]
                perf_values = [d['performance'] for d in data_points]
                all_oi_values.extend(oi_values)
                all_perf_values.extend(perf_values)
                # 每个子图有自己的图例
                ax.loglog(oi_values, perf_values, 'o', markersize=4, color=settings['color'], label=settings['label'])
            except FileNotFoundError:
                print(f"Warning: analysis file {file_path} not found for {chip_name}. Skipping data point plotting.")
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in file {file_path} for {chip_name}. Skipping data point plotting.")

        # 4. 调整Y轴范围以适应所有数据点
        if not plot_roofline_curves and all_perf_values:
            min_y = min(all_perf_values) * 0.5 if all_perf_values else 1e7
            max_y = max(all_perf_values) * 1.5 if all_perf_values else 1e12
            ax.set_ylim(bottom=min_y, top=max_y)

        # 5. 设置子图标题和图例
        ax.set_title(chip_name, fontsize=12)  # 子图标题为设备名称
        ax.legend(loc='lower right', fontsize='small')  # 子图独享图例

    # 6. 隐藏多余的子图
    for j in range(num_devices, rows * cols):
        if j < len(axes):
            fig.delaxes(axes[j])

    # 7. 保存总图
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))  # 调整布局以留出 suptitle 空间

    file_name = suptitle_title.replace(' ', '_') + (
        "_roofline" if plot_roofline_curves else "") + "_mult-device_subplot" + ".png"
    file_path = os.path.join(output_dir, file_name)

    plt.savefig(file_path)
    print(f"Save to {file_path}.")
    plt.close(fig)

if __name__ == "__main__":
    chip_list = [config["chip"] for config in analysis_data_list]
    for config in analysis_data_list:
        analysis_dir = config["analysis_dir"]
        config["merged_analysis_dir"] = os.path.join(analysis_dir, "merged")
        model_files_dict = CommonUtils.find_files_by_model(config["model_name_list"], config["merged_analysis_dir"])
        CommonUtils.sort_files_by_precision(model_files_dict)
        config["model_files_dict"] = model_files_dict

    model_files_dict_list = [config["model_files_dict"] for config in analysis_data_list]

    output_dir = os.path.expanduser("~/Code/llm_inference_roofline_detect/analysis")

    plot_inference_roofline_multi_device_subplot(model_files_dict_list=model_files_dict_list, chip_list=chip_list, output_dir=output_dir, color_type='model',
                                                 suptitle_title="Benchmark Overall", subplot_layout=(2, 2), plot_style="full")

