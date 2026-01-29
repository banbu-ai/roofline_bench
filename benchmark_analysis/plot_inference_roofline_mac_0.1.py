import os
from typing import Dict, List
from matplotlib import pyplot as plt
import json
from utils.common_utils import CommonUtils
from utils.bandwidth_test_utils import BandwidthTestUtils
from utils.device_info_utils import DeviceInfoUtils


model_name_list = ["Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "PLM-1.8B-Instruct", "Qwen3-0.6B"]
analysis_dir = os.path.expanduser("~/Code/llm_inference_roofline_detect/analysis/analysis_M1Pro")
merged_analysis_dir = os.path.join(analysis_dir, "merged")
output_dir = os.path.join(analysis_dir, "pic")


def create_plot_settings(model_files_dict: Dict[str, List[str]]):
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
                "label": f"{model_name} {precision} Inference",
                "file_path": file_path  # 添加文件路径，方便后续绘图
            }

    return settings


def create_plot_settings_by_model(model_files_dict: Dict[str, List[str]]):
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
    colors = ['#E65239',  # 珊瑚红
              '#1E6F8C',  # 深蓝
              '#5ABF49',  # 鲜绿
              '#F0B138']  # 金黄色

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
                "label": f"{model_name} {precision} Inference",
                "file_path": file_path
            }
    return settings


def create_plot_settings_by_precision(model_files_dict: Dict[str, List[str]]):
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
                "label": f"{model_name} {precision} Inference",
                "file_path": file_path
            }
    return settings


def _plot_roofline_base(ax, chip: str):
    """
    绘制屋顶线图的公共部分（理论和实测的屋顶线）。

    Args:
        ax (matplotlib.axes.Axes): 绘图的 Axes 对象。
        chip (str): 芯片名称，用于图表标题。
    """
    # 绘制屋顶线所需的基础数据
    theo_bandwidth, theo_peak_flops = BandwidthTestUtils.get_theo_bandwidth_peak_flops()
    x_range = [(2 ** i) / 3.0 for i in range(-6, 11)]

    # M1 Pro
    real_bandwidth = 10 ** 9 * 120
    real_flops = 10 ** 12 * 4.3

    # RTX 3070Ti
    # real_bandwidth = 10 ** 9 * 220
    # real_flops = 10 ** 12 * 9.5

    # Jetson Orin Nano Super 8G
    # real_bandwidth = 10 ** 9 * 59.4
    # real_flops = 10 ** 12 * 1.34

    real_ridge_x, _ = BandwidthTestUtils.calculate_ridge_point(real_flops, real_bandwidth)

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
    ax.set_ylim(bottom=1e7, top=theo_peak_flops * 2)


def plot_inference_roofline(model_files_dict: Dict[str, List[str]], output_dir: str, color_type: str):
    """
    为所有模型和精度绘制屋顶线图。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir:
            输出目录，用于保存生成的图像。
    """
    os.makedirs(output_dir, exist_ok=True)
    chip = DeviceInfoUtils.get_chip_model()

    if color_type == 'model':
        plot_settings = create_plot_settings_by_model(model_files_dict)
    elif color_type == 'prec':
        plot_settings = create_plot_settings_by_precision(model_files_dict)
    else:
        plot_settings = create_plot_settings(model_files_dict)

    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_roofline_base(ax, chip)

    for combination_key, settings in plot_settings.items():
        file_path = settings['file_path']
        try:
            with open(file_path, 'r') as f:
                data_points = json.load(f)
            oi_values = [d['operational_intensity'] for d in data_points]
            perf_values = [d['performance'] for d in data_points]
            ax.loglog(oi_values, perf_values, 'o', markersize=4, color=settings['color'], label=settings['label'])
        except FileNotFoundError:
            print(f"Warning: analysis file {file_path} not found. Skipping data point plotting.")

    # 设置图表标题和图例
    title = f"Inference Roofline Test on {chip}"
    ax.set_title(title)
    ax.legend(loc='lower right')

    # 保存并关闭图表
    plt.savefig(os.path.join(output_dir, title + ".png"))
    plt.close(fig)


def plot_inference_roofline_different_model(model_files_dict: Dict[str, List[str]], output_dir: str):
    """
    为每种模型绘制一个独立的屋顶线图，并在图上展示所有精度的相应数据。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir:
            输出目录，用于保存生成的图像。
    """
    os.makedirs(output_dir, exist_ok=True)
    chip = DeviceInfoUtils.get_chip_model()

    for model_name, file_list in model_files_dict.items():
        single_model_dict = {model_name: file_list}
        plot_settings = create_plot_settings_by_precision(single_model_dict)

        fig, ax = plt.subplots(figsize=(10, 8))
        _plot_roofline_base(ax, chip)

        for combination_key, settings in plot_settings.items():
            file_path = settings['file_path']
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
                oi_values = [d['operational_intensity'] for d in data_points]
                perf_values = [d['performance'] for d in data_points]
                ax.loglog(oi_values, perf_values, 'o', markersize=4, color=settings['color'], label=settings['label'])
            except FileNotFoundError:
                print(f"Warning: analysis file {file_path} not found. Skipping data point plotting.")

        # 设置图表标题和图例
        title = f"{model_name} Inference Roofline Test on {chip}"
        ax.set_title(title)
        ax.legend(loc='lower right')

        # 保存并关闭图表
        plt.savefig(os.path.join(output_dir, title + ".png"))
        plt.close(fig)


def plot_inference_roofline_different_precision(model_files_dict: Dict[str, List[str]], output_dir: str):
    """
    为每种精度绘制一个独立的屋顶线图，并在图上展示所有模型的相应数据。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir:
            输出目录，用于保存生成的图像。
    """
    os.makedirs(output_dir, exist_ok=True)
    chip = DeviceInfoUtils.get_chip_model()
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
        _plot_roofline_base(ax, chip)

        colors = ['#E65239', '#1E6F8C', '#5ABF49', '#F0B138', '#8A2BE2', '#00CED1']
        color_index = 0

        for model_name, file_path in model_data.items():
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)

                oi_values = [d['operational_intensity'] for d in data_points]
                perf_values = [d['performance'] for d in data_points]

                color = colors[color_index % len(colors)]
                label = f"{model_name} {precision} Inference"
                ax.loglog(oi_values, perf_values, 'o', markersize=4, color=color, label=label)
                color_index += 1
            except FileNotFoundError:
                print(f"Warning: analysis file {file_path} not found. Skipping data point plotting.")

        # 设置图表标题和图例
        title = f"Inference Roofline Test - {precision} on {chip}"
        ax.set_title(title)
        ax.legend(loc='lower right')

        # 保存并关闭图表
        plt.savefig(os.path.join(output_dir, title + ".png"))
        plt.close(fig)


def plot_inference_roofline_different_scenario(model_files_dict: Dict[str, List[str]], output_dir: str):
    """
    遍历所有模型文件，为每个文件绘制基于不同推理场景（SISO, SILO, LISO, LILO）的屋顶线图。

    Args:
        model_files_dict (Dict[str, List[str]]):
            一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        output_dir (str):
            保存图表的目录。
    """
    os.makedirs(output_dir, exist_ok=True)
    chip = DeviceInfoUtils.get_chip_model()

    # 定义场景的阈值，可以根据你的数据特征进行调整
    short_in_threshold = 2048
    short_out_threshold = 2048

    # 定义场景的颜色和标签
    scenario_settings = {
        'SISO': {'color': '#1E6F8C', 'label': 'SISO (Short In, Short Out)'},
        'SILO': {'color': '#E65239', 'label': 'SILO (Short In, Long Out)'},
        'LISO': {'color': '#F0B138', 'label': 'LISO (Long In, Short Out)'},
        'LILO': {'color': '#5ABF49', 'label': 'LILO (Long In, Long Out)'}
    }

    for model_name, file_list in model_files_dict.items():
        for file_path in file_list:
            try:
                with open(file_path, 'r') as f:
                    data_points = json.load(f)
            except FileNotFoundError:
                print(f"Error: Analysis file not found at {file_path}. Skipping.")
                continue

            # 根据文件名提取精度
            base_name = os.path.basename(file_path)
            if '_f16_' in base_name:
                precision = 'FP16'
            elif '_q8_0_' in base_name:
                precision = 'Q8_0'
            elif '_q4_k_m_' in base_name:
                precision = 'Q4_K_M'
            else:
                precision = 'Unknown Precision'

            # 初始化场景数据分组
            scenarios = {key: {'points': []} for key in scenario_settings.keys()}

            # 遍历数据点并进行归类
            for d in data_points:
                n_prompt = d.get('inference_info', {}).get('n_prompt', 0)
                n_gen = d.get('inference_info', {}).get('n_gen', 0)

                if n_prompt <= short_in_threshold and n_gen <= short_out_threshold:
                    scenarios['SISO']['points'].append(d)
                elif n_prompt <= short_in_threshold and n_gen > short_out_threshold:
                    scenarios['SILO']['points'].append(d)
                elif n_prompt > short_in_threshold and n_gen <= short_out_threshold:
                    scenarios['LISO']['points'].append(d)
                elif n_prompt > short_in_threshold and n_gen > short_out_threshold:
                    scenarios['LILO']['points'].append(d)

            # --- 开始绘图 ---
            fig, ax = plt.subplots(figsize=(10, 8))
            _plot_roofline_base(ax, chip)

            # 遍历场景并绘制数据点
            for scenario_name, scenario_data in scenarios.items():
                if scenario_data['points']:
                    oi_values = [p['operational_intensity'] for p in scenario_data['points']]
                    perf_values = [p['performance'] for p in scenario_data['points']]

                    color = scenario_settings[scenario_name]['color']
                    label = scenario_settings[scenario_name]['label']

                    ax.loglog(oi_values, perf_values, 'o', markersize=4, color=color, label=label)

            # 设置图表标题和图例
            title = f"{model_name} {precision} Different Inference Scenarios on {chip}"
            ax.set_title(title)
            ax.legend(loc='lower right')

            # 保存并关闭图表
            plt.savefig(os.path.join(output_dir, title + ".png"))
            plt.close(fig)

if __name__ == "__main__":
    model_files_dict = CommonUtils.find_files_by_model(model_name_list, merged_analysis_dir)
    print(model_files_dict)
    # CommonUtils.sort_files_by_precision(model_files_dict)
    # plot_inference_roofline(model_files_dict=model_files_dict, output_dir=output_dir, color_type="model")
    # plot_inference_roofline_different_model(model_files_dict=model_files_dict, output_dir=output_dir)
    # plot_inference_roofline_different_precision(model_files_dict=model_files_dict, output_dir=output_dir)
    # plot_inference_roofline_different_scenario(model_files_dict=model_files_dict, output_dir=output_dir)