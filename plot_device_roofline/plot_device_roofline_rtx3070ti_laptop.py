import torch
from matplotlib import pyplot as plt

from utils.bandwidth_test_utils import BandwidthTestUtils
from utils.device_info_utils import DeviceInfoUtils

platform = "Windows"
chip = "NVIDIA GeForce RTX 3070 Ti Laptop GPU" # 仅用作绘图标题显示
print(f"Platform: {platform}")
print(f"Chip: {chip}")

# Platform: Windows
# Chip: NVIDIA GeForce RTX 3070 Ti Laptop GPU
# Theoretical memory bandwidth: 448.00 GB/s
# Theoretical peak FLOPs (FP32): 16.60 TFLOPS
# Measured memory bandwidth: 217.00 GB/s
# Measured peak FLOPs (FP16): 31.76 TFLOPS
# Measured peak FLOPs (FP32): 9.51 TFLOPS

# 获取理论值
theo_bandwidth, theo_peak_flops = BandwidthTestUtils.get_theo_bandwidth_peak_flops_by_chip_name(chip)
print(f"Theoretical memory bandwidth: {theo_bandwidth / (10 ** 9):.2f} GB/s")
print(f"Theoretical peak FLOPs (FP32): {theo_peak_flops / (10 ** 12):.2f} TFLOPS")

# M1 Pro
# real_bandwidth = 120 * 10 ** 9
# real_flops = 4.3 * 10 ** 12

# RTX 3090
# real_bandwidth = 560.00 * 10 ** 9
# real_flops = 24.28 * 10 ** 12

# RTX 3070Ti Laptop
real_bandwidth = 220 * 10 ** 9
real_flops = 9.5 * 10 ** 12

# Jetson Orin Nano Super 8G
# real_bandwidth = 59.4 * 10 ** 9
# real_flops = 1.34 * 10 ** 12

# Raspberry Pi 5
# real_bandwidth = 3.90 * 10 ** 9
# real_flops = 78.50 * 10 ** 9

print(f"Measured memory bandwidth: {real_bandwidth / (10 ** 9):.2f} GB/s") # 转换为 GB/s
print(f"Measured peak FLOPs: {real_flops / (10 ** 12):.2f} TFLOPS") # 转换为 TFLOPS

# 计算理论和实测的 Ridge Point
theo_ridge_x, _ = BandwidthTestUtils.calculate_ridge_point(theo_peak_flops, theo_bandwidth)
real_ridge_x, _ = BandwidthTestUtils.calculate_ridge_point(real_flops, real_bandwidth)

# 为x轴（算术强度）生成一个数值范围
x_range = [(2 ** i) / 3.0 for i in range(21)]  # 范围从0到20，以获得更宽的图表

# Memory Access Bound & Computation Bound 理论值
memory_bound_line = [theo_bandwidth * ai for ai in x_range]
computation_bound_line = [theo_peak_flops for _ in x_range]

# 用实测值绘制
real_memory_bound_line = [real_bandwidth  * ai for ai in x_range]
real_computation_bound_line = [real_flops for _ in x_range]

# --- 开始绘图 ---
plt.figure(figsize=(8, 6))
plt.loglog(x_range, memory_bound_line, "-", linewidth=2, label="Theoretical Memory Access Bound")
plt.loglog(x_range, computation_bound_line, "-", linewidth=2, label="Theoretical Computation Bound")
plt.loglog(x_range, real_memory_bound_line, "--", linewidth=2, label="Measured Memory Access Bound")
plt.loglog(x_range, real_computation_bound_line, "--", linewidth=2, label="Measured Computation Bound")

# 在理论 Ridge Point 处绘制垂直线，并在图例中标注数值
# plt.axvline(x=theo_ridge_x, linestyle='--', linewidth=1, label=f"Theoretical Ridge: {theo_ridge_x:.2f} FLOPs/Byte")
# 在实测 Ridge Point 处绘制垂直线，并在图例中标注数值
plt.axvline(x=real_ridge_x, linestyle='--', linewidth=1, label=f"Measured Ridge: {real_ridge_x:.2f} FLOPs/Byte")

# 设置图表标签和标题
plt.xlabel("Operational Intensity (FLOPs/Byte)")
plt.ylabel("Performance (FLOPs/sec)")
plt.title(f"Roofline Analysis of {chip}")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

# 调整坐标轴范围以更好地展示屋顶线
plt.xlim(min(x_range), max(x_range))
plt.ylim(top=theo_peak_flops * 2)  # 将y轴上限设置为峰值性能的两倍

# 保存并显示图表
plt.savefig(f"Roofline Analysis of {chip}")
# plt.show()