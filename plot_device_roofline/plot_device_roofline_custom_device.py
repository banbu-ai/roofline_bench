import torch
from matplotlib import pyplot as plt

from utils.bandwidth_test_utils import BandwidthTestUtils
from utils.device_info_utils import DeviceInfoUtils

platform = DeviceInfoUtils.get_platform()
chip = DeviceInfoUtils.get_chip_model()
print(f"Platform: {platform}")
print(f"Chip: {chip}")

# 获取理论值
theo_bandwidth, theo_peak_flops = BandwidthTestUtils.get_theo_bandwidth_peak_flops()
print(f"Theoretical memory bandwidth: {theo_bandwidth / (10 ** 9):.2f} GB/s")
print(f"Theoretical peak FLOPs (FP32): {theo_peak_flops / (10 ** 12):.2f} TFLOPS")

# 获取实测值（10次取平均）
real_bandwidth = sum(BandwidthTestUtils.measure_memory_bandwidth(platform) for _ in range(10)) / 10
real_flops = sum(BandwidthTestUtils.measure_peak_flops(platform,torch.float32) for _ in range(10)) / 10
# real_flops = sum(BandwidthTestUtils.measure_peak_flops(platform,torch.float16) for _ in range(10)) / 10
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