import platform
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt


def measure_memory_bandwidth():
    """
    通过一个大规模的内存复制操作来测量内存带宽（GB/s）。
    """
    size = 2 ** 28  # 256 MB
    a = np.random.rand(size).astype(np.float32)
    b = np.empty_like(a)

    # 预热循环，确保数据在缓存中
    for _ in range(5):
        b[:] = a[:]

    start_time = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        b[:] = a[:]

    end_time = time.perf_counter()

    total_bytes = size * 4 * iterations  # 4字节/float32
    elapsed_time = end_time - start_time
    bandwidth_gb_s = (total_bytes / elapsed_time) / (1024 ** 3)

    print(f"Measured memory bandwidth: {bandwidth_gb_s:.2f} GB/s")
    return bandwidth_gb_s


def measure_peak_flops():
    """
    通过一个大规模的向量加法操作来测量计算性能（GFLOPS）。
    """
    size = 2 ** 26  # 64M 个浮点数
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    c = np.empty_like(a)

    # 预热循环
    for _ in range(5):
        c = a + b

    start_time = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        c = a + b

    end_time = time.perf_counter()

    total_flops = size * iterations  # 每次操作执行一次浮点运算
    elapsed_time = end_time - start_time
    flops_g = (total_flops / elapsed_time) / (10 ** 9)

    print(f"Measured peak FLOPs: {flops_g:.2f} GFLOPS")
    return flops_g


def get_chip_model():
    """
    在macOS上使用'sysctl'获取Apple Silicon芯片型号。
    """
    # 如果不是在macOS上运行，则返回一个默认值
    if platform.system() != "Darwin":
        print("Warning: Not running on macOS. Defaulting to 'Apple M1 Pro'.")
        return "Apple M1 Pro"
    try:
        output = (
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            .strip()
            .decode()
        )
        return output
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # 如果sysctl命令失败，也返回一个默认值
        print(f"Warning: Could not determine chip model ({e}). Defaulting to 'Apple M1 Pro'.")
        return "Apple M1 Pro"


chip = get_chip_model()

# 硬件相关的参数字典
bandwidth_dict = {
    "Apple M1 Pro": 2**30 * 200,  # 200 GB/s
    "Apple M1 Max": 2**30 * 400,  # 400 GB/s
    "Apple M2 Ultra": 2**30 * 800, # 800 GB/s
}
peak_flops_dict = {
    "Apple M1 Pro": 2**40 * 5.2,   # 5.2 TFLOPS (FP32)
    "Apple M1 Max": 2**40 * 10.4,  # 10.4 TFLOPS (FP32)
    "Apple M2 Ultra": 2**40 * 27.2, # 27.2 TFLOPS (FP32)
}

# 为检测到的芯片选择参数，如果芯片不在字典中则使用默认值
theo_bandwidth = bandwidth_dict.get(chip, bandwidth_dict["Apple M1 Pro"])
theo_peak_flops = peak_flops_dict.get(chip, peak_flops_dict["Apple M1 Pro"])

# 如果检测到的芯片不在我们的数据库中，则打印警告
if chip not in bandwidth_dict:
    print(f"Warning: Chip '{chip}' not in database. Using specs for 'Apple M1 Pro'.")
    chip = "Apple M1 Pro" # 更新芯片名称以确保图表标题一致

# 为x轴（算术强度）生成一个数值范围
aint = [(2**i) / 3.0 for i in range(21)] # 范围从0到20，以获得更宽的图表

# Memory Access Bound & Computation Bound 理论值
memory_bound_line = [theo_bandwidth * ai for ai in aint]
computation_bound_line = [theo_peak_flops for _ in aint]

print(f"Theoretical memory bandwidth: {theo_bandwidth / (2**30):.2f} GB/s")
print(f"Theoretical peak FLOPs: {theo_peak_flops / (2**40):.2f} TFLOPS")

# 获取实测值
real_bandwidth = measure_memory_bandwidth()
real_flops = measure_peak_flops()

# 用实测值绘制
real_memory_bound_line = [real_bandwidth * (2 ** 30) * ai for ai in aint]  # 转换为Bytes/sec
real_computation_bound_line = [real_flops * (2 ** 40) for _ in aint]  # 转换为FLOPs/sec

# --- 开始绘图 ---
plt.figure(figsize=(8, 6))
plt.loglog(aint, real_memory_bound_line, "--", linewidth=2, label="Measured Memory Access Bound")
plt.loglog(aint, real_computation_bound_line, "--", linewidth=2, label="Measured Computation Bound")
plt.loglog(aint, memory_bound_line, "-", linewidth=2, label="Theoretical Memory Access Bound")
plt.loglog(aint, computation_bound_line, "-", linewidth=2, label="Theoretical Computation Bound")

# 设置图表标签和标题
plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
plt.ylabel("Performance (FLOPs/sec)")
plt.title(f"Memory Access Bound & Computation Bound Test on {chip}")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

# 调整坐标轴范围以更好地展示屋顶线
plt.xlim(min(aint), max(aint))
plt.ylim(top=theo_peak_flops * 2) # 将y轴上限设置为峰值性能的两倍

# 保存并显示图表
# plt.savefig("Memory Access Bound & Computation Bound Test")
# plt.show()