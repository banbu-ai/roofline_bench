import platform
import subprocess
import matplotlib.pyplot as plt

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

# 计算两个边界的y值
memory_bound_line = [theo_bandwidth * ai for ai in aint]
computation_bound_line = [theo_peak_flops for _ in aint]

# --- 开始绘图 ---
plt.figure(figsize=(8, 6))

# 绘制内存访问边界和计算边界
plt.loglog(aint, memory_bound_line, "-", linewidth=2, label="Memory Access Bound")
plt.loglog(aint, computation_bound_line, "-", linewidth=2, label="Computation Bound")

# 设置图表标签和标题
plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
plt.ylabel("Performance (FLOPs/sec)")
plt.title(f"Theoretical Memory Access Bound & Computation Bound on {chip}")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

# 调整坐标轴范围以更好地展示屋顶线
plt.xlim(min(aint), max(aint))
plt.ylim(top=theo_peak_flops * 2) # 将y轴上限设置为峰值性能的两倍

# 保存并显示图表
# plt.savefig("Theoretical Memory Access Bound & Computation Bound.png")
plt.show()