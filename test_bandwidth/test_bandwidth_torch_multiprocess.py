import platform
import time
import subprocess
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import torch
import multiprocessing


def add_tensor(size, num_iterations, device):
    """
    单个进程执行的张量操作任务
    """
    a = torch.ones(size, device=device)
    b = torch.ones(size, device=device)
    for _ in range(num_iterations):
        c = a + b


def measure_memory_bandwidth_multiprocess(device):
    """
    使用多进程在CPU/MPS上测量内存带宽
    """
    if device == "cpu":
        num_processes = multiprocessing.cpu_count()  # 获取可用的CPU核心数
        print(f"Using {num_processes} processes to measure CPU memory bandwidth.")
    elif device == "mps":
        num_processes = 16
        print(f"Using {num_processes} processes to measure MPS memory bandwidth.")

    size = 10**7  # 每个进程的张量大小
    num_iterations = 1000

    # 记录开始时间
    start_time = time.time()

    processes = []
    for _ in range(num_processes):
        p = multiprocessing.Process(target=add_tensor, args=(size, num_iterations, device))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()  # 等待所有进程完成

    end_time = time.time()
    elapsed_time = end_time - start_time

    # 计算总处理的数据量
    # (张量大小 * 元素大小 * 2次读写 * 循环次数 * 进程数)
    total_data_processed = size * torch.ones(1).element_size() * 2 * num_iterations * num_processes
    bandwidth = total_data_processed / elapsed_time

    if device == "cpu":
        print(f"Measured CPU memory bandwidth (multiprocess): {bandwidth / (2**30):.2f} GB/s")
    elif device == "mps":
        print(f"Measured MPS memory bandwidth (multiprocess): {bandwidth / (2**30):.2f} GB/s")
    return bandwidth / (2**30)


def measure_memory_bandwidth_multithreaded():
    """
    使用多线程测量内存带宽，确保充分利用CPU的所有核心。
    """
    device = torch.device("mps")
    size = 10 ** 8  # 张量大小
    num_threads = torch.get_num_threads()  # 获取CPU核心数
    a = torch.ones(size, device=device)
    b = torch.ones(size, device=device)

    def tensor_addition():
        for _ in range(100):  # 增加循环次数
            c = a + b

    # 记录开始时间
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(tensor_addition) for _ in range(num_threads)]
        for future in futures:
            future.result()  # 等待所有线程完成
    torch.mps.synchronize()  # 确保所有操作完成
    end_time = time.time()

    elapsed_time = end_time - start_time
    bandwidth = (a.element_size() * a.nelement() * 2 * 100 * num_threads) / elapsed_time  # 单位：字节/秒
    print(f"Measured memory bandwidth (multithreaded): {bandwidth / (2**30):.2f} GB/s")
    return bandwidth / (2**30)  # 转换为GB/s


def measure_memory_bandwidth():
    """
    测量内存带宽，使用torch在MPS设备上进行大规模张量操作。
    """
    device = torch.device("mps")
    size = 10**8  # 增大张量大小
    a = torch.ones(size, device=device)
    b = torch.ones(size, device=device)

    # 记录开始时间
    start_time = time.time()
    for _ in range(100):  # 增加循环次数
        c = a + b
    torch.mps.synchronize()  # 确保所有操作完成
    end_time = time.time()

    elapsed_time = end_time - start_time
    bandwidth = (a.element_size() * a.nelement() * 2 * 100) / elapsed_time  # 单位：字节/秒
    print(f"Measured memory bandwidth: {bandwidth / (2**30):.2f} GB/s")
    return bandwidth / (2**30)  # 转换为GB/s


def measure_peak_flops(dtype=torch.float32):
    """
    测量FP32的峰值FLOPS，使用torch在MPS设备上进行矩阵乘法。
    """
    device = torch.device("mps")
    size = 4096  # 定义矩阵大小
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)

    # 记录开始时间
    start_time = time.time()
    for _ in range(10):  # 重复多次以获得更稳定的测量
        c = torch.matmul(a, b)
    torch.mps.synchronize()  # 确保所有操作完成
    end_time = time.time()

    elapsed_time = end_time - start_time
    flops = (2 * size ** 3 * 10) / elapsed_time  # 单位：FLOPS
    return flops / (10 ** 12)  # 转换为TFLOPS


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

if __name__ == '__main__':

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
    real_cpu_bandwidth = measure_memory_bandwidth_multiprocess("cpu")
    real_mps_bandwidth = measure_memory_bandwidth_multiprocess("mps")
    real_flops = measure_peak_flops()

    # 用实测值绘制
    real_memory_bound_line = [real_mps_bandwidth * (2 ** 30) * ai for ai in aint]  # 转换为Bytes/sec
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