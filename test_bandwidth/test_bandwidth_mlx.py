import platform
import time
import subprocess
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import torch
import mlx.core as mx
from multiprocessing import Process, Pipe


def add_tensor(size, conn):
    """
    工作进程，在MLX上执行计算，并将结果发送回主进程。
    """
    try:
        # 创建张量
        a = mx.ones(size)
        b = mx.ones(size)

        # 执行计算
        for _ in range(100):
            c = a + b

        # 强制同步并执行计算
        mx.eval(c)

        # 发送完成信号
        conn.send("done")
    except Exception as e:
        conn.send(f"error: {e}")
    finally:
        conn.close()


def measure_memory_bandwidth_mlx(num_cores):
    """
    使用多核心测量内存带宽，将任务分配给多个子进程。
    """
    total_size = 10 ** 8
    size_per_core = total_size // num_cores

    parents, children = zip(*[Pipe() for _ in range(num_cores)])
    processes = [
        Process(target=add_tensor, args=(size_per_core, child)) for child in children
    ]

    start_time = time.time()
    for p in processes:
        p.start()

    # 等待所有子进程完成
    for parent in parents:
        result = parent.recv()
        if result != "done":
            print(f"Error in a worker process: {result}")
            return

    for p in processes:
        p.join()

    end_time = time.time()

    elapsed_time = end_time - start_time

    element_size = 4  # 假设是32位浮点数，即4字节
    nelement = total_size

    # 2代表读取a和b，100代表循环次数
    bandwidth = (element_size * nelement * 2 * 100) / elapsed_time

    print(f"Using {num_cores} cores:")
    print(f"Measured memory bandwidth: {bandwidth / (2 ** 30):.2f} GB/s")

    return bandwidth / (2 ** 30)


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
    real_bandwidth = measure_memory_bandwidth_mlx(2)
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