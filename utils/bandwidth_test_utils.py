import time
import torch
import json
import pathlib

from concurrent.futures import ThreadPoolExecutor
from utils.device_info_utils import DeviceInfoUtils


class BandwidthTestUtils:

    # Apple M1 Pro https://en.wikipedia.org/wiki/Apple_M1
    # NVIDIA GeForce RTX 3070 Ti Laptop GPU https://www.techpowerup.com/gpu-specs/geforce-rtx-3070-ti-mobile.c3852
    # Jetson Orin Nano 8G https://www.techpowerup.com/gpu-specs/jetson-orin-nano-8-gb.c4082
    # Jetson Orin Nano Super 8G https://www.techpowerup.com/gpu-specs/jetson-orin-nano-super.c4377
    # Raspberry Pi 5 https://www.raspberrypi.com/news/introducing-raspberry-pi-5/

    current_file_path = pathlib.Path(__file__).resolve()
    project_root = current_file_path.parent.parent
    db_path = project_root / "device_info_db.json"
    with open(db_path, 'r', encoding='utf-8') as f:
        device_info_db = json.load(f)

    # 硬件相关的参数字典
    bandwidth_dict = device_info_db['bandwidth']
    peak_flops_dict = device_info_db['peak_flops_fp32']

    def __init__(self, chip_model):
        self.chip_model = chip_model

    def get_theo_bandwidth(self):
        return self.bandwidth_dict.get(self.chip_model, "Chip not found")

    def get_theo_peak_flops(self):
        return self.peak_flops_dict.get(self.chip_model, "Chip not found")

    @staticmethod
    def get_theo_bandwidth_peak_flops(chip_type="gpu") -> tuple:
        if chip_type == "cpu":
            chip = DeviceInfoUtils.get_chip_model(chip_type="cpu")
        else:
            chip = DeviceInfoUtils.get_chip_model(chip_type="gpu")
            # 检查是否获取到芯片型号，并处理 None 或不存在的情况
        if chip is None or chip not in BandwidthTestUtils.bandwidth_dict:
            if chip is not None:
                print(
                    f"Warning: Chip '{chip}' not in database. Using specs for 'NVIDIA GeForce RTX 3070 Ti Laptop GPU'.")
            else:
                print("Warning: No supported device detected. Using specs for 'NVIDIA GeForce RTX 3070 Ti Laptop GPU'.")
            chip = "NVIDIA GeForce RTX 3070 Ti Laptop GPU"
        # 为检测到的芯片选择参数
        theo_bandwidth = BandwidthTestUtils.bandwidth_dict.get(chip)
        theo_peak_flops = BandwidthTestUtils.peak_flops_dict.get(chip)
        return theo_bandwidth, theo_peak_flops

    @staticmethod
    def get_theo_bandwidth_peak_flops_by_chip_name(chip_name) -> tuple | str:
        theo_bandwidth = BandwidthTestUtils.bandwidth_dict.get(chip_name)
        theo_peak_flops = BandwidthTestUtils.peak_flops_dict.get(chip_name)
        if theo_bandwidth is not None and theo_peak_flops is not None:
            return theo_bandwidth, theo_peak_flops
        else:
            return "Chip not in database"
    
    @staticmethod
    def measure_memory_bandwidth(platform)-> float:
        """
        测量内存/显存带宽
        使用 torch 在 MPS 或 CUDA 上进行大规模张量操作
        单位：Byte/s
        """
        if platform == "Darwin" and torch.mps.is_available():
            device = torch.device("mps")
        elif platform == "Windows" or "Linux" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise RuntimeError("No supported GPU found for memory bandwidth measurement.")

        size = 10**8  # 增大张量大小
        a = torch.empty(size, device=device)
        b = torch.empty(size, device=device)
    
        # 记录开始时间
        start_time = time.time()
        for _ in range(100):  # 增加循环次数
            c = a + b
        torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()  # 确保所有操作完成
        end_time = time.time()
    
        elapsed_time = end_time - start_time
        bandwidth = (a.element_size() * a.nelement() * 2 * 100) / elapsed_time  # 单位：字节/秒
        torch.cuda.empty_cache() if device.type == "cuda" else torch.mps.empty_cache() # 清空缓存
        return bandwidth
    
    @staticmethod
    def measure_peak_flops(platform, dtype=torch.float32) -> float:
        """
        测量 FP16/FP32 的峰值 FLOPS
        使用 torch 在 MPS 或 CUDA 上进行矩阵乘法
        单位：FLOPS
        """
        if platform == "Darwin" and torch.mps.is_available():
            device = torch.device("mps")
        elif platform == "Windows" or "Linux" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise RuntimeError("No supported GPU found for memory bandwidth measurement.")

        device = torch.device(device)
        size = 4096  # 定义矩阵大小
        a = torch.empty(size, size, device=device, dtype=dtype)
        b = torch.empty(size, size, device=device, dtype=dtype)
    
        # 记录开始时间
        start_time = time.time()
        for _ in range(10):  # 重复多次以获得更稳定的测量
            c = torch.matmul(a, b)
        torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()  # 确保所有操作完成
        end_time = time.time()
    
        elapsed_time = end_time - start_time
        flops = (2 * size ** 3 * 10) / elapsed_time  # 单位：FLOPS
        torch.cuda.empty_cache() if device.type == "cuda" else torch.mps.empty_cache() # 清空缓存
        return flops

    @staticmethod
    def measure_memory_bandwidth_cpu() -> float:
        """
        测量 CPU 内存带宽
        使用 torch 在 CPU 上进行大规模张量操作
        单位：Byte/s
        """
        device = torch.device("cpu")
        size = 10 ** 8  # 增大张量大小
        a = torch.empty(size, device=device)
        b = torch.empty(size, device=device)

        # 记录开始时间
        start_time = time.time()
        for _ in range(100):  # 增加循环次数
            c = a + b
        torch.cpu.synchronize() # 确保所有操作完成
        end_time = time.time()

        elapsed_time = end_time - start_time
        bandwidth = (a.element_size() * a.nelement() * 2 * 100) / elapsed_time  # 单位：字节/秒
        return bandwidth

    @staticmethod
    def measure_peak_flops_cpu(dtype=torch.float32) -> float:
        """
        测量 CPU FP16/FP32 的峰值 FLOPS
        使用 torch 在 CPU 上进行矩阵乘法
        单位：FLOPS
        """
        device = torch.device("cpu")
        size = 4096  # 定义矩阵大小
        a = torch.empty(size, size, device=device, dtype=dtype)
        b = torch.empty(size, size, device=device, dtype=dtype)

        # 记录开始时间
        start_time = time.time()
        for _ in range(10):  # 重复多次以获得更稳定的测量
            c = torch.matmul(a, b)
        torch.cpu.synchronize()  # 确保所有操作完成
        end_time = time.time()

        elapsed_time = end_time - start_time
        flops = (2 * size ** 3 * 10) / elapsed_time  # 单位：FLOPS
        return flops

    @staticmethod
    def measure_memory_bandwidth_multithreaded(platform) -> float:
        """
        使用多线程测量测量内存/显存带宽
        使用 torch 在 MPS 或 CUDA 上进行大规模张量操作
        单位：Byte/s
        """
        if platform == "Darwin" and torch.mps.is_available():
            device = torch.device("mps")
        elif platform == "Windows" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise RuntimeError("No supported GPU found for memory bandwidth measurement.")

        size = 10 ** 8  # 张量大小
        num_threads = torch.get_num_threads()  # 获取CPU核心数
        a = torch.empty(size, device=device)
        b = torch.empty(size, device=device)
        def tensor_addition():
            for _ in range(100):  # 增加循环次数
                c = a + b
        # 记录开始时间
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(tensor_addition) for _ in range(num_threads)]
            for future in futures:
                future.result()  # 等待所有线程完成
        torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()  # 确保所有操作完成
        end_time = time.time()
    
        elapsed_time = end_time - start_time
        bandwidth = (a.element_size() * a.nelement() * 2 * 100 * num_threads) / elapsed_time  # 单位：字节/秒
        torch.cuda.empty_cache() if device.type == "cuda" else torch.mps.empty_cache() # 清空缓存
        print(f"Measured memory bandwidth (multithreaded): {bandwidth / (10 ** 9):.2f} GB/s")
        return bandwidth

    @staticmethod
    def calculate_ridge_point(peak_flops, memory_bandwidth) -> tuple:
        """
        Calculates the ridge point (the intersection of the memory bound and computation bound lines).
        Args:
            peak_flops (float): The peak performance in FLOPs/sec.
            memory_bandwidth (float): The memory bandwidth in Bytes/sec.
        Returns:
            tuple: A tuple (x, y) representing the ridge point coordinates.
        """
        if memory_bandwidth == 0:
            return (0, 0)  # Avoid division by zero

        # x-coordinate (Arithmetic Intensity) = Peak FLOPs / Memory Bandwidth
        ridge_x = peak_flops / memory_bandwidth
        # y-coordinate (Performance) = Peak FLOPs
        ridge_y = peak_flops

        return ridge_x, ridge_y
