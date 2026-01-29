import platform
import subprocess
import cpuinfo
import torch

class DeviceInfoUtils:

    @staticmethod
    def get_platform():
        return platform.system()

    @staticmethod
    def get_chip_model(chip_type="gpu") -> str | None:
        """
        获取 CPU 或 GPU 型号
        Args:
            chip_type (str): "cpu" 或 "gpu"
        Returns:
            str: 芯片型号字符串 或 None（未检测到或出错时）
        """
        platform = DeviceInfoUtils.get_platform()
        # Apple Silicon 采用 UMA 架构 不区分 CPU 和 GPU
        if platform == "Darwin":
            try:
                return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).strip().decode()
                # return cpuinfo.get_cpu_info().get('brand_raw', None)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Warning: Could not determine chip model on macOS ({e}). Defaulting to 'Apple M1 Pro'.")
                return None
        # Windows & Linux
        if chip_type == "gpu" and torch.cuda.is_available():
            try:
                # 获取 CUDA 设备的名称
                return torch.cuda.get_device_name(0)
            except Exception as e:
                print(f"Warning: Could not get CUDA device name ({e}).")
                return None
        elif chip_type == "cpu":
            try:
                if platform == "Windows":
                    # wmi 在 macOS 上不可用
                    # ModuleNotFoundError: No module named 'win32com'
                    # c = wmi.WMI()
                    # for processor in c.Win32_Processor():
                    #     return processor.Name
                    return cpuinfo.get_cpu_info().get('brand_raw', None)
                elif platform == "Linux":
                    return cpuinfo.get_cpu_info().get('brand_raw', None)
            except Exception as e:
                print(f"Warning: Could not determine CPU model ({e}).")
                return None
        else:
            print("Wrong Chip type specified. Use 'cpu' or 'gpu'.")
            return None