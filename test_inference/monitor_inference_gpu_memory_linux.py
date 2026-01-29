import time
import os
import psutil
import pynvml
import subprocess

from config import Config
from utils.common_utils import CommonUtils
from utils.config_utils import ConfigUtils
from utils.date_time_utils import DateTimeUtils

def get_process_id_by_name(process_name):
    """
    通过进程名查找对应的PID。
    如果找到，返回PID；如果未找到，返回None。
    """
    for proc in psutil.process_iter(['pid', 'name']):
        # 在try-except块中获取进程名，以处理权限问题
        try:
            process_name_info = proc.info.get('name')
            if process_name_info is not None and process_name in process_name_info:
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return None

def get_gpu_memory_usage_kb(pid):
    """
    通过 pynvml 查询指定 PID 的显存使用情况（KB）。
    """
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # 获取所有运行在当前 GPU 上的进程
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for proc in procs:
                # 检查进程ID是否匹配
                if proc.pid == pid:
                    # proc.usedGpuMemory 返回的是字节数 (Bytes)
                    # 将字节数转换为 KB
                    return CommonUtils.convert_byte_to_kb(proc.usedGpuMemory)
        return None
    except pynvml.NVMLError as error:
        print(f"pynvml error: {error}")
        return None
    finally:
        # 确保在使用后释放 pynvml 资源
        pynvml.nvmlShutdown()

def monitor_inference_memory(config: Config, p_tokens: int, n_tokens: int, dev: str) -> dict | None:
    log_dir = config.log_dir
    model_path = config.model_path
    timestamp = DateTimeUtils.get_current_timestamp()
    gpu_memory_log_path = os.path.join(log_dir, f"gpu_memory_usage_{timestamp}.log")
    interval = 0.01  # Interval in seconds to check memory usage

    # 将主进程的环境变量传递给子进程
    # project/test_inference/current_file
    new_env = os.environ.copy()
    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(current_file_path))
    if 'PYTHONPATH' in new_env:
        new_env['PYTHONPATH'] = root_dir + os.pathsep + new_env['PYTHONPATH']
    else:
        new_env['PYTHONPATH'] = root_dir

    command = [
        "python", "inference_muti_gpu.py",
        "--log_dir", log_dir,
        "--model_path", model_path,
        "--threads", str(config.threads),
        "--p_tokens", str(p_tokens),
        "--n_tokens", str(n_tokens),
        "--dev", dev,
        "--timestamp", timestamp
    ]

    # Start the inference script in the background
    try:
        subprocess.Popen(command, text=True, env=new_env)
        print("Inference script started in the background.")
    except FileNotFoundError:
        print(f"Error: Command '{command}' not found.")
        return None

    # 设置获取PID的超时时间（例如：30秒）
    timeout = 30
    start_time = time.time()
    inference_pid = None

    print("Waiting for llama-bench process to start...")
    # 循环尝试获取PID，直到超时或成功
    while inference_pid is None and time.time() - start_time < timeout:
        inference_pid = get_process_id_by_name("llama-bench")
        if inference_pid is None:
            time.sleep(0.1)  # 每隔0.1秒重试一次

    # 如果超时后仍然没有找到PID
    if inference_pid is None:
        print(f"Error: Timed out after {timeout} seconds. Could not find 'llama-bench' process.")
        return None

    print(f"Monitoring memory usage for llama-bench PID {inference_pid}...")

    # 开始监控内存使用情况
    try:
        p = psutil.Process(inference_pid)
        while p.is_running():
            try:
                # 记录 GPU 显存
                gpu_memory_kb = get_gpu_memory_usage_kb(inference_pid)
                if gpu_memory_kb is not None:
                    with open(gpu_memory_log_path, "a") as f:
                        f.write(f"{gpu_memory_kb}\n")
            except psutil.NoSuchProcess:
                print(f"Process {inference_pid} no longer exists.")
                break
            except psutil.AccessDenied:
                print(f"Access denied to process {inference_pid}.")
                break
            time.sleep(interval)
        print("Monitoring complete. The inference process has finished.")
        print(f"Log saved to {gpu_memory_log_path}.")
    except psutil.NoSuchProcess:
        print(f"Error: Process with PID {inference_pid} not found. Monitoring aborted.")
        return None

    inference_log_path = os.path.expanduser(os.path.join(log_dir, "inference_" + timestamp + ".json"))
    runtime_log_path = os.path.expanduser(os.path.join(log_dir, "runtime_" + timestamp + ".txt"))
    return {
        "memory_usage_log_path": gpu_memory_log_path,
        "inference_log_path": inference_log_path,
        "runtime_log_path": runtime_log_path,
    }


if __name__ == "__main__":
    log_path_dict = monitor_inference_memory(ConfigUtils.load_config("config.json"), p_tokens=512, n_tokens=128, dev="cuda0")