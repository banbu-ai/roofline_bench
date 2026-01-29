import time
import os
import psutil
import subprocess
import xml.etree.ElementTree as ET

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
        try:
            process_name_info = proc.info.get('name')
            if process_name_info is not None and process_name in process_name_info:
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return None


def get_gpu_memory_usage_for_single_process_kb(pid):
    """
    检查GPU上是否只有目标进程在运行，如果是，返回总的显存占用。

    参数:
    pid (int): 目标进程的PID。

    返回:
    str: 描述显存使用情况的字符串，或者错误信息。
    """
    try:
        # 使用subprocess运行nvidia-smi命令，获取XML输出
        result = subprocess.run(['nvidia-smi', '-q', '-x'],
                                capture_output=True, text=True, check=True)
        xml_output = result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return f"Error executing nvidia-smi: {e}"

    try:
        # 使用ElementTree解析XML
        root = ET.fromstring(xml_output)
        processes = root.findall('.//processes/process_info')

        # 检查进程总数和目标进程是否存在
        if len(processes) == 1:
            pid_element = processes[0].find('pid')
            if pid_element is not None and int(pid_element.text) == pid:
                # 如果只有一个进程且是目标进程，返回总的显存使用量
                fb_memory_usage = root.find('.//fb_memory_usage')
                if fb_memory_usage is not None:
                    used_memory = fb_memory_usage.find('used')
                    if used_memory is not None:
                        memory_mb = float(used_memory.text.strip(" MiB")) # 移除末尾的 " MiB"
                        return CommonUtils.convert_mb_to_kb(memory_mb)
        # 还有其他进程时，列出所有进程的PID
        pid_list = [int(p.find('pid').text) for p in processes if p.find('pid') is not None]
        # print(f"不满足条件。当前进程数: {len(processes)}, 目标PID {pid}。现有PID: {pid_list}")
        return None
    except (ET.ParseError, ValueError, AttributeError) as e:
        return f"Error parsing XML output or processing data: {e}"


def monitor_inference_memory(config: Config, p_tokens: int, n_tokens: int) -> dict | None:
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
        "python", "inference.py",
        "--log_dir", log_dir,
        "--model_path", model_path,
        "--threads", str(config.threads),
        "--p_tokens", str(p_tokens),
        "--n_tokens", str(n_tokens),
        "--timestamp", timestamp
    ]

    # Start the inference script in the background
    try:
        subprocess.Popen(command, text=True)
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
            time.sleep(0.1)

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
                # 记录所有进程 GPU 显存
                gpu_memory_kb = get_gpu_memory_usage_for_single_process_kb(inference_pid)
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
    log_path_dict = monitor_inference_memory(ConfigUtils.load_config("config_win.json"), p_tokens=512, n_tokens=128)