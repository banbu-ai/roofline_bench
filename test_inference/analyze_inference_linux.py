import json
import os
import time
import sys
# 将当前脚本文件所在目录的父目录添加到系统路径
# project/test_inference/current_file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from monitor_inference_gpu_memory_linux import monitor_inference_memory
from utils.cal_flops_utils import AttentionFlopsCalculator
from utils.common_utils import CommonUtils
from utils.config_utils import ConfigUtils
from utils.date_time_utils import DateTimeUtils


def get_peak_memory_kb(log_path: str) -> float:
    return CommonUtils.find_csv_max(log_path)

def get_peak_memory_byte(log_path: str) -> float:
    return CommonUtils.convert_kb_to_byte(CommonUtils.find_csv_max(log_path))

def get_inference_info(log_path: str) -> dict | None:
    try:
        with open(log_path, encoding='gbk') as f:
            inference_data = json.load(f)
            prompt = inference_data[0]
            generate = inference_data[1]
            cpu_info = prompt.get('cpu_info')
            gpu_info = prompt.get('gpu_info')
            backends = prompt.get('backends')
            model_filename = prompt.get('model_filename')
            model_type = prompt.get('model_type')
            model_n_params = prompt.get('model_n_params')
            n_prompt = prompt.get('n_prompt')
            prompt_ts = prompt.get('avg_ts')
            n_gen = generate.get('n_gen')
            generate_ts = generate.get('avg_ts')
            return {
                "cpu_info": cpu_info,
                "gpu_info": gpu_info,
                "backends": backends,
                "model_filename": model_filename,
                "model_type": model_type,
                "model_n_params": model_n_params,
                "n_prompt": n_prompt,
                "prompt_ts": prompt_ts,
                "n_gen": n_gen,
                "generate_ts": generate_ts
            }
    except FileNotFoundError:
        print("Inference log not found.")
        return None

def get_runtime(log_path: str) -> float | None:
    try:
        with open(log_path) as f:
            runtime = float(f.read())
            return runtime
    except FileNotFoundError:
        print("Runtime file not found.")
        return None

def get_inference_runtime_memory_usage_log_path(log_path_dict: dict) -> tuple:
    runtime_log_path = log_path_dict["runtime_log_path"]
    inference_log_path = log_path_dict["inference_log_path"]
    memory_usage_log_path = log_path_dict["memory_usage_log_path"]
    return runtime_log_path, inference_log_path, memory_usage_log_path

def cal_memory_traffic_kb(peak_memory_kb: float, runtime: float) -> float:
    return peak_memory_kb * runtime

def cal_memory_traffic_byte(peak_memory_byte: float, runtime: float) -> float:
    return peak_memory_byte * runtime

def cal_inference_flops(p_tokens: int, n_tokens:int, **model_info) -> float:
    model_info['p_tokens'] = p_tokens
    model_info['n_tokens'] = n_tokens
    calculator = AttentionFlopsCalculator()
    calculator.set_params(**model_info)
    total_flops = calculator.cal_total_flops()
    return total_flops

def cal_inference_flops_per_token(p_tokens: int, n_tokens:int, **model_info) -> float:
    model_info['p_tokens'] = p_tokens
    model_info['n_tokens'] = n_tokens
    calculator = AttentionFlopsCalculator()
    calculator.set_params(**model_info)
    flops_per_token = calculator.cal_flops_per_token()
    return flops_per_token

def cal_operational_intensity(total_flops: float, memory_traffic_byte: float) -> float:
    oi = total_flops / memory_traffic_byte # Operational Intensity (FLOPs/Byte)
    return oi

def cal_performance_flops_per_sec(total_flops: float, runtime: float) -> float:
    performance = total_flops / runtime # FLOPS
    return performance

def cal_performance_gflops_per_sec(total_flops: float, runtime: float) -> float:
    flops_per_sec = cal_performance_flops_per_sec(total_flops, runtime)
    return flops_per_sec / 1e9 # GFLOPS

def save_analysis_result(config: Config, data: str, add_timestamp: bool = False):
    analysis_dir = config.analysis_dir
    model_path = config.model_path
    model_filename = os.path.basename(model_path)
    if add_timestamp:
        analysis_path = os.path.join(analysis_dir, model_filename + "_inference_analysis_" + DateTimeUtils.get_current_timestamp() + ".json")
    else:
        analysis_path = os.path.join(analysis_dir, model_filename + "_inference_analysis" + ".json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(json.loads(data), f, indent=4, ensure_ascii=False)

def single_inference_analysis(config: Config, p_tokens: int, n_tokens: int, dev: str, save_json:bool = True) -> str:
    result_list = []
    log_path_dict = monitor_inference_memory(config=config, p_tokens=p_tokens, n_tokens=n_tokens, dev=dev)
    time.sleep(1)  # 防止日志文件未生成
    runtime_log_path, inference_log_path, memory_usage_log_path = get_inference_runtime_memory_usage_log_path(log_path_dict)
    inference_info = get_inference_info(inference_log_path)
    runtime = get_runtime(runtime_log_path)
    model_info = config.model_info.to_dict()
    total_flops = cal_inference_flops(p_tokens=p_tokens, n_tokens=n_tokens, **model_info)
    flops_per_token = cal_inference_flops_per_token(p_tokens=p_tokens, n_tokens=n_tokens, **model_info)
    peak_mem_byte = get_peak_memory_byte(memory_usage_log_path) # Byte
    memory_traffic_byte = cal_memory_traffic_byte(peak_memory_byte=peak_mem_byte, runtime=runtime)
    oi = cal_operational_intensity(total_flops=total_flops, memory_traffic_byte=memory_traffic_byte)
    perf = cal_performance_flops_per_sec(total_flops=total_flops, runtime=runtime)
    analysis_dict = {
        "model_info": model_info,
        "inference_info": inference_info,
        "total_flops": total_flops,
        "flops_per_token": flops_per_token,
        "peak_memory_kb": CommonUtils.convert_byte_to_kb(peak_mem_byte),
        "memory_traffic_kb": CommonUtils.convert_byte_to_kb(memory_traffic_byte),
        "runtime": runtime,
        "operational_intensity": oi,
        "performance": perf
    }
    result_list.append(analysis_dict)
    result_json = json.dumps(result_list, indent=4, ensure_ascii=False)
    if save_json:
        save_analysis_result(config=config, data=result_json)
    return result_json

def batch_inference_analysis(config: Config, p_tokens_list: list, n_tokens_list: list, dev: str, save_json=True) -> str:
    result_list = []
    if len(p_tokens_list) != len(n_tokens_list):
        raise ValueError("p_tokens_list 和 n_tokens_list 必须长度相同")
    for p_token, n_token in zip(p_tokens_list, n_tokens_list):
        analysis_json = single_inference_analysis(config=config, p_tokens=p_token, n_tokens=n_token, dev=dev, save_json=False)
        analysis_list = json.loads(analysis_json)
        result_list.append(analysis_list[0])
    result_json = json.dumps(result_list, indent=4, ensure_ascii=False)
    if save_json:
        save_analysis_result(config=config, data=result_json)
    return result_json


if __name__ == '__main__':
    # single inference analysis
    p_tokens = 512
    n_tokens = 128
    config_path = "config.json"
    config = ConfigUtils.load_config(config_path)

    analysis_json = single_inference_analysis(config=config, p_tokens=p_tokens, n_tokens=n_tokens, dev="cuda0")
    analysis_dict = json.loads(analysis_json)[0]
    total_flops = analysis_dict["total_flops"]
    flops_per_token = analysis_dict['flops_per_token']
    peak_mem = analysis_dict['peak_memory_kb']
    mem_traffic = analysis_dict['memory_traffic_kb']
    runtime = analysis_dict['runtime']
    oi = analysis_dict['operational_intensity']
    perf = analysis_dict['performance']
    print(f"Total FLOPs: {total_flops:.2f}")
    print(f"FLOPs Per Token: {flops_per_token:.2f}")
    print(f"Peak DRAM (MB): {peak_mem / 2**10:.2f}")
    print(f"DRAM Traffic (MB): {mem_traffic / 2**10:.2f}")
    print(f"Inference Time (s): {runtime:.2f}")
    print(f"Operational Intensity (FLOPs/Byte): {oi:.2f}")
    print(f"Performance (GFLOPs/s): {perf / 10 ** 9:.2f}")


    # batch inference analysis
    # n_tokens_list = [128, 128, 4096, 4096]
    # p_tokens_list = [128, 4096, 128, 4096]
    # config_path = "config.json"
    # config = ConfigUtils.load_config(config_path)
    #
    # batch_analysis_json = batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, dev="cuda0")


    # different threads test
    # p_tokens = 512
    # n_tokens = 2048
    # config_path = "config.json"
    # config = ConfigUtils.load_config(config_path)
    # result_list = []
    # for thread in [1, 2, 4, 8, 16, 32]:
    #     config.threads = thread
    #     analysis_json = single_inference_analysis(config=config, p_tokens=p_tokens, n_tokens=n_tokens, dev="cuda0", save_json=False)
    #     analysis_list = json.loads(analysis_json)
    #     result_list.append(analysis_list[0])
    # result_json = json.dumps(result_list, indent=4, ensure_ascii=False)
    # save_analysis_result(config=config, data=result_json, add_timestamp=True)