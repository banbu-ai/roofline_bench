import argparse
import subprocess
import sys
import time
import os


def single_inference(log_dir: str, model_path:str, threads: int, p_tokens: int, n_tokens: int, timestamp: str):
    """
    Run a single inference using llama.cpp and log the results.
    """
    # Use a variable to handle the log directory path
    log_dir = log_dir
    log_path = os.path.join(log_dir, "inference_" + timestamp + ".json")
    runtime_path = os.path.join(log_dir, "runtime_" + timestamp + ".txt")

    # Access the configuration values
    model_path = model_path

    # --- Run the command and capture output ---
    print(f"Running llama.cpp inference...")

    start_time = time.time()

    command = [
        "llama-bench",
        "-m", model_path,
        "-p", str(p_tokens),
        "-n", str(n_tokens),
        "-t", str(threads),
        "-o", "json"
    ]
    print(f"Executing command: {' '.join(command)}")

    result = subprocess.run(command, capture_output=True, text=True, check=True)

    end_time = time.time()
    # --- Process output and calculate runtime ---
    # Save the JSON output to a file
    with open(log_path, "w") as f:
        f.write(result.stdout)

    runtime_sec = end_time - start_time

    # Save the runtime to a file
    with open(runtime_path, "w") as f:
        f.write(str(runtime_sec))
    print(f"Inference time: {runtime_sec:.2f} seconds")
    print(f"Log saved to {log_path}.")


# Deprecated
# def batch_inference(config_path: str, p_tokens_list: list, n_tokens_list: list) -> list:
#     """
#     对多个 token 对执行推理并返回包含详细信息的列表。
#     参数:
#         config_path (str): 配置文件路径。
#         p_tokens_list (list): 包含多个 p_tokens 的列表。
#         n_tokens_list (list): 包含多个 n_tokens 的列表。
#     返回:
#         list: 一个包含推理结果字典列表，每个字典包含 p_token、n_token 和 runtime_sec。
#     """
#     results = []
#     if len(p_tokens_list) != len(n_tokens_list):
#         raise ValueError("p_tokens_list 和 n_tokens_list 必须长度相同")
#
#     for p_token, n_token in zip(p_tokens_list, n_tokens_list):
#         runtime_sec = single_inference(config_path, p_token, n_token)
#
#         # 手动构建包含所有信息的字典
#         inference_details = {
#             "p_token": p_token,
#             "n_token": n_token,
#             "runtime_sec": runtime_sec
#         }
#         results.append(inference_details)
#
#     return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single inference with specified parameters.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory to save logs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file.")
    parser.add_argument("--threads", type=int, required=True, help="Number of threads to use for inference.")
    parser.add_argument("--p_tokens", type=int, required=True, help="Number of prompt tokens.")
    parser.add_argument("--n_tokens", type=int, required=True, help="Number of generated tokens.")
    parser.add_argument("--timestamp", type=str, required=False, help="Timestamp for logging purposes.")

    args = parser.parse_args()

    try:
        single_inference(args.log_dir,args.model_path, args.threads, args.p_tokens, args.n_tokens, args.timestamp)
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        sys.exit(1)
