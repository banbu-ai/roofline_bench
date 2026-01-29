import os
import sys
# 将当前脚本文件所在目录的父目录添加到系统路径
# project/test_inference/current_file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_inference.analyze_inference_win import batch_inference_analysis
from config import Config
from utils.config_utils import ConfigUtils


def qwen2_batch_inference_analysis(config: Config, p_tokens_list: list, n_tokens_list: list, prec: str = "f16"):
    # Qwen2.5-1.5B-Instruct
    config.model_name = "Qwen2.5-1.5B-Instruct"
    config.model_info.hidden_size = 1536
    config.model_info.num_attention_heads = 12
    config.model_info.num_key_value_heads = 2
    model_dir = os.path.dirname(config.model_path)
    if prec == "f16":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-f16.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")
    elif prec == "q8_0":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-q8_0.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.].")
    elif prec == "q4_k_m":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-q4_k_m.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")

def qwen3_batch_inference_analysis(config: Config, p_tokens_list: list, n_tokens_list: list, prec: str = "f16"):
    # Qwen3-0.6B
    config.model_name = "Qwen3-0.6B"
    config.model_info.hidden_size = 1024
    config.model_info.num_attention_heads = 16
    config.model_info.num_key_value_heads = 8
    model_dir = os.path.dirname(config.model_path)
    if prec == "f16":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-f16.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")
    elif prec == "q8_0":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-q8_0.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")
    elif prec == "q4_k_m":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-q4_k_m.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name}.")

def llama3_batch_inference_analysis(config: Config, p_tokens_list: list, n_tokens_list: list, prec: str = "f16"):
    # Llama-3.2-1B-Instruct
    config.model_name = "Llama-3.2-1B-Instruct"
    config.model_info.hidden_size = 2048
    config.model_info.num_attention_heads = 32
    config.model_info.num_key_value_heads = 8
    model_dir = os.path.dirname(config.model_path)
    if prec == "f16":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-f16.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")
    elif prec == "q8_0":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-q8_0.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")
    elif prec == "q4_k_m":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-q4_k_m.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")

def plm_batch_inference_analysis(config: Config, p_tokens_list: list, n_tokens_list: list, prec: str = "f16"):
    # PLM-1.8B-Instruct
    config.model_name = "PLM-1.8B-Instruct"
    config.model_info.hidden_size = 2048
    config.model_info.num_attention_heads = 16
    config.model_info.num_key_value_heads = 16
    config.model_info.qk_nope_head_dim = 128
    config.model_info.qk_rope_head_dim = 64
    model_dir = os.path.dirname(config.model_path)
    if prec == "f16":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-f16.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")
    elif prec == "q8_0":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-q8_0.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")
    elif prec == "q4_k_m":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-q4_k_m.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")

def fox_batch_inference_analysis(config: Config, p_tokens_list: list, n_tokens_list: list, prec: str = "f16"):
    # Fox-1-1.6B
    config.model_name = "Fox-1-1.6B"
    config.model_info.hidden_size = 2048
    config.model_info.num_attention_heads = 16
    config.model_info.num_key_value_heads = 4
    model_dir = os.path.dirname(config.model_path)
    if prec == "f16":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-f16.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")
    elif prec == "q8_0":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-q8_0.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")
    elif prec == "q4_k_m":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-q4_k_m.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")

def smollm2_batch_inference_analysis(config: Config, p_tokens_list: list, n_tokens_list: list, prec: str = "f16"):
    # SmolLM2-1.7B-Instruct
    config.model_name = "SmolLM2-1.7B-Instruct"
    config.model_info.hidden_size = 2048
    config.model_info.num_attention_heads = 32
    config.model_info.num_key_value_heads = 32
    model_dir = os.path.dirname(config.model_path)
    if prec == "f16":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-f16.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")
    elif prec == "q8_0":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-q8_0.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")
    elif prec == "q4_k_m":
        for layer in range(16, 33, 2):
            model_name = f"{config.model_name}-{layer}-layers-q4_k_m.gguf"
            next_model_path = os.path.join(model_dir, model_name)
            config.model_path = next_model_path
            config.model_info.num_hidden_layers = layer
            print(f"Running batch inference analysis for {layer} layers...")
            batch_inference_analysis(config=config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list)
        print(f"Finished running batch inference analysis for {config.model_name} {prec}.")

if __name__ == '__main__':
    n_tokens_list = [128, 256, 512, 1024, 2048, 4096]
    p_tokens_list = [128]
    qwen2_config = ConfigUtils.load_config("config_win/config_qwen2.json")
    qwen3_config = ConfigUtils.load_config("config_win/config_qwen3.json")
    llama3_config = ConfigUtils.load_config("config_win/config_llama3.json")
    plm_config = ConfigUtils.load_config("config_win/config_plm.json")
    fox_config = ConfigUtils.load_config("config_win/config_fox1.json")
    # qwen2_batch_inference_analysis(config=qwen2_config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, prec="f16")
    # qwen2_batch_inference_analysis(config=qwen2_config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, prec="q8_0")
    # qwen2_batch_inference_analysis(config=qwen2_config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, prec="q4_k_m")
    # qwen3_batch_inference_analysis(config=qwen3_config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, prec="f16")
    # qwen3_batch_inference_analysis(config=qwen3_config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, prec="q8_0")
    qwen3_batch_inference_analysis(config=qwen3_config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, prec="q4_k_m")
    # llama3_batch_inference_analysis(config=llama3_config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, prec="f16")
    # llama3_batch_inference_analysis(config=llama3_config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, prec="q8_0")
    # llama3_batch_inference_analysis(config=llama3_config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, prec="q4_k_m")
    # plm_batch_inference_analysis(config=plm_config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, prec="f16")
    # plm_batch_inference_analysis(config=plm_config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, prec="q8_0")
    # plm_batch_inference_analysis(config=plm_config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, prec="q4_k_m")
    # fox_batch_inference_analysis(config=fox_config, p_tokens_list=n_tokens_list, n_tokens_list=n_tokens_list, prec="f16")
    # fox_batch_inference_analysis(config=fox_config, p_tokens_list=n_tokens_list, n_tokens_list=n_tokens_list, prec="q8_0")
    # fox_batch_inference_analysis(config=fox_config, p_tokens_list=n_tokens_list, n_tokens_list=n_tokens_list, prec="q4_k_m")
    n_tokens_list = [128, 256, 512, 1024, 2048, 4096]
    p_tokens_list = [4096]
    qwen3_batch_inference_analysis(config=qwen3_config, p_tokens_list=p_tokens_list, n_tokens_list=n_tokens_list, prec="q4_k_m")