import os
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_dir = os.path.expanduser("E:/Code/llm_inference_roofline_detect/models")
# model_name = "Qwen2.5-1.5B-Instruct"
model_name = "Qwen3-0.6B"
# model_name = "Llama-3.2-1B-Instruct"
# model_name = "PLM-1.8B-Instruct"
# model_name = "Fox-1-1.6B"
# model_name = "SmolLM2-1.7B-Instruct"
model_path = os.path.join(model_dir, model_name)
modify_model_dir = os.path.expanduser(f"E:/Code/llm_inference_roofline_detect/models/modified")


def modify_model_layers(model_path, num_modify_layers, modify_model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    original_layers = config.num_hidden_layers
    print(f"原始模型层数: {original_layers}")

    # 加载原始模型，不传入修改后的config
    # 这样模型会按照原始配置的层数加载所有权重
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto'
    )

    current_layers = len(model.model.layers)
    print(f"当前加载模型的实际层数: {current_layers}")

    # 调整模型层数
    if num_modify_layers < current_layers:
        model.model.layers = model.model.layers[:num_modify_layers]
        print(f"模型层已截断至 {num_modify_layers} 层")

    elif num_modify_layers > current_layers:
        layers_to_add = num_modify_layers - current_layers
        print(f"需要增加 {layers_to_add} 层")
        last_layer = model.model.layers[-1]
        new_layers = [deepcopy(last_layer) for _ in range(layers_to_add)]
        model.model.layers.extend(new_layers)
        print(f"已成功添加 {layers_to_add} 层，新层是原始最后一层的副本")

    config.num_hidden_layers = num_modify_layers
    model.save_pretrained(modify_model_path)
    config.save_pretrained(modify_model_path)

    # 保存修改后的模型、配置和tokenizer
    model.save_pretrained(modify_model_path)
    config.save_pretrained(modify_model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(modify_model_path)

    print(f"修改后的模型已成功保存至: {modify_model_path}")
    print("-" * 300)


for num_modify_layers in range(2, 65, 2):
    modify_model_path = os.path.join(modify_model_dir, f"{model_name}-{num_modify_layers}-layers")
    modify_model_layers(model_path, num_modify_layers, modify_model_path)