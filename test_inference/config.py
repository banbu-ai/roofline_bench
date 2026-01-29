import os
import json
from typing import Dict, Any

class ModelInfo:
    """
    模型架构信息。
    """
    def __init__(self,
                 attention_type: str,
                 hidden_size: int,
                 num_hidden_layers: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 qk_nope_head_dim: int,
                 qk_rope_head_dim: int,
                 is_modified_arch: bool):
        self.attention_type = attention_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.is_modified_arch = is_modified_arch

    def __repr__(self) -> str:
        # 为 ModelInfo 对象提供一个清晰的字符串表示
        # 使用 f-string 格式化所有属性
        return (f"ModelInfo(attention_type='{self.attention_type}', "
                f"hidden_size={self.hidden_size}, "
                f"num_hidden_layers={self.num_hidden_layers}, "
                f"num_attention_heads={self.num_attention_heads}, "
                f"num_key_value_heads={self.num_key_value_heads}, "
                f"qk_nope_head_dim={self.qk_nope_head_dim}, "
                f"qk_rope_head_dim={self.qk_rope_head_dim}, "
                f"is_modified_arch={self.is_modified_arch})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attention_type": self.attention_type,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "is_modified_arch": self.is_modified_arch
        }

class Config:
    """
    代表 config.json 文件的实体类。
    """
    def __init__(self,
                 log_dir: str,
                 analysis_dir: str,
                 threads: int,
                 model_name: str,
                 model_type: str,
                 model_path: str,
                 model_info: ModelInfo):
        self.log_dir = os.path.expanduser(log_dir)
        self.analysis_dir = os.path.expanduser(analysis_dir)
        self.threads = threads
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = os.path.expanduser(model_path)
        self.model_info = model_info

    def __repr__(self) -> str:
        # 为 Config 对象提供一个清晰的字符串表示。
        # 使用 f-string 格式化所有属性，包括 ModelInfo 子对象
        return (f"Config(log_dir='{self.log_dir}', "
                f"analysis_dir='{self.analysis_dir}', "
                f"threads={self.threads}, "
                f"model_name='{self.model_name}', "
                f"model_type='{self.model_type}', "
                f"model_path='{self.model_path}', "
                f"model_info={self.model_info})")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """
        从字典创建 Config 实例。
        """
        model_info_data = data.get('model_info', {})
        model_info = ModelInfo(
            attention_type=model_info_data.get('attention_type'),
            hidden_size=model_info_data.get('hidden_size'),
            num_hidden_layers=model_info_data.get('num_hidden_layers'),
            num_attention_heads=model_info_data.get('num_attention_heads'),
            num_key_value_heads=model_info_data.get('num_key_value_heads'),
            qk_nope_head_dim=model_info_data.get('qk_nope_head_dim'),
            qk_rope_head_dim=model_info_data.get('qk_rope_head_dim'),
            is_modified_arch=model_info_data.get('is_modified_arch')
        )
        return cls(
            log_dir=data.get('log_dir'),
            analysis_dir=data.get('analysis_dir'),
            threads=data.get('threads'),
            model_name=data.get('model_name'),
            model_type=data.get('model_type'),
            model_path=data.get('model_path'),
            model_info=model_info
        )

    @classmethod
    def from_json(cls, file_path: str) -> 'Config':
        """
        从 JSON 文件加载并创建 Config 实例。
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_dir": self.log_dir,
            "analysis_dir": self.analysis_dir,
            "threads": self.threads,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "model_info": self.model_info.to_dict()
        }


if __name__ == '__main__':
    config = Config.from_json('config_mac/config_plm.json')
    print(config.to_dict())
    print(config.model_info.to_dict())