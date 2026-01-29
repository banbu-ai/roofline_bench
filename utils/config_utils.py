import json
import os

from test_inference.config import Config
from utils.common_utils import CommonUtils


class ConfigUtils:
    """
    一个用于加载和处理配置文件的工具类。
    """

    @staticmethod
    def load_config(config_path: str) -> Config:
        """
        从 JSON 文件加载配置并返回一个 Config 类的实例。

        该方法会自动创建日志和分析目录。
        如果文件不存在或格式错误，则会退出程序。

        :param config_path: 配置文件的路径。
        :return: 一个 Config 类的实例。
        """
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config_dict = json.load(config_file)

                # 使用 Config 类的 from_dict 工厂方法来创建实例
                config_instance = Config.from_dict(config_dict)

                # 针对不同操作系统规范化路径
                config_instance.log_dir = CommonUtils.normalize_path(os.path.expanduser(config_instance.log_dir))
                config_instance.analysis_dir = CommonUtils.normalize_path(os.path.expanduser(config_instance.analysis_dir))
                config_instance.model_path = CommonUtils.normalize_path(os.path.expanduser(config_instance.model_path))

                # 确保 log_dir 和 analysis_dir 存在
                ConfigUtils.create_dir(config_instance.log_dir)
                ConfigUtils.create_dir(config_instance.analysis_dir)

                return config_instance

        except FileNotFoundError:
            print("Error: config.json not found. Please create the file with the required settings.")
            exit(1)
        except json.JSONDecodeError:
            print("Error: Failed to parse config.json. Please check for syntax errors.")
            exit(1)

    @staticmethod
    def create_dir(dir_path: str):
        """
        创建目录，如果它不存在。
        :param dir_path: 要创建的目录路径。
        """
        os.makedirs(dir_path, exist_ok=True)


# 示例用法
if __name__ == '__main__':
    try:
        config = ConfigUtils.load_config('../test_inference/config.json')

        print(f"log_dir: {config.log_dir}")
        print(f"model_name: {config.model_name}")
        print(f"num_hidden_layers: {config.model_info.num_hidden_layers}")
        print(f"is_modified_arch: {config.model_info.is_modified_arch}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")