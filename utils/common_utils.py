import os
import pandas as pd
from typing import List, Dict


class CommonUtils:
    @staticmethod
    def normalize_path(path: str) -> str:
        """
        Normalizes a given file path by replacing forward slashes with
        the correct operating system-specific separator.

        Args:
            path: The file path string to normalize.

        Returns:
            The normalized file path string.
        """
        return path.replace("/", os.sep)

    @staticmethod
    def find_csv_max(file_path: str) -> float | None:
        try:
            # 使用 read_csv 读取文件，header=None 表示没有标题行
            # skip_blank_lines=True 忽略空行
            data = pd.read_csv(file_path, header=None, skip_blank_lines=True)
            return float(data.max().max())
        except FileNotFoundError:
            print("File not found.")
            return None
        except pd.errors.EmptyDataError:
            print("File is empty or does not contain valid data.")
            return None

    @staticmethod
    def convert_byte_to_kb(byte_value: float) -> float:
        """
        Convert bytes to kilobytes.
        :param byte_value: Value in bytes.
        :return: Value in kilobytes.
        """
        return byte_value / 2 ** 10

    @staticmethod
    def convert_kb_to_byte(kb_value: float) -> float:
        """
        Convert kilobytes to bytes.
        :param kb_value: Value in kilobytes.
        :return: Value in bytes.
        """
        return kb_value * 2 ** 10

    @staticmethod
    def convert_kb_to_mb(kb_value: float) -> float:
        """
        Convert kilobytes to megabytes.
        :param kb_value: Value in kilobytes.
        :return: Value in megabytes.
        """
        return kb_value / 2 ** 10

    @staticmethod
    def convert_mb_to_kb(mb_value: float) -> float:
        """
        Convert megabytes to kilobytes.
        :param mb_value: Value in megabytes.
        :return: Value in kilobytes.
        """
        return mb_value * 2 ** 10

    @staticmethod
    def convert_mb_to_bytes(mb_value: float) -> float:
        """
        Convert megabytes to bytes.
        :param mb_value: Value in megabytes.
        :return: Value in bytes.
        """
        return mb_value / 2 ** 20

    @staticmethod
    def convert_byte_to_mb(byte_value: float) -> float:
        """
        Convert bytes to megabytes.
        :param byte_value: Value in bytes.
        :return: Value in megabytes.
        """
        return byte_value / 2 ** 20

    @staticmethod
    def find_file_in_directory(directory: str, strings_to_find: List[str], extension: str) -> List[str]:
        """
        在指定目录中查找包含特定字符和特定扩展名的文件，不遍历子目录。

        Args:
            directory (str): 要搜索的目录路径。
            strings_to_find (List[str]): 包含要搜索的字符串的列表。
            extension (str): 要过滤的文件扩展名（例如：'.txt', '.py'）。

        Returns:
            list: 找到的文件的完整路径列表。
        """
        found_files = []
        for item in os.listdir(directory):
            full_path = os.path.join(directory, item)
            # 确保是文件且扩展名匹配
            if os.path.isfile(full_path) and item.endswith(extension):
                # 检查文件名是否包含任何一个搜索字符串
                if all(s in item for s in strings_to_find):
                    found_files.append(full_path)
        return found_files

    @staticmethod
    def find_file_in_directory_and_subdirectories(directory: str, strings_to_find: List[str], extension: str) -> list:
        """
        查找指定目录及其子目录中包含任意特定字符和特定扩展名的所有文件。

        Args:
            directory (str): 要搜索的目录路径。
            strings_to_find (List[str]): 包含要搜索的字符串的列表。
            extension (str): 要过滤的文件扩展名（例如：'.txt', '.py'）。

        Returns:
            list: 找到的文件的完整路径列表。
        """
        found_files = []
        # os.walk() 遍历所有子目录
        for root, _, files in os.walk(directory):
            for file in files:
                # 检查文件扩展名是否匹配
                if file.endswith(extension):
                    # 检查文件名是否包含 strings_to_find 列表中的任何一个字符串
                    if all(s in file for s in strings_to_find):
                        # 使用 os.path.join() 创建完整路径
                        full_path = os.path.join(root, file)
                        found_files.append(full_path)
        return found_files

    @staticmethod
    def find_files_by_model(model_name_list: List[str], analysis_dir: str) -> Dict[str, List[str]]:
        """
        根据模型名称列表，调用 find_files() 函数查找并返回相应的 JSON 文件路径字典。
        Args:
            model_name_list (List[str]): 要查找的模型名称列表。
            analysis_dir (str): 搜索的根目录。

        Returns:
            Dict[str, List[str]]: 一个字典，键是模型名称，值是找到的 JSON 文件路径列表。
        """
        model_files_dict = {}
        for model_name in model_name_list:
            # 调用 find_files 函数来查找文件
            found_files = CommonUtils.find_file_in_directory(analysis_dir, [model_name], '.json')
            model_files_dict[model_name] = found_files
        return model_files_dict

    @staticmethod
    def sort_files_by_precision(model_files_dict: Dict[str, List[str]]):
        """
        对模型文件字典中的文件列表进行原地排序，确保按FP16, Q8, Q4的顺序。
        """
        # 定义精度排序的映射
        precision_order = {'_f16_': 0, '_q8_0_': 1, '_q4_k_m_': 2}

        for model_name, file_list in model_files_dict.items():
            # 使用自定义的排序键对文件列表进行排序
            file_list.sort(key=lambda f: next((precision_order[p] for p in precision_order if p in f), 99))

    @staticmethod
    def remove_model_size(model_name: str) -> str:
        """
        Removes the model size (e.g., '1.5B', '1B') from a single model name string.

        Args:
            model_name: A string representing the model's full name.

        Returns:
            The model name with the size removed.
        """
        parts = model_name.split('-')
        # Check if the last part is a model size (ends with 'B' or 'b' followed by something, or just 'B' or 'b')
        # A more robust check might involve regular expressions, but this simple check works for the provided examples.
        if len(parts) > 1 and parts[-1].lower().endswith('b'):
            return "-".join(parts[:-1])
        # For cases like "Qwen2.5-1.5B-Instruct", we need to check the part before "Instruct"
        elif len(parts) > 2 and parts[-2].lower().endswith('b'):
            parts.pop(-2)  # Remove the size part
            return "-".join(parts)
        elif len(parts) > 1 and parts[-1].lower().endswith('m'):
            return "-".join(parts[:-1])
        elif len(parts) > 2 and parts[-2].lower().endswith('m'):
            parts.pop(-2)  # Remove the size part
            return "-".join(parts)
        else:
            return model_name

    @staticmethod
    def batch_remove_model_size(model_name_list: list) -> list:
        """
        Removes the model size from each model name in a list.

        Args:
            model_name_list: A list of model name strings.

        Returns:
            A new list with the model sizes removed.
        """
        return [CommonUtils.remove_model_size(name) for name in model_name_list]