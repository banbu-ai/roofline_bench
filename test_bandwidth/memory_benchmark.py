import numpy as np
import time


def test_memory_bandwidth(array_size_gb=2.0, num_iterations=5):
    """
    通过对大型NumPy数组进行操作来测试内存带宽。

    参数:
    - array_size_gb (float): 用于测试的每个数组的大小（单位：GB）。
                           建议大小至少为2GB，以确保超过CPU缓存。
    - num_iterations (int): 运行测试的次数，以获得更稳定的平均结果。

    返回:
    - float: 估算出的内存带宽（单位：GB/s）。
    """
    # 将GB转换为字节数，并计算数组元素的数量（np.float64是8字节）
    array_size_bytes = int(array_size_gb * 1024 ** 3)
    num_elements = array_size_bytes // 8  # 每个float64是8个字节

    print(f"--- 开始内存带宽测试 ---")
    print(f"每个数组大小: {array_size_gb:.2f} GB")
    print(f"数组元素数量: {num_elements:,}")
    print(f"迭代次数: {num_iterations}")
    print("正在初始化数组...")

    # 1. 创建三个大型数组
    # 使用随机数据来防止任何可能的内存压缩优化
    try:
        a = np.random.rand(num_elements)
        b = np.random.rand(num_elements)
        c = np.zeros(num_elements)
    except MemoryError:
        print("\n错误：内存不足！无法分配所需大小的数组。")
        print("请尝试减小 `array_size_gb` 的值。")
        return None

    print("数组初始化完成。开始基准测试...")

    # 预热一次，确保所有设置都已就绪
    c = a + b

    # 2. 核心测试循环
    start_time = time.monotonic()
    for _ in range(num_iterations):
        # 这个操作是内存密集型的：
        # - 读取数组 a (array_size_gb)
        # - 读取数组 b (array_size_gb)
        # - 写入数组 c (array_size_gb)
        # 总数据传输量 = 3 * array_size_gb
        c = a + b
    end_time = time.monotonic()

    elapsed_time = end_time - start_time

    # 3. 计算结果
    # 总共移动的数据量 = 迭代次数 * 3个数组 * 每个数组的大小
    total_data_moved_gb = num_iterations * 3 * array_size_gb

    # 带宽 = 总数据量 / 总时间
    bandwidth = total_data_moved_gb / elapsed_time

    print("\n--- 测试完成 ---")
    print(f"总耗时: {elapsed_time:.4f} 秒")
    print(f"总数据传输量: {total_data_moved_gb:.2f} GB")
    print(f"估算内存带宽: {bandwidth:.2f} GB/s")

    return bandwidth


if __name__ == "__main__":
    # 您可以调整这里的参数
    # 对于有 16GB RAM 的 MacBook，使用 2GB 或 4GB 的数组是安全的
    # 如果您的 RAM 更大（如 32GB 或 64GB），可以尝试更大的值
    test_memory_bandwidth(array_size_gb=4.0, num_iterations=10)