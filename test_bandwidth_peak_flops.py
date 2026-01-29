import torch

from utils.bandwidth_test_utils import BandwidthTestUtils
from utils.device_info_utils import DeviceInfoUtils

platform = DeviceInfoUtils.get_platform()
chip = DeviceInfoUtils.get_chip_model()
print(f"Platform: {platform}")
print(f"Chip: {chip}")

theo_bandwidth, theo_peak_flops = BandwidthTestUtils.get_theo_bandwidth_peak_flops()
print(f"Theoretical memory bandwidth: {theo_bandwidth / (10 ** 9):.2f} GB/s")
print(f"Theoretical peak FLOPs (FP32): {theo_peak_flops / (10 ** 12):.2f} TFLOPS")

real_bandwidth = sum(BandwidthTestUtils.measure_memory_bandwidth(platform) for _ in range(10)) / 10
real_flops_fp16 = sum(BandwidthTestUtils.measure_peak_flops(platform, torch.float16) for _ in range(10)) / 10
real_flops_fp32 = sum(BandwidthTestUtils.measure_peak_flops(platform, torch.float32) for _ in range(10)) / 10
print(f"Measured memory bandwidth: {real_bandwidth / (10 ** 9):.2f} GB/s")
print(f"Measured peak FLOPs (FP16): {real_flops_fp16 / (10 ** 12):.2f} TFLOPS")
print(f"Measured peak FLOPs (FP32): {real_flops_fp32 / (10 ** 12):.2f} TFLOPS")

# Platform: Darwin
# Chip: Apple M1 Pro
# Theoretical memory bandwidth: 204.80 GB/s
# Theoretical peak FLOPs: 5.20 TFLOPS
# Measured memory bandwidth: 120.03 GB/s
# Measured peak FLOPs (FP16): 4.61 TFLOPS
# Measured peak FLOPs (FP32): 4.31 TFLOPS

# Platform: Linux
# Chip: NVIDIA GeForce RTX 3090
# Theoretical memory bandwidth: 936.20 GB/s
# Theoretical peak FLOPs (FP32): 35.58 TFLOPS
# Measured memory bandwidth: 560.02 GB/s
# Measured peak FLOPs (FP16): 66.20 TFLOPS
# Measured peak FLOPs (FP32): 24.28 TFLOPS

# Platform: Windows
# Chip: NVIDIA GeForce RTX 3070 Ti Laptop GPU
# Theoretical memory bandwidth: 448.00 GB/s
# Theoretical peak FLOPs (FP32): 16.60 TFLOPS
# Measured memory bandwidth: 217.00 GB/s
# Measured peak FLOPs (FP16): 31.76 TFLOPS
# Measured peak FLOPs (FP32): 9.51 TFLOPS

# Platform: Linux
# Chip: Orin
# Theoretical memory bandwidth: 68.22 GB/s
# Theoretical peak FLOPs (FP32): 1.28 TFLOPS
# Measured memory bandwidth: 59.40 GB/s
# Measured peak FLOPs (FP16): 9.56 TFLOPS
# Measured peak FLOPs (FP32): 1.34 TFLOPS

# Platform: Linux
# Chip: Raspberry Pi 5
# Theoretical memory bandwidth: 17.10 GB/s
# Theoretical peak FLOPs (FP32): 153.60 GFLOPS
# Measured memory bandwidth: 3.98 GB/s
# Measured peak FLOPs (FP16): 1.48 GFLOPS
# Measured peak FLOPs (FP32): 78.56 GFLOPS