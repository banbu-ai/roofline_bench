import torch
import sys

from utils.bandwidth_test_utils import BandwidthTestUtils

f = open("raspberry_pi_bandwidth_peak_flops_test.txt", 'w+')
sys.stdout = f

platform = "Linux"
chip = "Raspberry Pi 5"
print(f"Platform: {platform}")
print(f"Chip: {chip}")

theo_bandwidth, theo_peak_flops = BandwidthTestUtils.get_theo_bandwidth_peak_flops_by_chip_name(chip)
print(f"Theoretical memory bandwidth: {theo_bandwidth / (10 ** 9):.2f} GB/s")
print(f"Theoretical peak FLOPs (FP32): {theo_peak_flops / (10 ** 9):.2f} GFLOPS")

real_bandwidth = sum(BandwidthTestUtils.measure_memory_bandwidth_cpu() for _ in range(10)) / 10
real_flops_fp16 = sum(BandwidthTestUtils.measure_peak_flops_cpu(torch.float16) for _ in range(10)) / 10
real_flops_fp32 = sum(BandwidthTestUtils.measure_peak_flops_cpu(torch.float32) for _ in range(10)) / 10
print(f"Measured memory bandwidth: {real_bandwidth / (10 ** 9):.2f} GB/s")
print(f"Measured peak FLOPs (FP16): {real_flops_fp16 / (10 ** 9):.2f} GFLOPS")
print(f"Measured peak FLOPs (FP32): {real_flops_fp32 / (10 ** 9):.2f} GFLOPS")