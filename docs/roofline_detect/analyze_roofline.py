import re

# 🧮 手动填入模型结构信息（或从 meta.json 自动读取）
layers = 24
hidden_size = 2048
flops_per_token = 2 * layers * hidden_size * hidden_size  # 粗略估算

# ⏱️ 读取推理时间
with open("runtime.txt") as f:
    runtime = float(f.read())

# 📦 解析 tegrastats 日志中 DRAM 使用峰值
dram_pattern = re.compile(r"RAM (\d+)/(\d+)MB")
with open("tegrastats.log") as f:
    lines = f.readlines()

dram_peak = 0
for line in lines:
    match = dram_pattern.search(line)
    if match:
        usage = int(match.group(1))
        if usage > dram_peak:
            dram_peak = usage

# 🚦 假设这部分都为模型造成的 DRAM traffic（保守估算）
dram_bytes = dram_peak * 1024 * 1024

# 📊 输出结果
oi = flops_per_token / dram_bytes
throughput = flops_per_token / runtime / 1e9  # GFLOPs

print(f"Total FLOPs: {flops_per_token:.2e}")
print(f"DRAM Traffic (Bytes): {dram_bytes:.2e}")
print(f"Inference Time (s): {runtime:.3f}")
print(f"Operational Intensity (FLOPs/Byte): {oi:.2f}")
print(f"Throughput (GFLOPs/s): {throughput:.2f}")
