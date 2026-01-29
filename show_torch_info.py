import platform
import torch

platform = platform.system()
print(f"Platform: {platform}")
print(f"torch version: {torch.__version__}")
if platform == "Darwin":
    print(f"torch.mps.is_available: {torch.mps.is_available()}")
elif platform == "Windows" or "Linux" and torch.cuda.is_available():
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"cuda version: {torch.version.cuda}")
    print(f"cudnn version: {torch.backends.cudnn.version()}")