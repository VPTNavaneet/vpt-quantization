import torch
import minerl
import cv2
import numpy as np

print("=== Mac Verification ===")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"MineRL: installed OK")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
if torch.backends.mps.is_available():
    print(f"Device: MPS (Apple Silicon GPU) ✅")
else:
    print(f"Device: CPU")
print("=== All checks passed! ===")
