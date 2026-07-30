import torch

print("=== CHECKING GPU VIA PYTORCH CUDA ===")
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Count:", torch.cuda.device_count())
    print("Device Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA Device Not Found")
