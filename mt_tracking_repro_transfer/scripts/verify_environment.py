import sys
import numpy as np

print("python:", sys.version.replace("\n", " "))
print("numpy:", np.__version__)

try:
    import torch
except Exception as exc:
    print("torch import: FAILED", repr(exc))
    raise

print("torch:", torch.__version__)
print("torch compiled cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn((2048, 2048), device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    print("gpu tensor test: PASS", tuple(y.shape), y.device)
else:
    print("gpu tensor test: SKIPPED")
