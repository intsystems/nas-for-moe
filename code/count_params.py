import sys
sys.path.insert(0, '/pbabkin/main/mipt/nas-for-moe/code')
sys.path.insert(0, '/pbabkin/main/mipt/nas-for-moe/code/nas_moe')
import torch, torch.nn as nn
from nni_utils import OPS  # или from nas_moe.nni_utils import OPS

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

C = 32      # подставьте нужное число каналов
stride = 1
affine = True

for name, factory in OPS.items():
    try:
        m = factory(C, stride, affine)
        print(f"{name:20s} -> params: {count_params(m)}")
    except Exception as e:
        print(f"{name:20s} -> ERROR: {e}")
