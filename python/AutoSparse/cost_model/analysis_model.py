from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, parameter_count_table
import os, sys
import torch
from model import AutoSparseNet
from dataset_loader import *


file_dir = os.path.dirname(os.path.abspath(__file__))
root = os.getenv("AUTOSPARSE_HOME")

device = torch.device("cuda:0")

net = AutoSparseNet(
        in_channels=1,
        middle_channels=128,
        embedding_size=128,
        D=2,
        tensor_name_set=None,
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 假设你有一个名为model的模型实例
total_params = count_parameters(net)
print(f"Total number of parameters: {total_params}")
memory_params = sum([param.nelement() * param.element_size() for param in net.parameters()])
memory_buffers = sum([buf.nelement() * buf.element_size() for buf in net.buffers()])
memory_usage = (memory_params + memory_buffers) / 1024**2  # MB
print(f"Model size: {memory_usage:.2f} MB")

