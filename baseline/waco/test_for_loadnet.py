import os, sys
import numpy as np
import hnswlib
from AutoSparse import *
from AutoSparse import cuda_device_id
from tqdm import tqdm
import multiprocessing
import heapq

from buildKNN import TrainingScheduleDataset
from model import *



autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
waco_prefix = os.path.join(autosparse_prefix, "baseline", "waco")
device = torch.device("cuda:" + str(cuda_device_id) if torch.cuda.is_available() else "cpu")

net = ResNet14SpMM(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
net = net.to(device)
state_dict = torch.load(os.path.join(waco_prefix, "SpMM", 'resnet.pth'), map_location=device)
# print(state_dict)

new_state_dict = {}
for key, value in state_dict.items():
    if key in net.state_dict():
        new_state_dict[key] = value

net.load_state_dict(new_state_dict)
net.eval()