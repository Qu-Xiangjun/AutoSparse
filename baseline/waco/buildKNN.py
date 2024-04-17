import os, sys
import numpy as np
import hnswlib
import torch
from tqdm import tqdm
import time

from model import *


current_directory = os.path.dirname(os.path.abspath(__file__))

class TrainingScheduleDataset(torch.utils.data.Dataset):
    def __init__(self, filename, task_name = "SpMM", extend=False):
        if task_name == "SpMM" or task_name == "SDDMM":
            split_ = [1<<p for p in range(17)]
            index_ = ['i1', 'i0', 'k1', 'k0', 'j1', 'j0']
            format_ = [0, 1] #(C,U)
            parnum_ = [48]
            parchunk_ = [1<<p for p in range(9)]
        elif task_name == "SpMV":
            split_ = [1<<p for p in range(17)]
            index_ = ['i1', 'i0', 'k1', 'k0']
            format_ = [0, 1] #(C,U)
            parnum_ = [48]
            parchunk_ = [1<<p for p in range(9)]
        else:
            assert False, "Error task name."

        schedules = []
        schedules_str = []

        with open(filename) as f :
            names = f.read().splitlines()
            uniqstr = set()
    
        waco_prefix = os.getenv("AUTOSPARSE_HOME")
        waco_prefix = os.path.join(waco_prefix, "baseline", "waco")
        print("Embedd all super schedule.")
        for name in tqdm(names, total=len(names)) : 
            with open(os.path.join(waco_prefix, task_name, "TrainingData", "CollectedData", name+".txt")) as f:
                lines = f.read().splitlines()
                lines = [line.split() for line in lines]
        
            for idx, line in enumerate(lines) :
                if (" ".join(line[:-2]) in uniqstr) : continue
                uniqstr.add(" ".join(line[:-2]))          

                if task_name == "SpMM" or task_name == "SDDMM":
                    i0s = split_.index(int(line[0]))
                    k0s = split_.index(int(line[1]))
                    j0s = split_.index(int(line[2]))
                    if int(line[0]) > 256 or int(line[1]) > 256 or int(line[2]) > 256:
                        continue

                    order = line[3:9]
                    perm = np.zeros((len(index_), len(index_)))
                    perm[index_.index(order[0]),0] = 1
                    perm[index_.index(order[1]),1] = 1
                    perm[index_.index(order[2]),2] = 1
                    perm[index_.index(order[3]),3] = 1
                    perm[index_.index(order[4]),4] = 1
                    perm[index_.index(order[5]),5] = 1
                    perm = perm.flatten()

                    i1f = format_.index(int(line[9]))
                    i0f = format_.index(int(line[10]))
                    k1f = format_.index(int(line[11]))
                    k0f = format_.index(int(line[12]))

                    p1 = index_.index(line[13])
                    p2 = parnum_.index(int(line[14]))
                    p3 = parchunk_.index(int(line[15]))

                    concat = np.array([i0s,k0s,j0s,
                                        perm,
                                        i1f,i0f,k1f,k0f,
                                        p1,p2,p3], dtype=object)
                elif task_name == "SpMV": 
                    i0s = split_.index(int(line[0]))
                    k0s = split_.index(int(line[1]))
                    if int(line[1]) > 256 or int(line[0]) > 256:
                        continue

                    order = line[2:6]
                    perm = np.zeros((len(index_), len(index_)))
                    perm[index_.index(order[0]),0] = 1
                    perm[index_.index(order[1]),1] = 1
                    perm[index_.index(order[2]),2] = 1
                    perm[index_.index(order[3]),3] = 1
                    perm = perm.flatten()

                    i1f = format_.index(int(line[6]))
                    i0f = format_.index(int(line[7]))
                    k1f = format_.index(int(line[8]))
                    k0f = format_.index(int(line[9]))

                    p1 = index_.index(line[10])
                    p2 = parnum_.index(int(line[11]))
                    p3 = parchunk_.index(int(line[12]))

                    concat = np.array([i0s,k0s,
                                        perm,
                                        i1f,i0f,k1f,k0f,
                                        p1,p2,p3], dtype=object)
                concat = np.hstack(concat)
                schedules.append(concat)
                schedules_str.append(" ".join(line[:-2]))

        schedules = np.stack(schedules, axis=0)
        self.schedules = schedules.astype(np.float32)
        self.schedules_str = schedules_str

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, idx):
        return self.schedules[idx], self.schedules_str[idx] 


def buildKNN(task_name):
    # task_name = "SpMM" # "SpMV" "SDDMM"
    waco_prefix = os.getenv("AUTOSPARSE_HOME")
    waco_prefix = os.path.join(waco_prefix, "baseline", "waco")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    schedules = TrainingScheduleDataset(os.path.join(waco_prefix, task_name, "TrainingData", "total.txt"), task_name)
    schedule_loader = torch.utils.data.DataLoader(schedules, batch_size=128, shuffle=False, num_workers=0)

    if (task_name == 'SpMM'):
        net = ResNet14SpMM(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    elif task_name == 'SpMV':
        net = ResNet14SpMV(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    elif task_name == 'SDDMM':
        net = ResNet14SDDMM(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    else:
        assert (False, "Error task name.")
    net = net.to(device)
    net.load_state_dict(torch.load(os.path.join(waco_prefix, task_name, 'resnet.pth')))
    net.eval()

    start = time.time()  
    names = []
    embeddings = [] 
    for batch_idx, (data, string) in tqdm(enumerate(schedule_loader), total=len(schedule_loader)):
        data = data.to(device)
        embedding = net.embed_super_schedule(data)
        embeddings.extend(embedding.detach().cpu().tolist())
        names.extend(string)
    embeddings = np.array(embeddings)
    print("Load Embedding : ", time.time()-start)

    num_elements = embeddings.shape[0]
    dim = embeddings.shape[1] 
    p = hnswlib.Index(space = 'l2', dim = dim) 
    p.init_index(max_elements = num_elements, ef_construction = 200, M = 32)

    start = time.time()
    p.add_items(embeddings, np.arange(num_elements))
    print("Gen Index : ", time.time()-start)
    p.save_index(os.path.join(waco_prefix, task_name, "hnsw_schedule.bin"))

if __name__ == "__main__":
    buildKNN("SpMM")
    buildKNN("SpMV")
    buildKNN("SDDMM")