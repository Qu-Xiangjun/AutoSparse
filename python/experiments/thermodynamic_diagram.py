import torch
import random
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv

plt.figure(figsize=(12, 6))

def hash_mapping(params, k):
    memory_size = int(int(params[1]) * int(params[3]) / (int(params[-2])+1) / int(params[-1]))
    dict_encode = {'i0':1, 'j0':2, 'j1':3, 'k0':4, 'i1':5, 'k1':6}
    format_encode = 0
    for axis in params[14:20]:
        format_encode = format_encode*10 + dict_encode[axis]
    format_encode_mode = 0
    for mode in params[8:11]:
        format_encode_mode = format_encode_mode*10 + int(mode)+1
    i = memory_size % k
    j = (format_encode + format_encode_mode) % k
    return i, j

def hash_mapping_waco(params, k):
    memory_size = int(int(params[0]) * int(params[1]) / (1) / int(params[-1]))
    dict_encode = {'i0':1, 'j0':2, 'j1':3, 'k0':4, 'i1':5, 'k1':6}
    format_encode = 0
    for axis in params[3:9]:
        format_encode = format_encode*10 + dict_encode[axis]
    format_encode_mode = 0
    for mode in params[9:13]:
        format_encode_mode = format_encode_mode*10 + int(mode)+1
    i = memory_size % k
    j = (format_encode + format_encode_mode) % k
    return i, j

def draw_thermodynamic_diagram(filepath: str, k: int = 30):
    data = torch.load(filepath) # List[List[str, float]]

    sorted_data = sorted(data, key=lambda x: x[1])
    data = sorted_data[:int(len(sorted_data))]
    
    grid = np.zeros((k, k))
    
    for params, time in data:
        params = params.split()
        candidate_params_idx = [4, 6, 10, 12, 16, 17, 18, 19,23, 25, 27, 29, 34, 36, 38, 39 ,40, 41, 42, 43, 44, 45, 46, 47, 49]
        candidate_params = [params[i] for i in candidate_params_idx]
        i, j = hash_mapping(candidate_params, k)
        
        if grid[i, j] == 0 or time < grid[i, j]:
            grid[i, j] = time
    
    for i in range(k):
        for j in range(k):
            if grid[i, j] >= 0.000001:
                grid[i, j] = 1 / grid[i, j]
    
    print(np.max(grid))
    print(np.min(grid))

    # plt.subplot(1, 2, 1)
    plt.imshow(grid, cmap='viridis', vmin=0.7, vmax=1, interpolation='nearest')
    plt.colorbar(shrink=0.95)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlim(0, k-1)
    plt.ylim(0, k-1)
    # plt.xlabel('Configure the hash X')
    # plt.ylabel('Configure the hash Y')
    # plt.show()
    folder_path = os.path.join(os.path.dirname(filepath), "")
    plt.savefig(os.path.join(folder_path, "Thermodynamic.pdf"), format='pdf')
    plt.clf()

def save_data_to_csv(filepath: str, k: int):
    data = torch.load(filepath)
    folder_path = os.path.join(os.path.dirname(filepath), "")
    with open(os.path.join(folder_path, "Thermodynamic_data.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(["Parameter i0",
                         "P i1",
                         "P k0",
                         "P k1",
                         "P order0",
                         "P order1",
                         "P order2",
                         "P order3",
                         "P mode0",
                         "P mode1",
                         "P mode2",
                         "P mode3",
                         "P j0",
                         "P j1",
                         "P lorder0",
                         "P lorder1",
                         "P lorder2",
                         "P lorder3",
                         "P lorder4",
                         "P lorder5",
                         "P para",
                         "P vec",
                         "P unroll",
                         "P un_fac",
                         "P parchunk",
                          "Mapped X", "Mapped Y", "Mapped", "Execution Time"])
        
        for params_str, time in data:
            params = params_str.split()
            candidate_params_idx = [4, 6, 10, 12, 16, 17, 18, 19,23, 25, 27, 29, 34, 36, 38, 39 ,40, 41, 42, 43, 44, 45, 46, 47, 49]
            candidate_params = [params[i] for i in candidate_params_idx]
            i, j = hash_mapping(candidate_params, k)
            writer.writerow([*candidate_params, i, j, i*100+j, time])


def draw_thermodynamic_diagram_waco(filepath: str, k: int = 30):
    data = torch.load(filepath) # List[List[str, float]]
    sorted_data = sorted(data, key=lambda x: x[1])
    data = sorted_data[:int(len(sorted_data))]
    
    grid = np.zeros((k, k))
    
    for params, time in data:
        params = params.split()
        i, j = hash_mapping_waco(params, k)
        
        if grid[i, j] == 0 or time < grid[i, j]:
            grid[i, j] = time

    for i in range(k):
        for j in range(k):
            if grid[i, j] >= 0.000001:
                grid[i, j] =  1 / grid[i, j]
    
    
    print(np.max(grid))
    print(np.min(grid))

    # plt.subplot(1, 2, 2)
    plt.imshow(grid , vmin=0.7, vmax=1, cmap='viridis', interpolation='nearest')
    plt.colorbar(shrink=0.95)
    plt.tick_params(axis='both', labelsize=12)
    plt.ylim(0, k-1)
    plt.xlim(0, k-1)
    # plt.xlabel('Configure the hash X')
    # plt.ylabel('Configure the hash Y')
    # plt.show()
    folder_path = os.path.join(os.path.dirname(filepath), "")
    plt.savefig(os.path.join(folder_path, "Thermodynamic.pdf"), format='pdf')
    plt.clf()

def save_data_to_csv_waco(filepath: str, k: int):
    data = torch.load(filepath)
    folder_path = os.path.join(os.path.dirname(filepath), "")
    with open(os.path.join(folder_path, "Thermodynamic_data.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow([
                         "P i1",
                         "P k1",
                         "P j1",
                         "P lorder0",
                         "P lorder1",
                         "P lorder2",
                         "P lorder3",
                         "P lorder4",
                         "P lorder5",
                         "P mode0",
                         "P mode1",
                         "P mode2",
                         "P mode3",
                         "P para",
                         "P nt",
                         "P parchunk",
                          "Mapped X", "Mapped Y", "Mapped", "Execution Time"])
        
        for params, time in data:
            params = params.split()
            i, j = hash_mapping_waco(params, k)
            writer.writerow([*params, i, j, i*100+j, time])


k = 40

autosparse_prefix = os.getenv("AUTOSPARSE_HOME")

draw_thermodynamic_diagram(os.path.join(
    autosparse_prefix, 'python', 'experiments', 'xeon_platinum8272cl_evaluation_spmm', 
    'mhd4800a', 'q_sa_searching_False_2025_02_26_15_52_57.pth'
), k=k)
# save_data_to_csv(os.path.join(
#     autosparse_prefix, 'python', 'experiments', 'xeon_platinum8272cl_evaluation_spmm', 
#     'mhd4800a', 'q_sa_searching_True_2025_02_26_16_46_07.pth'
# ), k=k)

# baseline for waco

draw_thermodynamic_diagram_waco(os.path.join(
    autosparse_prefix, 'python', 'experiments', 'xeon_platinum8272cl_evaluation_spmm', 
    'mhd4800a', 'q_sa_searching_True_2025_02_26_16_46_07.pth'
), k=k)
# save_data_to_csv_waco(os.path.join(
#     autosparse_prefix, 'baseline', 'waco', 'SpMM', 'xeon_evaluation',
#     'cca', 'schedule_data_2024_04_30_20_21_05.pth'
# ), k=k)