import os
import torch
import numpy as np
from typing import *

from AutoSparse.cost_model.utils import LossPlot



def PlotCostModelLoss(model_names: List[str], net_names):
    root = os.getenv("AUTOSPARSE_HOME")
    
    model_dir = os.path.join(root, 'python', 'AutoSparse', 'cost_model', 'model_log')
    save_dir = os.path.join(root, 'python', 'experiments', 'loss_img')
    global_train_losses = []
    global_val_losses = []
    for idx, mname in enumerate(model_names):
        mpath = os.path.join(model_dir, mname)
        state_dict = torch.load(mpath)
        train_losses_batch = state_dict['losses']['train_losses_batch']
        val_losses_batch = state_dict['losses']['val_losses_batch']
        train_losses = []
        val_losses = []
        train_batch_num = int(len(train_losses_batch) / 75)
        val_batch_num = int(len(val_losses_batch) / 75)
        for i in range(75):
            train_losses.append(np.array(train_losses_batch[i*train_batch_num:(i+1)*train_batch_num]).mean().item())
            val_losses.append(np.array(val_losses_batch[i*val_batch_num:(i+1)*val_batch_num]).mean().item())
        LossPlot([train_losses], [val_losses], net_names[idx:idx+1], save_dir, mname[0:-4]+'.png')

        global_train_losses.append(train_losses)
        global_val_losses.append(val_losses)
    
    LossPlot(global_train_losses, global_val_losses, net_names, save_dir, 'full_'+model_names[-1][0:-4]+'.png')

if __name__ == "__main__":
    model_names = [
        'autosparse_net_1_1_xeon_platinum8272cl_spmm_epoch_74_20250105_100509.pth',
        'autosparse_net_1_1_xeon_platinum8272cl_spmv_epoch_74_20250106_100352.pth',
        'autosparse_net_1_1_xeon_platinum8272cl_spmv_xeon_platinum8272cl_spmm_xeon_platinum8272cl_sddmm_epoch_74_20250106_031941.pth'
    ]
    net_names = [
        'AutoSparseNet_SpMM',
        'AutoSparseNet_SpMV',
        'AutoSparseNet_Mixed'
    ]
    PlotCostModelLoss(model_names, net_names)