import os
import torch
from copy import deepcopy
from AutoSparse.utils import logger_init
from AutoSparse.model import cuda_device_id
import logging

class Config:
    def __init__(self):
        # Train data config
        self.dataset_dirname_prefixs_lst = ["epyc_7543/spmm"]
        self.batch_size = 512
        self.model_save_per_epoch = 5
        self.leaning_rate = 0.02
        self.epoch = 100
        self.device = torch.device(
            "cuda:" + str(cuda_device_id) if torch.cuda.is_available() else "cpu"
        )
        self.loss_fn = "LambdaRankingLoss"
        self.data_handle_method = "relative_max"

        self.is_save_loss_data = True

        # Model config
        self.net_name_prefix = "AutoSparseNet"
        self.tensor_name_set = None
        self.is_waco_net = False
        self.is_net_forward1 = True
        self.in_channels = 1
        self.D = 2
        self.middle_channel_num = 64
        self.token_embedding_size = 128

    def LoggerInit(self):
        data_flag = "_".join(("_".join(self.dataset_dirname_prefixs_lst)).split(os.sep))
        root = os.getenv("AUTOSPARSE_HOME")
        log_dir = os.path.join(root, "python", "AutoSparse", "cost_model", "log")
        logger_init(
            self.net_name_prefix
            + "_cost_model_train_" 
            + "_"
            + str(self.is_waco_net)
            + "_"
            + str(self.is_net_forward1)
            + data_flag,
            log_dir=log_dir,
        )


def PrintConfig(config):
    msg = {
        "dataset_dirname_prefixs_lst": config.dataset_dirname_prefixs_lst,
        "batch_size": config.batch_size,
        "model_save_per_epoch": config.model_save_per_epoch,
        "leaning_rate": config.leaning_rate,
        "epoch": config.epoch,
        "is_save_loss_data": config.is_save_loss_data,
        "tensor_name_set": config.tensor_name_set,
        "is_waco_net": config.is_waco_net,
        "is_net_forward1": config.is_net_forward1,
        "in_channels": config.in_channels,
        "D": config.D,
        "middle_channel_num": config.middle_channel_num,
        "token_embedding_size": config.token_embedding_size,
        "loss_fn": config.loss_fn,
        "data_handle_method": config.data_handle_method,
    }
    for k, v in msg.items():
        print(k, " : ", str(v))