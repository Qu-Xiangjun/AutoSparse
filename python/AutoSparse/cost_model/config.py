import os
import torch
from copy import deepcopy
from AutoSparse.utils import logger_init
from AutoSparse.model import cuda_device_id


class Config:
    def __init__(self):
        # Train data config
        self.dataset_dirname_prefixs_lst = ["epyc_7543/spmm"]
        self.batch_size = 512
        self.model_save_per_epoch = 15
        self.leaning_rate = 0.02
        self.epoch = 75
        self.device = torch.device(
            "cuda:" + str(cuda_device_id) if torch.cuda.is_available() else "cpu"
        )
        self.loss_fn = "LambdaRankingLoss"
        self.data_handle_method = "relative_max"

        self.is_save_loss_data = True

        # Model config
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
            "cost_model_train_"
            + "_"
            + str(self.is_waco_net)
            + "_"
            + str(self.is_net_forward1)
            + data_flag,
            log_dir=log_dir,
        )
