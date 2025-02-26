import os, sys
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List
from datetime import datetime
from copy import deepcopy
import logging

from AutoSparse.cost_model.model import AutoSparseNet
from AutoSparse.cost_model.config import Config

file_dir = os.path.dirname(os.path.abspath(__file__))
root = os.getenv("AUTOSPARSE_HOME")


def LossPlot(
    train_losses: List[List[float]],
    val_losses: List[List[float]],
    net_names: List[str],
    save_dir: str = None,
    filename: str = None,
):
    if len(train_losses) != len(val_losses) or len(train_losses) != len(net_names):
        raise ValueError(
            "The number of networks in train_losses, val_losses and net_names must be the same."
        )

    colors = ["b", "g", "r", "c", "m", "y", "k"]
    linestyles = ["-", "--"]

    for i, name in enumerate(net_names):
        plt.plot(
            range(len(train_losses[i])),
            train_losses[i],
            color=colors[i % len(colors)],
            linestyle=linestyles[0],
            label=f"{name} train",
        )
        plt.plot(
            range(len(val_losses[i])),
            val_losses[i],
            color=colors[i % len(colors)],
            linestyle=linestyles[1],
            label=f"{name} val",
        )

    plt.title("Train-Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    ax = plt.gca()  # 获取当前坐标轴
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    max_epoch = max([len(losses) for losses in (train_losses + val_losses)])
    epoch_interval = 5
    epochs = range(0, max_epoch, epoch_interval)
    ax.set_xticks(epochs)

    if filename is None:
        current_date = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # 获取当前日期并格式化为 YYYYMMDD 格式
        filename = f"loss_plot_{current_date}.png"
    if save_dir == None:
        save_dir = os.path.join(root, "img")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 保存图形到本地
    plt.savefig(
        os.path.join(save_dir, filename), format="png", dpi=300, bbox_inches="tight"
    )

    print(f"Image saved to {os.path.join(save_dir, filename)}")

    # 关闭图形以释放内存
    plt.close()



def SaveModelAndConfig(net, config: Config, losses, filepath: str):
    state_dict = deepcopy(net.state_dict())
    checkpoint = {
        "model_state_dict": state_dict,
        "config": {
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
        },
        "losses": losses,
    }
    torch.save(checkpoint, filepath)


def LoadModelAndConfig(Net, config, filepath: str):
    if not os.path.exists(filepath):
        logging.error(f"Model don't exist: {filepath}")
        exit(1)
    state_dict = torch.load(filepath)
    model_state_dict = state_dict['model_state_dict']
    config_dict = state_dict['config']
    losses = state_dict['losses']

    config = Config()
    config.dataset_dirname_prefixs_lst = config_dict['dataset_dirname_prefixs_lst']
    config.batch_size = config_dict['batch_size']
    config.model_save_per_epoch = config_dict['model_save_per_epoch']
    config.leaning_rate = config_dict['leaning_rate']
    config.epoch = config_dict['epoch']
    config.is_save_loss_data = config_dict['is_save_loss_data']
    config.tensor_name_set = None
    config.is_waco_net = config_dict['is_waco_net']
    config.is_net_forward1 = config_dict['is_net_forward1']
    config.in_channels = config_dict['in_channels']
    config.D = config_dict['D']
    config.middle_channel_num = config_dict['middle_channel_num']
    config.token_embedding_size = config_dict['token_embedding_size']
    config.data_handle_method = config_dict['data_handle_method']
    config.loss_fn = config_dict['loss_fn']
    
    
    if Net == AutoSparseNet:
        net = Net(config)
    else:
        net = Net
    net.load_state_dict(model_state_dict)

    return net, losses


def SaveSparseMatrixFeatureVec(feature_tensor, mtx_name, dir_path):
    filepath = os.path.join(dir_path, mtx_name+'.pth')
    torch.save(feature_tensor, filepath)
    print(f"Fearture tensor has been saved to {filepath}")


def LoadSparseMatrixFeatureVec(mtx_name, dir_path):
    filepath = os.path.join(dir_path, mtx_name+'.pth')
    feature_tensor = torch.load(filepath)
    print(f"Fearture tensor has been loaded from {filepath}")
    return feature_tensor


if __name__ == "__main__":
    # 示例数据
    train_losses = [[0.35, 0.25, 0.2, 0.18, 0.16], [0.3, 0.22, 0.18, 0.16, 0.14]]
    val_losses = [[0.37, 0.28, 0.22, 0.2, 0.18], [0.32, 0.25, 0.2, 0.18, 0.16]]
    net_names = ["Network A", "Network B"]

    # 调用函数
    LossPlot(train_losses, val_losses, net_names)
