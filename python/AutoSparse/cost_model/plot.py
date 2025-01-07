import os, sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List
from datetime import datetime

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
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前日期并格式化为 YYYYMMDD 格式
        filename = f'loss_plot_{current_date}.png'
    if save_dir == None:
        save_dir = os.path.join(root, 'img')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    # 保存图形到本地
    plt.savefig(os.path.join(save_dir, filename), format='png', dpi=300, bbox_inches='tight')
    
    print(f"Image saved to {os.path.join(save_dir, filename)}")
    
    # 关闭图形以释放内存
    plt.close()


if __name__ == "__main__":
    # 示例数据
    train_losses = [[0.35, 0.25, 0.2, 0.18, 0.16], [0.3, 0.22, 0.18, 0.16, 0.14]]
    val_losses = [[0.37, 0.28, 0.22, 0.2, 0.18], [0.32, 0.25, 0.2, 0.18, 0.16]]
    net_names = ["Network A", "Network B"]

    # 调用函数
    LossPlot(train_losses, val_losses, net_names)
