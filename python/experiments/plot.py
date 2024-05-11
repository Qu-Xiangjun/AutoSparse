import os
import pandas as pd
import matplotlib.pyplot as plt

platform = "epyc"

def plot(mtx_name: str):
    # 定义CSV文件所在的文件夹路径
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    folder_path = os.path.join(autosparse_prefix, "python", "experiments", platform+"_evaluation_spmm", mtx_name)

    # 获取文件夹下所有的CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 遍历每个CSV文件
    for csv_file in csv_files:
        # 读取CSV文件
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        
        # 提取Round和Value列
        round_values = df['Round']
        values = df['Value']
        
        # 绘制折线图
        plt.plot(round_values, values, label=csv_file[:-30])  # 去除文件名后缀.csv并作为label
        
    # 添加标签和标题
    plt.xlabel('Round')
    plt.ylabel('Value')
    plt.title('Value vs Round')
    plt.legend()

    # 显示图表
    plt.savefig(os.path.join(folder_path, "value_vs_round_plot.png"))
    plt.clf()
    print(f"[Plot] Already save figure for {mtx_name}")

if __name__ == "__main__":
    mtx_names = [
        "bcsstk38",
        "mhd4800a",
        "conf5_0-4x4-18",
        "cca",
        "Trefethen_20000",
        "pf2177",
        "msc10848",
        "cfd1",
        "net100",
        "vanbody",
        "net150",
        "Chevron3_4x16_1",
        "vibrobox_1x1_0",
        "NACA0015_16x8_9",
        "nemspmm1_16x4_0",
        "Trec6_16x16_9",
        "crystk01_2x16_1",
        "t2dal_a_8x4_3",
        "EX1_8x8_4"
    ]
    for mtx_name in mtx_names:
        plot(mtx_name)