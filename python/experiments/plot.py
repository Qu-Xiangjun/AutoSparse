import os
import pandas as pd
import matplotlib.pyplot as plt

# 定义CSV文件所在的文件夹路径
autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
folder_path = os.path.join(autosparse_prefix, "python", "experiments", "evaluation", "cca")

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
plt.savefig(os.path.join(folder_path,"value_vs_round_plot.png"))
