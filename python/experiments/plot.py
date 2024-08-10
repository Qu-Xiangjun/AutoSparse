import os
import pandas as pd
import matplotlib.pyplot as plt

platform = "epyc"

def plot(mtx_name: str):
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    folder_path = os.path.join(autosparse_prefix, "python", "experiments", platform+"_evaluation_spmm", mtx_name)

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        
        round_values = df['Round']
        values = df['Value']
        
        plt.plot(round_values, values, label=csv_file[:-30])  # 去除文件名后缀.csv并作为label
        
    plt.xlabel('Round')
    plt.ylabel('Value')
    plt.title('Value vs Round')
    plt.legend()

    plt.savefig(os.path.join(folder_path, "value_vs_round_plot.png"))
    plt.clf()
    print(f"[Plot] Already save figure for {mtx_name}")

if __name__ == "__main__":
    mtx_names = [
        "bcsstk38",
        "mhd4800a",
        "cca"
    ]
    for mtx_name in mtx_names:
        plot(mtx_name)