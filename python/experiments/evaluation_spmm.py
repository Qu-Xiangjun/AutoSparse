
import os, sys
import csv
import time
import numpy as np
import torch

current = os.path.join(os.getcwd(), "python")
sys.path.append(current)

from AutoSparse import *

def Evaluation(platform):
    mtx_names = [
        # "bcsstk38",
        # "mhd4800a",
        # "conf5_0-4x4-18",
        # "cca",
        # "Trefethen_20000",
        # "pf2177",
        # "msc10848",
        # "cfd1",
        # "net100",
        # "vanbody",
        # "net150",
        # "Chevron3_4x16_1",
        # "vibrobox_1x1_0",
        # "NACA0015_16x8_9",
        "nemspmm1_16x4_0",
        # "Trec6_16x16_9",
        # "crystk01_2x16_1",
        # "t2dal_a_8x4_3",
        # "EX1_8x8_4"
    ]
    search_methods = ["random_searching", "p_searching", 'q_sa_searching'] # "random_searching", "p_searching", "batch_p_searching", "sa_searching", "q_searching"
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")

    result_filepath = os.path.join(autosparse_prefix, "python", "experiments", platform + "_evaluation", "result.csv")
    with open(result_filepath, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["name", "method", "config command", "csr_time", "best_time"])

    for mtx in mtx_names:
        for method in search_methods:
            mtx_filepath = os.path.join(
                autosparse_prefix, "dataset", "demo_dataset", mtx+'.csr'
            )
            num_row, num_col, num_nonezero = np.fromfile(
                mtx_filepath, count=3, dtype = '<i4'
            )
            print(f"num_row={num_row}, num_col={num_col}, num_nonezero={num_nonezero}")
            """Axis declarations"""
            i = Axis(int(num_row), ModeType.DENSE, "i")
            k = Axis(int(num_col), ModeType.COMPRESSED, "k")
            k_ = Axis(int(num_col), ModeType.DENSE, "k")
            j = Axis(128, ModeType.DENSE, "j")
            """Tensor declaration"""
            A = Tensor((i, k), is_sparse=True)
            B = Tensor((k_, j), is_sparse=False)
            """Calculation declaration"""
            C = Compute(A@B)
            """Auto-Tune and excute"""
            A.LoadData(mtx_filepath)
            print(CreateSchedule(C).GenConfigCommand()[0])

            sch = AutoTune(C, method = method, population_size=100, trial = 100,
                           early_stop=100, save_schedule_data=True, save_best_trace=True,
                           save_dirpath = os.path.join(autosparse_prefix, "python", "experiments", platform + "_evaluation"))
            func = Build(sch)
            time_val = min([func.Run() for i in range(10)])
            print(time_val)

            result_filepath = os.path.join(autosparse_prefix, "python", "experiments", platform + "_evaluation", "result.csv")
            with open(result_filepath, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([mtx, method, sch.GenConfigCommand()[1], func.origin_time, time_val])

if __name__ == "__main__":
    Evaluation(platform = "xeon")




# nohup python evaluation.py > ./log/evaluation_$(date +%Y%m%d%H%M).log 2>&1 & 