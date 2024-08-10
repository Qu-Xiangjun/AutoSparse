
import os, sys
import csv
import time
import numpy as np
import torch

current = os.path.join(os.getcwd(), "python")
sys.path.append(current)

from AutoSparse import *

def EvaluationSpMM(platform):
    mtx_names = [
        'strides_mask'
    ]
    search_methods = ['rl_sa_searching'] # "random_searching", "all_anns_searching", "batch_all_anns_searching", "sa_searching", "rl_searching"
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")

    folder_path = os.path.join(autosparse_prefix, "python", "experiments", 'usages', platform + "_evaluation_sddmm")
    os.makedirs(folder_path, exist_ok=True)
    result_filepath = os.path.join(folder_path, "result.csv")
    with open(result_filepath, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["name", "method", "config command", "csr_time", "best_time"])

    for mtx in mtx_names:
        for method in search_methods:
            mtx_filepath = os.path.join(
                autosparse_prefix, "python", "experiments", 'usages', mtx+'.csr'
            )
            num_row, num_col, num_nonezero = np.fromfile(
                mtx_filepath, count=3, dtype = '<i4'
            )
            print(f"{mtx} {method} num_row={num_row}, num_col={num_col}, num_nonezero={num_nonezero}")
            """Axis declarations"""
            i = Axis(int(num_row), ModeType.DENSE, "i")
            j = Axis(int(num_col), ModeType.COMPRESSED, "j")
            i_ = Axis(int(num_row), ModeType.DENSE, "i")
            k_ = Axis(256, ModeType.DENSE, "k")
            k__ = Axis(256, ModeType.DENSE, "k")
            j_ = Axis(int(num_col), ModeType.DENSE, "j")
            i__ = Axis(int(num_row), ModeType.DENSE, "i")
            j__ = Axis(int(num_col), ModeType.COMPRESSED, "j")
            """Tensor declaration"""
            A = Tensor((i, j), is_sparse=True)
            B = Tensor((i_, k_), is_sparse=False)
            C = Tensor((k__, j_), is_sparse=False)
            """Calculation declaration"""
            D = Compute(A*(B@C), is_sparse=True, format=(i__, j__))
            """Auto-Tune and excute"""
            A.LoadData(mtx_filepath)
            D.LoadData(mtx_filepath)
            print(CreateSchedule(D).GenConfigCommand())

            sch = AutoTune(D, method = method, population_size=100, trial = 100,
                           early_stop=100, save_schedule_data=True, save_best_trace=True,
                           save_dirpath = os.path.join(autosparse_prefix, "python", "experiments", 'usages', platform + "_evaluation_sddmm"))
            func = Build(sch)
            time_val = min([func.Run() for i in range(10)])
            print(time_val)

            result_filepath = os.path.join(autosparse_prefix, "python", "experiments", 'usages', platform + "_evaluation_sddmm", "result.csv")
            with open(result_filepath, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([mtx, method, sch.GenConfigCommand()[1], func.origin_time, time_val])

if __name__ == "__main__":
    EvaluationSpMM(platform = "xeon")




# nohup python evaluation_sddmm.py > ../log/usages_xeon_evaluation_sddmm_$(date +%Y%m%d%H%M).log 2>&1 & 