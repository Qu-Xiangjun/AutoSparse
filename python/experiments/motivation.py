import os, sys
import csv
import time
import numpy as np
import torch

current = os.path.join(os.getcwd(), "python")
sys.path.append(current)

from AutoSparse import *
from AutoSparse.model import DQNAgentGroup

def Evaluation():
    mtx_names = ["obstclae", "delaunay_n15"]
    search_methods = ['q_sa_searching']
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    result = []
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
                           save_dirpath = os.path.join(autosparse_prefix, "python", "experiments", "motivation"),
                           eval_warm_times=5)
            func = Build(sch)
            time_val = min([func.Run() for i in range(10)])
            print(time_val)
            result.append([mtx, method, sch.GenConfigCommand()[1], func.origin_time, time_val])
    print(result)

def DoSchedule(cmd: str, mtx_names):
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    for mtx in mtx_names:
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
        sch = CreateSchedule(C)
        func = Build(sch)
        time_val = min([func.RunWithScheduleCommand(sch.GenConfigCommand()[0], cmd) for i in range (10)])
        print(f"{mtx} origin time = {func.origin_time}, run = {time_val:.8f}, speedup = {func.origin_time / time_val}")

if __name__ == "__main__":
    # Evaluation()

    # autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    # filepath = os.path.join(
    #     autosparse_prefix, "python", "experiments", "motivation", "msc10848",
    #     "q_sa_searchingschedule_data.pth"
    # )
    # _, cmd, value = DQNAgentGroup.LoadScheduleData(filepath)
    # print(cmd, value)
    # mtx_names = ["obstclae","cca","msc10848"]
    # DoSchedule(cmd, mtx_names)

    """
    2 i 2 i0 128 i1 256 k 2 k0 128 k1 256 1 A 4 i0 i1 k0 k1 A 4 i0 1 i1 0 k0 2 k1 3 1 j 2 j0 1 j1 128 6 i0 i1 j0 k0 k1 j1 i0 j1 i1 256 32 4
    2 i 2 i0 256 i1 128 k 2 k0 128 k1 256 1 A 4 i0 i1 k0 k1 A 4 i0 1 i1 2 k0 4 k1 3 1 j 2 j0 2 j1 64 6 i0 i1 k0 k1 j0 j1 i0 None j1 64 32 4
    2 i 2 i0 256 i1 32 k 2 k0 4096 k1 2 1 A 4 i0 k0 i1 k1 A 4 i0 0 i1 1 k0 1 k1 0 1 j 2 j0 2 j1 64 6 i0 k0 i1 k1 j0 j1 i0 j1 k1 2 32 2
    """

    # num_row=40000
    # num_col=40000
    # num_nonezero=197608
    # mtx_names = ["obstclae"]
    # cmd = "2 i 2 i0 128 i1 256 k 2 k0 128 k1 256 1 A 4 i0 i1 k0 k1 A 4 i0 1 i1 0 k0 2 k1 3 1 j 2 j0 1 j1 128 6 i0 i1 j0 k0 k1 j1 i0 j1 i1 256 32 4"
    # for i in range(11):
    #     factor = pow(2, i)
    #     i0 = int((num_row + factor - 1) / factor)
    #     cmd = cmd.split(" ")
    #     cmd[4] = str(i0)
    #     cmd[6] = str(factor)
    #     cmd[-3] = str(factor)
    #     cmd = " ".join(cmd)
    #     print(cmd)
    #     DoSchedule(cmd, mtx_names)


    # num_row=49152
    # num_col=49152
    # num_nonezero=139264
    # mtx_names = ["cca"]
    # cmd = "2 i 2 i0 256 i1 128 k 2 k0 128 k1 256 1 A 4 i0 i1 k0 k1 A 4 i0 1 i1 2 k0 4 k1 3 1 j 2 j0 2 j1 64 6 i0 i1 k0 k1 j0 j1 i0 None j1 64 32 4"
    # for i in range(11):
    #     factor = pow(2, i)
    #     i0 = int((num_row + factor - 1) / factor)
    #     cmd = cmd.split(" ")
    #     cmd[4] = str(i0)
    #     cmd[6] = str(factor)
    #     cmd = " ".join(cmd)
    #     print(cmd)
    #     DoSchedule(cmd, mtx_names)

    num_row=10848
    num_col=10848
    num_nonezero=1229778
    mtx_names = ["msc10848"]
    cmd = "2 i 2 i0 256 i1 32 k 2 k0 4096 k1 2 1 A 4 i0 k0 i1 k1 A 4 i0 0 i1 1 k0 1 k1 0 1 j 2 j0 2 j1 64 6 i0 k0 i1 k1 j0 j1 i0 j1 k1 2 32 2"
    for i in range(11):
        factor = pow(2, i)
        i0 = int((num_row + factor - 1) / factor)
        cmd = cmd.split(" ")
        cmd[4] = str(i0)
        cmd[6] = str(factor)
        cmd = " ".join(cmd)
        print(cmd)
        DoSchedule(cmd, mtx_names)


