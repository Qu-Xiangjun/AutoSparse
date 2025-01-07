"""Test how many time will use in every search part."""

import os, sys
import csv
import time
import numpy as np

from AutoSparse import *

autosparse_prefix = os.getenv("AUTOSPARSE_HOME")


def EvaluationTestSpMM(platform):
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
    statistics_times = []
    method = "q_sa_searching"  # "random_searching", "p_searching", "batch_p_searching", "sa_searching", "q_searching"

    result_filepath = os.path.join(
        autosparse_prefix,
        "python",
        "experiments",
        "motivation",
        "times_tests.csv",
    )
    with open(result_filepath, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Total Time", "Search", "RL Train", "Hardware Tests", "Other Time"])

    for mtx in mtx_names:
        mtx_filepath = os.path.join(
            autosparse_prefix, "dataset", "demo_dataset", mtx + ".csr"
        )
        num_row, num_col, num_nonezero = np.fromfile(mtx_filepath, count=3, dtype="<i4")
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
        C = Compute(A @ B)
        """Auto-Tune and excute"""
        A.LoadData(mtx_filepath)
        print(CreateSchedule(C).GenConfigCommand()[0])
        
        GET_TIMES = [0., 0., 0., 0.]
        sch = AutoTune(
            C,
            method=method,
            population_size=100,
            trial=70,
            early_stop=100,
            save_schedule_data=False,
            save_best_trace=False,
            save_dirpath=os.path.join(
                autosparse_prefix, "python", "experiments", platform + "_evaluation"
            ),
            GET_TIMES = GET_TIMES,
        )
        assert len(GET_TIMES) == 4
        GET_TIMES.append(GET_TIMES[0] - GET_TIMES[1])
        print(["Name", "Total Time", "Search", "RL Train", "Hardware Tests", "Other Time"])
        print([mtx] + GET_TIMES)

        with open(result_filepath, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [mtx] + GET_TIMES
            )


if __name__ == "__main__":
    EvaluationTestSpMM(platform="xeon")


# nohup python motivation_times_test.py > ./log/motivation_times_test_$(date +%Y%m%d%H%M).log 2>&1 &
