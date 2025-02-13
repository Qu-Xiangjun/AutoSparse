import os, sys
import numpy as np
from tqdm import tqdm
from AutoSparse import *

import requests

autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
platform = "epyc_7543" # xeon_e52620v4 epyc_7543

task_name = "spmm" # spmv sddmm

# webhook
robutl = "https://open.feishu.cn/open-apis/bot/v2/hook/8d1dc46d-b71a-4363-9d01-99a3e9ba9706"

def send_message(text):
    url=robutl
    try:
        body = {"msg_type": "text", "content": {"text": text}}
        requests.post(url=url, json=body, verify=False)
    except Exception as e:
        print("[Robot Error] Send request error.")


def make_dataset():
    total_filepath = os.path.join(autosparse_prefix, "dataset", "total.txt")
    dataset_dirpath = os.path.join(autosparse_prefix, "cost_model_dataset_build", "search_base", platform, task_name)
    os.makedirs(dataset_dirpath, exist_ok=True)

    with open(total_filepath) as f :
        mtx_names = f.read().splitlines() 
    
    # Already collect the performace data
    have_collected_mtx = get_all_files_in_directory(dataset_dirpath)
    have_collected_mtx = set(have_collected_mtx)

    for idx, mtx in enumerate(tqdm(mtx_names, total=len(mtx_names))):
        if (mtx+'.txt') in have_collected_mtx or (mtx+mtx+'.txt')in have_collected_mtx:
            continue
        mtx_filepath = os.path.join(
            autosparse_prefix, "dataset", "total_dataset", mtx+'.csr'
        )
        num_row, num_col, nnz = np.fromfile(
            mtx_filepath, count=3, dtype = '<i4'
        )
        print(f"{mtx} : num_row={num_row}, num_col={num_col}, nnz={nnz}")

        if task_name == "spmm":
            """Axis declarations"""
            i = Axis(int(num_row), ModeType.DENSE, "i")
            k = Axis(int(num_col), ModeType.COMPRESSED, "k")
            k_ = Axis(int(num_col), ModeType.DENSE, "k")
            j = Axis(128, ModeType.DENSE, "j")
            """Tensor declaration"""
            A = Tensor((i, k), is_sparse=True)
            B = Tensor((k_, j), is_sparse=False)
            """Calculation declaration"""
            Res_Tensor = Compute(A@B)
            """Auto-Tune and excute"""
            A.LoadData(mtx_filepath)
            print(CreateSchedule(Res_Tensor).GenConfigCommand()[0])
        elif task_name == "spmv":
            """Axis declarations"""
            i = Axis(int(num_row), ModeType.DENSE, "i")
            j = Axis(int(num_col), ModeType.COMPRESSED, "j")
            j_ = Axis(int(num_col), ModeType.DENSE, "j")
            """Tensor declaration"""
            A = Tensor((i, j), is_sparse=True)
            B = Tensor((j_, ), is_sparse=False)
            """Calculation declaration"""
            Res_Tensor = Compute(A@B)
            """Auto-Tune and excute"""
            A.LoadData(mtx_filepath)
            print(CreateSchedule(Res_Tensor).GenConfigCommand()[0])
        elif task_name == "sddmm":
            """Axis declarations"""
            i = Axis(int(num_row), ModeType.DENSE, "i")
            j = Axis(int(num_col), ModeType.COMPRESSED, "j")
            i_ = Axis(int(num_row), ModeType.DENSE, "i")
            k_ = Axis(128, ModeType.DENSE, "k")
            k__ = Axis(128, ModeType.DENSE, "k")
            j_ = Axis(int(num_col), ModeType.DENSE, "j")
            i__ = Axis(int(num_row), ModeType.DENSE, "i")
            j__ = Axis(int(num_col), ModeType.COMPRESSED, "j")
            """Tensor declaration"""
            A = Tensor((i, j), is_sparse=True)
            B = Tensor((i_, k_), is_sparse=False)
            C = Tensor((k__, j_), is_sparse=False)
            """Calculation declaration"""
            Res_Tensor = Compute(A*(B@C), is_sparse=True, format=(i__, j__))
            """Auto-Tune and excute"""
            A.LoadData(mtx_filepath)
            Res_Tensor.LoadData(mtx_filepath)
            print(CreateSchedule(Res_Tensor).GenConfigCommand())
        else:
            raise ValueError()
        
        AutoTune(
            Res_Tensor, method = "q_sa_searching", population_size=28, trial = 20,
            early_stop=100, save_schedule_data=True,
            save_dirpath = os.path.join(autosparse_prefix, "cost_model_dataset_build", "search_base", platform, task_name)
        )

        send_message(f"[{platform}, {task_name}] {idx}/{len(mtx_names)} program sampling for {mtx} "
                    "(num_row={num_row}, num_col={num_col}, nnz={nnz}) successed!")


if __name__ == "__main__":
    try:
        send_message(f"********************************\n <at user_id=\"all\">Master</at> : {task_name} for {platform} start.")
        print(f"{task_name} for {platform} start.")
        make_dataset()
        send_message(f"********************************\n <at user_id=\"all\">Master</at> : {task_name} for {platform} have completed.")
    except Exception as e:
        error_message = f"Error Type: {type(e).__name__}, Message: {e}, Args: {e.args}"
        send_message("<at user_id=\"all\">Master</at> :" + error_message)

# nohup python make_dataset.py > ./log/evaluation_epyc_$(date +%Y%m%d%H%M).log 2>&1 & 
