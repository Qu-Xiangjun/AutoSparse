import torch
import torch.autograd.profiler as profiler
import numpy as np
import os, sys
import random

from AutoSparse.utils import get_coo_from_csr_file

autosparse_prefix = os.getenv("AUTOSPARSE_HOME")


def SpMM(csr_filename, M = 128, dtype = torch.float32):
    filepath = os.path.join(
        autosparse_prefix, "dataset", "demo_dataset", csr_filename + ".csr"
    )
    num_row, num_col, nnz, coo = get_coo_from_csr_file(filepath)
    print(f"[MKL_Pytorch] {csr_filename}: num_row={num_row} num_col={num_col}"
          f" nnz={nnz}")
    A = torch.zeros((num_row, num_col), dtype=dtype)
    A[coo[:,0], coo[:,1]] = random.randint(0, 1)
    A_sp = A.to_sparse_csr()
    B = torch.ones((num_col, M), dtype=dtype)

    for i in range(50):
        torch.matmul(A, B)

    # 使用profiler来分析torch.matmul()操作
    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("matmul"):
            C = torch.matmul(A, B)
    print(C[0][0])
    print(C[0][1])
    print(C[0][2])
    print(C.shape)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=5))

if __name__ == "__main__":
    SpMM("nemspmm1_16x4_0")