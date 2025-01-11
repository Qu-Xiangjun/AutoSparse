"""Test for SpMM"""
import os
from AutoSparse import *

autosparse_prefix = os.getenv("AUTOSPARSE_HOME")

mtx_filepath = os.path.join(
    autosparse_prefix, "dataset", "demo_dataset", 'bcsstk38.csr'
)
num_row, num_col, num_nonezero = np.fromfile(
    mtx_filepath, count=3, dtype = '<i4'
)
M = int(num_row)
N = 256
K = int(num_col)

"""Axis declarations"""
i = Axis(M, ModeType.DENSE, "i")
k = Axis(K, ModeType.COMPRESSED, "k")
k_ = Axis(K, ModeType.DENSE, "k")
j = Axis(N, ModeType.DENSE, "j")
"""Tensor declaration"""
A = Tensor((i, k), is_sparse=True)
B = Tensor((k_, j), is_sparse=False)
"""Calculation declaration"""
C = Compute(A@B)
"""Auto-Tune and excute"""
A.LoadData(os.path.join(autosparse_prefix, 'dataset', 'demo_dataset', 'bcsstk38.csr'))
sch = AutoTune(C, method = "q_sa_searching", use_cost_model = True)
func = Build(sch)
time = func.Run()
print(time)





