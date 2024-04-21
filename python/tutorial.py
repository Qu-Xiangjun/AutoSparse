"""Test for SpMM"""
from AutoSparse import *

"""Axis declarations"""
i = Axis(64, ModeType.DENSE, "i")
k = Axis(128, ModeType.COMPRESSED, "k")
k_ = Axis(128, ModeType.DENSE, "k")
j = Axis(256, ModeType.DENSE, "j")
"""Tensor declaration"""
A = Tensor((i, k), is_sparse=True)
B = Tensor((k_, j), is_sparse=False)
"""Calculation declaration"""
C = Compute(A@B)
"""Auto-Tune and excute"""
sch = AutoTune(C, method = "Q_leaning", use_cost_model = True)
A.LoadData("/home/qxj/AutoSparse/dataset/demo_dataset/__test_matrix.csr")
func = Build(C)
time = func.Run()
print(time)





