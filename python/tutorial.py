"""Test for SpMM"""
from AutoSparse import *

"""Axis declarations"""
i = Axis(32, ModeType.DENSE, "i")
k = Axis(64, ModeType.COMPRESSED, "k")
k_ = Axis(64, ModeType.DENSE, "k")
j = Axis(256, ModeType.DENSE, "j")
"""Tensor declaration"""
A = Tensor((i, k), is_sparse=False)
B = Tensor((k_, j), is_sparse=True)
"""Calculation declaration"""
C = Compute(A@B)
"""Auto-Tune and excute"""
sch = AutoTune(C, method = "Q_leaning", use_cost_model = True)
func = Build(sch)
# func.Run()





