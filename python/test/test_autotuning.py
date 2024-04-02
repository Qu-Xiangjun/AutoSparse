"""Test for SpMM"""

import os, sys
current = os.path.join(os.getcwd(), "python")
sys.path.append(current)

from AutoSparse import *

def test_spmv_randomsearching():
    """Axis declarations"""
    i = Axis(64, ModeType.DENSE, "i")
    j = Axis(128, ModeType.COMPRESSED, "j")
    j_ = Axis(128, ModeType.DENSE, "j")
    """Tensor declaration"""
    A = Tensor((i, j), is_sparse=True)
    B = Tensor((j_, ), is_sparse=False)
    """Calculation declaration"""
    C = Compute(A@B)
    """Auto-Tune and excute"""
    A.LoadData("/home/qxj/AutoSparse/dataset/demo_dataset/__test_matrix.csr")

    sch = AutoTune(C, method = "random_searching", population_size=100,
                   performance_model_path = True, trial = 100)
    func = Build(sch)
    time = func.Run()
    print(time)


def test_spmm_randomsearching():
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
    A.LoadData("/home/qxj/AutoSparse/dataset/demo_dataset/__test_matrix.csr")

    sch = AutoTune(C, method = "random_searching", population_size=100,
                   performance_model_path = True, trial = 100)
    func = Build(sch)
    time = min([func.Run() for i in range(10)])
    print(time)

def test_sddmm_randomsearching():
    """Axis declarations"""
    i = Axis(64, ModeType.DENSE, "i")
    j = Axis(128, ModeType.COMPRESSED, "j")
    i_ = Axis(64, ModeType.DENSE, "i")
    k_ = Axis(128, ModeType.DENSE, "k")
    k__ = Axis(128, ModeType.DENSE, "k")
    j_ = Axis(128, ModeType.DENSE, "j")
    """Tensor declaration"""
    A = Tensor((i, j), is_sparse=True)
    B = Tensor((i_, k_), is_sparse=False)
    C = Tensor((k__, j_), is_sparse=False)
    """Calculation declaration"""
    C = Compute(A*(B@C))
    """Auto-Tune and excute"""
    A.LoadData("/home/qxj/AutoSparse/dataset/demo_dataset/__test_matrix.csr")

    sch = AutoTune(C, method = "random_searching", population_size=100,
                   performance_model_path = True, trial = 100)
    func = Build(sch)
    time = func.Run()
    print(time)

test_spmm_randomsearching()