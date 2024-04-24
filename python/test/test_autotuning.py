"""Test for SpMM"""

import os, sys
import pytest
current = os.path.join(os.getcwd(), "python")
sys.path.append(current)

from AutoSparse import *

@pytest.mark.parametrize("filename", ['__test_matrix.csr', 
                        'nemspmm1_16x4_0.csr', 'NACA0015_16x8_9.csr'])
@pytest.mark.parametrize("method", ['random_searching', 'batch_p_searching',
                        'sa_searching', 'q_searching', 'q_sa_searching'])
def test_spmv(filename: str, method: str):
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    mtx_filepath = os.path.join(
        autosparse_prefix, "dataset", "demo_dataset", filename
    )
    num_row, num_col, num_nonezero = np.fromfile(
        mtx_filepath, count=3, dtype = '<i4'
    )
    print(f"num_row={num_row}, num_col={num_col}, num_nonezero={num_nonezero}")
    """Axis declarations"""
    i = Axis(int(num_row), ModeType.DENSE, "i")
    j = Axis(int(num_col), ModeType.COMPRESSED, "j")
    j_ = Axis(int(num_col), ModeType.DENSE, "j")
    """Tensor declaration"""
    A = Tensor((i, j), is_sparse=True)
    B = Tensor((j_, ), is_sparse=False)
    """Calculation declaration"""
    C = Compute(A@B)
    """Auto-Tune and excute"""
    A.LoadData(mtx_filepath)
    print(CreateSchedule(C).GenConfigCommand()[0])

    sch = AutoTune(C, method = method, population_size=100, trial = 100)
    func = Build(sch)
    time = min([func.Run() for i in range(10)])
    print(time)

@pytest.mark.parametrize("filename", ['__test_matrix.csr', 
                        'nemspmm1_16x4_0.csr', 'NACA0015_16x8_9.csr'])
@pytest.mark.parametrize("method", ['random_searching', 'batch_p_searching',
                        'sa_searching', 'q_searching', 'q_sa_searching'])
def test_spmm(filename: str, method: str):
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    mtx_filepath = os.path.join(
        autosparse_prefix, "dataset", "demo_dataset", filename
    )
    num_row, num_col, num_nonezero = np.fromfile(
        mtx_filepath, count=3, dtype = '<i4'
    )
    print(f"num_row={num_row}, num_col={num_col}, num_nonezero={num_nonezero}")
    """Axis declarations"""
    i = Axis(int(num_row), ModeType.DENSE, "i")
    k = Axis(int(num_col), ModeType.COMPRESSED, "k")
    k_ = Axis(int(num_col), ModeType.DENSE, "k")
    j = Axis(256, ModeType.DENSE, "j")
    """Tensor declaration"""
    A = Tensor((i, k), is_sparse=True)
    B = Tensor((k_, j), is_sparse=False)
    """Calculation declaration"""
    C = Compute(A@B)
    """Auto-Tune and excute"""
    A.LoadData(mtx_filepath)
    print(CreateSchedule(C).GenConfigCommand()[0])

    sch = AutoTune(C, method = method, population_size=100, trial = 100)
    func = Build(sch)
    time = min([func.Run() for i in range(10)])
    print(time)

@pytest.mark.parametrize("filename", ['__test_matrix.csr', 
                        'nemspmm1_16x4_0.csr', 'NACA0015_16x8_9.csr'])
@pytest.mark.parametrize("method", ['random_searching', 'batch_p_searching',
                        'sa_searching', 'q_searching', 'q_sa_searching'])
def test_sddmm(filename, method: str):
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    mtx_filepath = os.path.join(
        autosparse_prefix, "dataset", "demo_dataset", filename
    )
    num_row, num_col, num_nonezero = np.fromfile(
        mtx_filepath, count=3, dtype = '<i4'
    )
    print(f"num_row={num_row}, num_col={num_col}, num_nonezero={num_nonezero}")
    """Axis declarations"""
    i = Axis(int(num_row), ModeType.DENSE, "i")
    j = Axis(int(num_col), ModeType.COMPRESSED, "j")
    i_ = Axis(int(num_row), ModeType.DENSE, "i")
    k_ = Axis(256, ModeType.DENSE, "k")
    k__ = Axis(256, ModeType.DENSE, "k")
    j_ = Axis(int(num_col), ModeType.DENSE, "j")
    """Tensor declaration"""
    A = Tensor((i, j), is_sparse=True)
    B = Tensor((i_, k_), is_sparse=False)
    C = Tensor((k__, j_), is_sparse=False)
    """Calculation declaration"""
    C = Compute(A*(B@C))
    """Auto-Tune and excute"""
    A.LoadData(mtx_filepath)
    print(CreateSchedule(C).GenConfigCommand()[0])

    sch = AutoTune(C, method = method, population_size=100, trial = 100)
    func = Build(sch)
    time = min([func.Run() for i in range(10)])
    print(time)

test_spmm('nemspmm1_16x4_0.csr', "random_searching")