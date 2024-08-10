"""utils.py"""
from typing import *
import numpy as np
import math
import os, sys
from scipy.sparse import coo_matrix
from scipy.io import mmwrite

def GetAlphabet26BaseNumber(n: int, is_upper: bool, string: str = None) -> str:
    """Translate integer n to English alphabet 26 base number

    Parameters
    ----------
    n: int
        The alphabet 26 base number string index.
    is_upper: bool
        Using upper alphabet?
    
    Return
    ------
    name: str
    """
    if string == None:
        if is_upper:
            string = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        else:
            string = "ijklmnopqrstuvwxyzabcdefgh"
    name = ""
    while(n >= 0):
        name += string[n % 26]
        n = n // 26 - 1
    return name[::-1]

def GetNearPowerOfX(x, value):
    """Get the closest x-power to the value that is less than or equal it."""
    assert isinstance(value, int) and value > 0
    return int(math.pow(x, int(math.log(value, x))))

def IsPowerX(x, val):
    assert isinstance(val, int) and val > 0
    return math.fabs(math.pow(x, int(math.log(val, x))) - val) < 1e-20


def GetFactorList(value: int):
    """Factorization value and get all the factors."""
    assert value >= 0
    ret = set()
    sqrt_value = math.sqrt(value)
    for i in range(1, math.ceil(sqrt_value) + 1):
        if value % i == 0:
            ret.add(i)
            ret.add(value // i)
    return list(ret)

def PowerXList(x: int, left: int, right:int):
    """The numbers in range between left and right and which 
        must base x.
    """
    ret = []
    num = 1
    while num < left:
        num *= x
    while num <= right:
        ret.append(num)
        num *= x
    return ret

def SplitWithFactorizationRecursiveHelp(
    value: int, cur: List, number: int, ret: List,
    policy: str = "power2"
):
    if number == 1:
        if value > 256: # Note: May bad condition.
            return
        ret.append(cur + [value])
        return 
    factor_lst = []
    if (policy == "power2"):
        factor_lst.extend(PowerXList(2, 1, value)) 
    elif (policy == "factorization"):
        factor_lst.extend(GetFactorList(value))
    else:
        factor_lst.extend(PowerXList(2, 1, value))
        factor_lst.extend(GetFactorList(value))
    
    for factor in factor_lst:
        SplitWithFactorizationRecursiveHelp(
            value // factor, cur + [factor], number-1, ret, policy
        )


def SplitWithFactorization(value: int, number: int, policy: str = "power2"):
    """Split value with factorization
    
    Parameters
    ----------
    value: int
        Value need to be splited.
    number: int
        How many have to decomposed into.
    policy: str optinal("power2")
        Factorization policy, there have "power2" (only factorize to number
        based 2), "factorization", and "mixing" (contain 2 method).
    """
    assert policy in ["power2", "factorization", "mixing"]
    ret = []
    if (policy in ["power2", "mixing"]):
        value = GetNearPowerOfX(2, value)
    SplitWithFactorizationRecursiveHelp(value, [], number, ret, policy)
    return ret

def Permute(lst):
    """Using permutations function to generates all permutations of the list."""
    from itertools import permutations
    return [list(x) for x in permutations(lst, len(lst))]

def Flatten(x):
    """将多维的list tuple展开为一维的list"""
    ret = []
    for v in x:
        if isinstance(v, (list, tuple)):
            ret.extend(list(Flatten(v)))
        else:
            ret.append(v)
    return ret


def get_coo_from_csr_file(filepath) :
    assert ".csr" in filepath
    
    csr = np.fromfile(filepath, dtype='<i4')
    num_row,num_col,nnz = csr[0],csr[1],csr[2]
    coo = np.zeros((nnz,3),dtype=int)
    coo[:,1] = csr[3+num_row+1:3+num_row+1+nnz] # col
    bins = np.array(csr[4:num_row+4]) - np.array(csr[3:num_row+3])
    coo[:,0] = np.repeat(range(num_row), bins)
    coo[:,2] = np.ones(nnz)
    return num_row, num_col, nnz, coo

def write_mtx_from_coo(num_row, num_col, coo, filepath):
    assert ".mtx" in filepath
    sparse_matrix = coo_matrix((coo[:, 2], (coo[:, 0], coo[:, 1])), shape=(num_row, num_col))
    mmwrite(filepath, sparse_matrix)



if __name__ == "__main__":
    # for i in range(100):
    #     print(GetAlphabet26BaseNumber(i, True))
    #     print(GetAlphabet26BaseNumber(i, False))
    # print(SplitWithFactorization(256, 4))
    # print(Permute([0,1,2,3,4,5,6]))
    # print(Flatten([[[1,2,3],[4,5]],[6,7,9]]))
    num_row, num_col, nnz, coo = get_coo_from_csr_file("/home/qxj/AutoSparse/dataset/demo_dataset/nemspmm1_16x4_0.csr")
    write_mtx_from_coo(num_row, num_col, coo, "/home/qxj/AutoSparse/dataset/mtx_demo_dataset/nemspmm1_16x4_0.mtx")
