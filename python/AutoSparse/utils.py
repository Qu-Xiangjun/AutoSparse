"""utils.py"""
from typing import *
import numpy as np
import math

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


if __name__ == "__main__":
    for i in range(100):
        print(GetAlphabet26BaseNumber(i, True))
        print(GetAlphabet26BaseNumber(i, False))
    print(SplitWithFactorization(256, 4))
    print(Permute([0,1,2,3,4,5,6]))
    print(Flatten([[[1,2,3],[4,5]],[6,7,9]]))