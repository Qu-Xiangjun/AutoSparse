"""utils.py"""
from typing import *

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


if __name__ == "__main__":
    for i in range(100):
        print(GetAlphabet26BaseNumber(i, True))
        print(GetAlphabet26BaseNumber(i, False))