""" Define sparse format. """
import copy
from typing import List, Optional, Tuple, Union
from enum import Enum

FormatMode = [
    "DENSE", "COMPRESSED", "COMPRESSED_UN", 
    "SINGLETON", "SINGLETON_UN"
]

class ModeType(Enum):
    DENSE = 0
    COMPRESSED = 1
    COMPRESSED_UN = 2
    SINGLETON = 3
    SINGLETON_UN = 4

global_name_set = set()
auto_name_set = set()


class Axis:
    """Define the axis."""
    name: str
    size: int
    mode: ModeType

    def __init__(
        self, 
        size, 
        mode = ModeType.DENSE, 
        name = None
    ) -> None:
        assert isinstance(size, int), \
                f"[AutoSparse.Axis] The data type of 'size' must be int, and not {type(size)}."
        if name is None:
            self.name = self.__CreateAxisName()
        elif isinstance(name, str):
            self.name = name
            global_name_set.add(name)
        else:
            assert False, \
                f"[AutoSparse.Axis] The data type of 'name' must be str, and not {type(name)}."
        self.size = size
        self.mode = mode
    
    @staticmethod
    def __CreateAxisName():
        string = "ijklmnopqrstuvwxyzabcdefgh"
        name = ""
        cnt = len(auto_name_set)
        flag = True
        while (flag):
            while(cnt >= 0):
                name += string[cnt % 26]
                cnt = cnt // 26 - 1
            if name in global_name_set:
                cnt += 1
            else:
                global_name_set.add(name[::-1])
                auto_name_set.add(name[::-1])
                flag = False
        return name[::-1]

    def __str__(self) -> str:
        return "AutoSparse.Axis(" + str(self.name) + ", " + \
            str(self.size) + ", " + str(self.mode) + ")"

    def __eq__(self, other: "Axis"):
        """Only need size and name are equal."""
        return self.name == other.name and self.size == other.size

class Format:
    """Define format constructed by axes."""
    shape: Tuple
    axes: Tuple["Axis"]
    order: Tuple[int] # The order of loop traversal.

    def __init__(self, axes: Tuple["Axis"], order: Tuple[int] = None) -> None:
        shape = tuple([item.size for item in axes])
        if order is None:
            self.order = tuple([i for i in range(len(axes))])
        else:
            self.order = order
        self.axes = axes
        self.shape = shape
        self.axes_name = {item.name: item for item in self.axes}
    
    def __str__(self) -> str:
        if self.axes != None or len(self.axes):
            axes_str = "("
            for axis in self.axes:
                axes_str += str(axis)[11:] + ", "
            axes_str = axes_str[:-2] + ")"
        else:
            axes_str = "()"
        return "AutoSparse.Format(axes=" + axes_str + ", order=" + str(self.order) + ")" 


class CSR(Format):
    def __init__(self, shape, name0 = None, name1 = None) -> None:
        assert len(shape) == 2, "[AutoSparse.Axis] CSR format only support 2D."
        i = Axis(shape[0], name = name0, mode = ModeType.DENSE)
        j = Axis(shape[1], name = name1, mode = ModeType.COMPRESSED)
        super().__init__((i, j))

class CSC(Format):
    def __init__(self, shape, name0 = None, name1 = None) -> None:
        assert len(shape) == 2, "[AutoSparse.Axis] CSC format only support 2D."
        i = Axis(shape[0], name = name0, mode = ModeType.DENSE)
        j = Axis(shape[1], name = name1, mode = ModeType.COMPRESSED)
        super().__init__((i, j), order = (1, 0))

class COO(Format):
    def __init__(self, shape, name0 = None, name1 = None) -> None:
        assert len(shape) == 2, "[AutoSparse.Axis] COO format only support 2D."
        i = Axis(shape[0], name = name0, mode = ModeType.COMPRESSED_UN)
        j = Axis(shape[1], name = name1, mode = ModeType.SINGLETON)
        super().__init__((i, j))

class DCSR(Format):
    def __init__(self, shape, name0 = None, name1 = None) -> None:
        assert len(shape) == 2, "[AutoSparse.Axis] DCSR format only support 2D."
        i = Axis(shape[0], name = name0, mode = ModeType.COMPRESSED)
        j = Axis(shape[1], name = name1, mode = ModeType.COMPRESSED)
        super().__init__((i, j))

class DCSC(Format):
    def __init__(self, shape, name0 = None, name1 = None) -> None:
        assert len(shape) == 2, "[AutoSparse.Axis] DCSC format only support 2D."
        i = Axis(shape[0], name = name0, mode = ModeType.COMPRESSED)
        j = Axis(shape[1], name = name1, mode = ModeType.COMPRESSED)
        super().__init__((i, j), order = (1, 0))



################################################################################
################################################################################
        
def test_CreateAxisName():
    string = "ijklmnopqrstuvwxyzabcdefgh"
    for cnt in range(100):
        name = ""
        while(cnt >= 0):
            name += string[cnt % 26]
            cnt = cnt // 26 - 1
        print(name[::-1])