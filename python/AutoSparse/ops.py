""" Tensor operator """

from typing import List, Optional, Tuple, Union, Any

from .tensor import Operator, Value
from .format import *

class EWiseMul(Operator):
    "Element-wise Multiplication"
    def Compute(self, a: Value, b: Value):
        assert a.shape == b.shape, \
            """[AutoSparse.Operator] The shape of two operands don't match."""
        assert a.format.axes_name.keys() == b.format.axes_name.keys(), \
            """[AutoSparse.Operator] The axis name of two operands don't match."""
        axes = []
        for i in range(len(a.shape)):
            axes.append(Axis(a.format.axes[i].size, name = a.format.axes[i].name))
        return Format(tuple(axes)), (a, b), a.dtype

class ScalarMul(Operator):
    """Multiplicate with a scalar"""
    def __init__(self, scalar):
        self.scalar = scalar
    def Compute(self, a: Value):
        axes = []
        for i in range(len(a.shape)):
            axes.append(Axis(a.format.axes[i].size, name = a.format.axes[i].name))
        return Format(tuple(axes)), (a), a.dtype

class EWiseAdd(Operator):
    def Compute(self, a: Value, b: Value):
        assert a.shape == b.shape, \
            """[AutoSparse.Operator] The shape of two operands don't match."""
        assert a.format.axes_name.keys() == b.format.axes_name.keys(), \
            """[AutoSparse.Operator] The axis name of two operands don't match."""
        axes = []
        for i in range(len(a.shape)):
            axes.append(Axis(a.format.axes[i].size, name = a.format.axes[i].name))
        return Format(tuple(axes)), (a, b), a.dtype

class ScalarAdd(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def Compute(self, a: Value):
        axes = []
        for i in range(len(a.shape)):
            axes.append(Axis(a.format.axes[i].size, name = a.format.axes[i].name))
        return Format(tuple(axes)), (a), a.dtype

class EWiseSub(Operator):
    def Compute(self, a: Value, b: Value):
        assert a.shape == b.shape, \
            """[AutoSparse.Operator] The shape of two operands don't match."""
        assert a.format.axes_name.keys() == b.format.axes_name.keys(), \
            """[AutoSparse.Operator] The axis name of two operands don't match."""
        axes = []
        for i in range(len(a.shape)):
            axes.append(Axis(a.format.axes[i].size, name = a.format.axes[i].name))
        return Format(tuple(axes)), (a, b), a.dtype

class ScalarSub(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def Compute(self, a: Value):
        axes = []
        for i in range(len(a.shape)):
            axes.append(Axis(a.format.axes[i].size, name = a.format.axes[i].name))
        return Format(tuple(axes)), (a), a.dtype

class Negate(Operator):
    def Compute(self, a: Value):
        axes = []
        for i in range(len(a.shape)):
            axes.append(Axis(a.format.axes[i].size, name = a.format.axes[i].name))
        return Format(tuple(axes)), (a), a.dtype

class EWiseDiv(Operator):
    def Compute(self, a: Value, b: Value):
        assert a.shape == b.shape, \
            """[AutoSparse.Operator] The shape of two operands don't match."""
        assert a.format.axes_name.keys() == b.format.axes_name.keys(), \
            """[AutoSparse.Operator] The axis name of two operands don't match."""
        axes = []
        for i in range(len(a.shape)):
            axes.append(Axis(a.format.axes[i].size, name = a.format.axes[i].name))
        return Format(tuple(axes)), (a, b), a.dtype

class ScalarDiv(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def Compute(self, a: Value):
        axes = []
        for i in range(len(a.shape)):
            axes.append(Axis(a.format.axes[i].size, name = a.format.axes[i].name))
        return Format(tuple(axes)), (a), a.dtype

class Summation(Operator):
    def __init__(self, axes) -> None:
        if isinstance(axes, int):
            axes = tuple(axes)
        self.axes = axes
    def Compute(self, a: Value):
        axes = []
        for i in range(len(a.shape)):
            if i in self.axes:
                continue
            axes.append(Axis(a.format.axes[i].size, name = a.format.axes[i].name))
        if len(axes) == 0:
            axes.append(Axis(1))
        return Format(tuple(axes)), (a), a.dtype

class Matmul(Operator):
    def Compute(self, a: Value, b: Value):
        axes = []
        flag = True
        if len(a.shape) == 1:
            axis_a = a.format.axes[0]
            axis_b = b.format.axes_name.get(axis_a.name, None)
            flag = axis_b and axis_a == axis_b
            for item in b.format.axes:
                if item == axis_a:
                    continue
                axes.append(Axis(item.size, name = item.name))
        elif len(b.shape) == 1:
            axis_b = b.format.axes[0]
            axis_a = a.format.axes_name.get(axis_b.name, None)
            flag = axis_a and axis_a == axis_b
            for item in a.format.axes:
                if item == axis_b:
                    continue
                axes.append(Axis(item.size, name = item.name))
        elif len(a.shape) == len(b.shape):
            if a.format.axes[0:-2] == b.format.axes[0:-2] and a.format.axes[-1] == b.format.axes[-2]:
                for item in a.format.axes[0:-2]:
                    axes.append(Axis(item.size, name = item.name))
                axes.append(Axis(a.format.axes[-2].size, name = a.format.axes[-2].name))
                axes.append(Axis(b.format.axes[-1].size, name = b.format.axes[-1].name))
            else:
                flag = False
        else:
            flag = False
        assert flag, \
            "[AutoSparse.Operator] The axes don't match of two tensor."
        if len(axes) == 0:
                axes.append(Axis(1))
        return Format(tuple(axes)), (a, b), a.dtype

