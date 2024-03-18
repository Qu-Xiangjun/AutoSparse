from typing import *

from .tensor import Value, Tensor, ComputeTensor, FindTopoSort
from .format import Axis

class Schedule:
    op: ComputeTensor
    all_inputs_op: List["Value"]
    all_axes: Dict[str, "Axis"]
    config: Dict[str, Any]

    def __init__(self, op: ComputeTensor) -> None:
        self.op = op
        self.schedules
        ops = FindTopoSort(self.op)
        for op in ops:
            if isinstance(op, Value):
                self.all_inputs_op.append(op)
            for axis in op.format.axes:
                

    def FormatSplit(self):
        pass
    
    def FormatReorder(self):
        raise NotImplementedError()
    
    def FormatMode(self):
        raise NotImplementedError()
    
    def LoopSplit(self):
        raise NotImplementedError()
    
    def LoopReorder(self):
        raise NotImplementedError()
    
    def LoopUnroll(self):
        raise NotImplementedError()
    
    def LoopVectorize(self):
        raise NotImplementedError()
    
    def LoopParallel(self):
        raise NotImplementedError()
    
    def GetScheduleCommand():
        raise NotImplementedError()
    
    def Clear(self):
        self.config = dict()

    def GenConfigCommand(self):
        raise NotImplementedError()

def CreateSchedule(C: ComputeTensor):
    pass