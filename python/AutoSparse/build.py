from typing import *

from .schedule import Schedule, CreateSchedule
from .tensor import Tensor, ComputeTensor

def Build(sch: Union[Schedule, ComputeTensor]):
    """ Build and compile computation function. """
    if isinstance(sch, ComputeTensor):
        sch = CreateSchedule(sch)
    
    return sch.GenConfigCommand()

    # There use a command string transfer computation, format and schedule 
    # information to C++ backend with taco compiler.
