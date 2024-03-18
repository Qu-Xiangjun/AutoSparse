from typing import Union

from .tensor import ComputeTensor
from .schedule import Schedule


def AutoTune(
    input: Union[ComputeTensor, Schedule],
    method: str = "random_searching",
    use_cost_model: bool = False
):
    pass