from typing import *

from .schedule import Schedule, CreateSchedule
from .tensor import Tensor, ComputeTensor
from .backend import auto_sparse_backend

class Build(object):
    def __init__(self, sch: Union[Schedule, ComputeTensor]):
        """ Build and compile computation function. """
        if isinstance(sch, ComputeTensor):
            sch = CreateSchedule(sch)
        
        self.pure_comp_desc, self.schedules = sch.GenConfigCommand()

        self.filepaths = []
        origin_input_tensors = sch.origin_input_tensors
        for tensor in origin_input_tensors:
            if tensor.is_sparse:
                assert tensor.data, \
                    f"[AutoSparse.Build] Tensor {tensor} have not load data."
                self.filepaths.append(tensor.data)
        
        self.device = auto_sparse_backend.BackEndAPI(
            self.pure_comp_desc, self.filepaths
        )

    @property
    def origin_time(self):
        return self.device.origin_time
    
    def Run(self, sch: Union[Schedule, ComputeTensor] = None, 
            warm = 10, round = 50, time_policy = "avg"):
        """Excute new schedule
        
        Parameters
        ----------
        sch: Union[Schedule, ComputeTensor] optinal(None)
            if Schedule is None will run without schedule computation.
        warm: int optional(10)
            Warm run times.
        round: int optinal(50)
            Test times.
        time_policy: str optinal("avg")
            "avg" mean return average test time set.
            "mid" mean return middile number of test time set.
            "best mean return best one of test time set.
        Return
        ------
        excution_time: float
            -1 mean error schedule
            correct computation and schedule will return excution time 
            with millisecond unit.
        """
        if sch == None:
            return self.device.Compute(
                self.schedules, warm = warm, round = round, time_policy = time_policy
            )

        if isinstance(sch, ComputeTensor):
            sch = CreateSchedule(sch)
        pure_comp_desc, schedules = sch.GenConfigCommand()
        assert pure_comp_desc == self.pure_comp_desc, \
            "[AutoSparse.Build] New Schedule is different builded computation."
        
        test_time = self.device.Compute(
            schedules, warm = warm, round = round, time_policy = time_policy
        )

        return test_time
