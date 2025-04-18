from typing import *

from .schedule import Schedule, CreateSchedule
from .tensor import Tensor, ComputeTensor
from .backend import auto_sparse_backend

class Build(object):
    def __init__(self, sch: Union[Schedule, ComputeTensor]):
        """ Build and compile computation function. """
        if isinstance(sch, ComputeTensor):
            sch = CreateSchedule(sch)

        self.filepaths = []
        for tensor in sch.all_tensors_bk:
            if tensor.is_sparse:
                assert tensor.data, \
                    f"[AutoSparse.Build] Tensor {tensor} have not load data."
                self.filepaths.append(tensor.data)
        
        self.pure_comp_desc, self.schedules = sch.GenConfigCommand()

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
            "avg" mean to return average test time set.
            "mid" mean to return middile number of test time set.
            "best mean to return best one of test time set.
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
        # print([pure_comp_desc, schedules])
        assert pure_comp_desc == self.pure_comp_desc, \
            "[AutoSparse.Build] New Schedule is different builded computation."
        
        test_time = self.device.Compute(
            schedules, warm = warm, round = round, time_policy = time_policy
        )

        return test_time
    
    def RunWithScheduleCommand(self, pure_comp_desc: str, schedules: str,
                                warm = 10, round = 50, time_policy = "avg"):
        """Excute new schedule
        
        Parameters
        ----------
        pure_comp_desc: 
            Describe origin computation.
        schedules: 
            Describe the shcedules for computation.
        warm: int optional(10)
            Warm run times.
        round: int optinal(50)
            Test times.
        time_policy: str optinal("avg")
            "avg" mean to return average test time set.
            "mid" mean to return middile number of test time set.
            "best mean to return best one of test time set.

        Return
        ------
        excution_time: float
            -1 mean error schedule
            correct computation and schedule will return excution time 
            with millisecond unit.
        """
        assert pure_comp_desc == self.pure_comp_desc, \
            "[AutoSparce.Build] New Schedule is different builded computation."
        
        test_time = self.device.Compute(
            schedules, warm = warm, round = round, time_policy = time_policy
        )

        return test_time
