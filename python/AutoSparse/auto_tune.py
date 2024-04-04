from typing import Union
import signal
import time
import torch
import datetime
import threading
from torch.utils.tensorboard import SummaryWriter

from .tensor import ComputeTensor
from .schedule import Schedule, CreateSchedule
from .space import *
from .model import *
from .build import Build

def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

def call_with_timeout(func, timeout, *args, **kwargs):
    # Set timer
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        # invoking
        result = func(*args, **kwargs)
    finally:
        # Close timer
        signal.alarm(0)
    return result

def _Excute(func, timeout, *args, **kwargs):
    try:
        result = call_with_timeout(func, timeout, *args, **kwargs)
    except TimeoutError as e:
        print("TimeoutError -------------------------")
        result = float("inf")
    return result

def Evaluate(
    schedule: Schedule,
    func: Build,
    eval_warm_times: int = 10,
    eval_round: int = 100,
    eval_timeout: float = None,
    eval_policy: str = "avg",
):
    """Evaluate a config performance.
    
    Parameters
    ----------
    schedule: Schedule
    config: Dict[subspace name, subspace entry]
    """
    res = _Excute(
        func.Run, timeout=eval_timeout,
        sch=schedule,
        warm=eval_warm_times,
        round=eval_round,
        time_policy=eval_policy
    )
    if res < 0:
        res = float("inf")
    return res


def CheckModeHelp(freordered_vars : list, vars_mode : dict, dimensions : dict):
    # c can't put in first axis, only if first axis length is 1 and second axis
    # is dense, and also the second axis can be a sequence of end by length 1 sparse axis. 
    # Moreover, the axes sequence between c and dense also can have q or c or u mode of length 1.
    cds_flag = False
    first_item = freordered_vars[0]
    if vars_mode[first_item] == 4:
        if dimensions[first_item] != 1:
            return False
        dsu1_flag = False
        for idx, item in enumerate(freordered_vars[1:]):
            if vars_mode[item] == 0: # find dense axis
                dsu1_flag = True
                break
            elif vars_mode[item] == 1 and dimensions[item] == 1:
                dsu1_flag = True
                break
            elif vars_mode[item] == 1:
                return False
            elif vars_mode[item] > 1 and dimensions[item] == 1:
                continue
            elif vars_mode[item] > 1:
                return False
        if dsu1_flag == False: # No d or length 1 of s or u endding.
            return False
        
    # q must be last one, only if all the axis followed by q is 1 of dimension.
    q_flag = False 
    for idx, item in enumerate(freordered_vars):
        if q_flag and dimensions[item] != 1:
            return False
        if vars_mode[item] == 3 and dimensions[item] == 1:
            q_flag = True
    # If dense follow the continuous q or c axes, all the axes length must be 1. 
    dc_flag = False
    for idx, item in enumerate(freordered_vars):
        if idx == 0:
            continue
        last_item = freordered_vars[idx - 1]
        if vars_mode[last_item] == 0 and vars_mode[item] > 2:
            dc_flag = True
        if dc_flag:
            if vars_mode[item] < 3:
                dc_flag = False
            elif dimensions[item] != 1:
                return False
            else:
                continue
    # The previous axis of c can't be dense axis, unless c axis' length is 1.
    # If there are some axes separated in between c and dense axis, all the axes 
    # can't be c axes of length 1.
    c_flag = False
    for idx, item in enumerate(freordered_vars[::-1]):
        if c_flag:
            if ((vars_mode[item] == 4 or vars_mode[item] == 3 ) \
                and dimensions[item] == 1):
                continue
            elif vars_mode[item] == 0:
                return False
        if vars_mode[item] == 4 and dimensions[item] != 1:
            c_flag = True
    
    # Last axis of 4 will cause error if its length is't 1.
    last_item = freordered_vars[-1]
    if vars_mode[last_item] == 4 and dimensions[last_item] != 1:
        return False
    # If previous axis of `c` is sparse or singleton, all the length of `c` and
    # axis followed by `c` must be 1.
    sqc_flag = False
    for idx, item in enumerate(freordered_vars):
        if idx == 0:
            continue
        if sqc_flag:
            if dimensions[item] == 1:
                continue
            else:
                return False
        last_item = freordered_vars[idx - 1]
        if (vars_mode[item] == 4 and dimensions[item] == 1 and \
            (vars_mode[last_item] == 1 or vars_mode[last_item] == 3)):
            sqc_flag = True
    # Dense can't follow u c q, only if all the axis followed u length is 1,
    # or if all the axis behind dense length is 1.
    u_flag = False
    ud_flag = True
    for idx, item in enumerate(freordered_vars):
        if vars_mode[item] == 2: # u
            u_flag = True
        if u_flag:
            if vars_mode[item] == 0:
                # All the item follow ud must be length 1.
                for item2 in freordered_vars[idx:]:
                    if dimensions[item2] != 1:
                        ud_flag = False
                        break
            elif vars_mode[item] == 1:
                u_flag = False
            else:
                continue
            if ud_flag == False: 
                # All the behind d axes length must be 1.
                for item2 in freordered_vars[:idx+1]:
                    if dimensions[item2] != 1:
                        return False
    # There can't be d*d condition, and the * indicate continous q or c, which
    # c at least occur 1 times. Moreover, the sequence can't be head of mode 
    # sequence, and can't follow axes sequence which all length is 1.
    dcd_d_flag = False
    dcd_cd_flag = False
    for idx, item in enumerate(freordered_vars[::-1]):
        if dcd_cd_flag: # find first d
            if vars_mode[item] == 0:
                # all the item behind must be length 1.
                if idx < len(freordered_vars) - 1:
                    for item2 in freordered_vars[::-1][idx+1:]:
                        if dimensions[item2] > 1:
                            return False
            elif vars_mode[item] > 2:
                continue
            else:
                dcd_cd_flag = False
                dcd_d_flag = False
                continue
        if dcd_d_flag: # find c
            if vars_mode[item] == 3:
                continue
            elif vars_mode[item] == 4:
                dcd_cd_flag = True
            else:
                dcd_d_flag = False
        if vars_mode[item] == 0: # find last d
            dcd_d_flag = True

    # Return
    return True

def CheckMode(freordered_vars, tensor_all_axes_size, vars_mode):
    for idx, item in enumerate(freordered_vars):
        if tensor_all_axes_size[item] == 1:
            continue # Size 1 can select all the mode
        elif idx == 0:
            # The condition make level mode can't be singleton.
            # Only if the dimensions is 1, the mode can be singleton.
            if vars_mode[item] > 2:
                return False
        elif vars_mode[freordered_vars[idx - 1]] == 0: # dense
            # (un_)singleton follow dense level will make computation error, 
            # only if dimensions is 1.
            if vars_mode[item] > 2:
                return False
        elif vars_mode[freordered_vars[idx - 1]] == 1:
            if vars_mode[item] > 2:
                return False
        elif vars_mode[freordered_vars[idx - 1]] == 2: # un_sparse
            # only last axis can contain singleton followed un_sparse.
            if idx == len(freordered_vars) - 1: 
                # Notice last one can't be 4
                if vars_mode[item] not in [1, 2, 3]:
                    return False
            elif vars_mode[item] not in [1, 2, 4]:
                return False
        elif vars_mode[freordered_vars[idx - 1]] == 3: # singleton
            if vars_mode[item] not in [1, 2]:
                return False
        elif vars_mode[freordered_vars[idx - 1]] == 4: # un_singleton
            # only last axis can contain singleton followed un_sparse.
            if idx == len(freordered_vars) - 1: 
                # Notice last one cant be 4
                if vars_mode[item] not in [1, 2, 3]:
                    return False
            elif vars_mode[item] not in [1, 2, 4]:
                return False
    
    return CheckModeHelp(freordered_vars, vars_mode, tensor_all_axes_size)


def AddSimpleSchedule(compute_tensor: ComputeTensor, config: Dict):
    sch = CreateSchedule(compute_tensor)

    origin_axes_dict = sch.origin_axes
    sparse_axes_names = set()
    for tensor in sch.all_tensors_bk:
        if tensor.is_sparse == False:
            continue
        for axis in tensor.format.axes:
            sparse_axes_names.add(axis.name)

    # FSplit
    fsplited_axes_names = {} # {Splited_axis_name: (new_axes, axes_size)}
    for axis_name in list(sparse_axes_names):
        split_subspace_name = "splite_{}".format(axis_name)
        if config.get(split_subspace_name, None) == None: # It did not FSplit.
            fsplited_axes_names[axis_name] = \
                (axis_name, origin_axes_dict[axis_name][0].size)
        else:
            split_subspace_entry = config[split_subspace_name]
            new_axes = sch.FormatSplit(axis_name, split_subspace_entry)
            fsplited_axes_names[axis_name] = (new_axes, split_subspace_entry)
    
    # LSplit
    only_dense_axes_names = set(origin_axes_dict.keys()) - sparse_axes_names
    lsplited_axes_names = {} # {Splited_axis_name: (new_axes, axes_size)}
    for axis_name in list(only_dense_axes_names):
        split_subspace_name = "splite_{}".format(axis_name)
        if config.get(split_subspace_name, None) == None:
            lsplited_axes_names[axis_name] = \
                (axis_name, origin_axes_dict[axis_name][0].size)
        else:
            split_subspace_entry = config[split_subspace_name]
            new_axes = sch.LoopSplit(axis_name, split_subspace_entry)
            lsplited_axes_names[axis_name] = (new_axes, split_subspace_entry)
    
    sorted_splited_axes_names = []
    all_axes_size = {} # Static axes size.
    for value in fsplited_axes_names.values():
        sorted_splited_axes_names.extend(value[0])
        for axis_name, axis_size in zip(value[0], value[1]):
            all_axes_size[axis_name] = axis_size
    for value in lsplited_axes_names.values():
        sorted_splited_axes_names.extend(value[0])
        for axis_name, axis_size in zip(value[0], value[1]):
            all_axes_size[axis_name] = axis_size

    # Reorder
    sorted_splited_axes_names = sorted(sorted_splited_axes_names)

    reordered_vars_indices = config.get("reorder", None)
    if reordered_vars_indices:
        assert len(sorted_splited_axes_names) == len(reordered_vars_indices), \
            "[AutoSparse.AddSimpleSchedule] Reorder dim error in config."
        reordered_vars = [sorted_splited_axes_names[i] for i in reordered_vars_indices]
        sch.LoopReorder(reordered_vars)
    else:
        reordered_vars = sorted_splited_axes_names
    

    # FormatReorder && Mode setting and checking
    axes_mode = {} # {axis_name: mode}
    for idx, tensor in enumerate(sch.all_tensors_bk):
        if tensor.is_sparse == False:
            continue
        tensor_all_axes_name = tensor.format.axes_name.keys()
        
        new_ordered_axes = []
        for axis_name in reordered_vars:
            if axis_name in tensor_all_axes_name:
                new_ordered_axes.append(axis_name)

        format_mode_subspace_name = "format_mode_{}".format(sch.tensor_name_lst[idx])
        format_modes = config.get(format_mode_subspace_name, None)
        if format_modes != None:
            assert len(tensor.shape) == len(format_modes), \
                    "[AutoSparse.AddSimpleSchedule] Format mode count differ from tensor."
            format_modes_dict = {}
            for i in range(len(new_ordered_axes)):
                format_modes_dict[new_ordered_axes[i]] = format_modes[i]
            if CheckMode(new_ordered_axes, all_axes_size, format_modes_dict) == False:
                raise ValueError()
        
        if config.get("reorder", None) != None:
            new_tensor = sch.FormatReorder(tensor, new_ordered_axes)
        else:
            new_tensor = tensor
        if format_modes != None:
            for i, axis_name in enumerate(new_ordered_axes):
                sch.FormatMode(new_tensor, axis_name, format_modes[i])
                # If there are multiple tensors with the same axis, take the largest mode. 
                axes_mode[axis_name] = max(axes_mode.get(axis_name, 0), format_modes[i])
    
    # Parallel
    is_parallel = config.get("parallel", [])
    parallel_var = None
    vectorize_var = None
    unroll_var = None
    if len(is_parallel) and is_parallel[0] == 1: # parallel
        for axis_name in reordered_vars:
            is_reduce = False
            for reduce_axis_name in sch.reduce_axes.keys():
                if reduce_axis_name in axis_name:
                    is_reduce = True
            if axes_mode.get(axis_name, 0) <= 2 and is_reduce == False:
                parallel_var = axis_name
                break
        if parallel_var != None:
            sch.LoopParallel(parallel_var)
        
    if len(is_parallel) >= 2 and is_parallel[1] == 1: # vectorize
        candidate_var = reordered_vars[-1]
        is_reduce = False
        for reduce_axis_name in sch.reduce_axes.keys():
            if reduce_axis_name in candidate_var:
                is_reduce = True
        if (axes_mode.get(candidate_var, 0) == 0 and is_reduce == False \
            and candidate_var != parallel_var):
            vectorize_var = candidate_var
        if vectorize_var != None:
            sch.LoopVectorize(vectorize_var)
    
    if len(is_parallel) >= 3 and is_parallel[2] == 1: # unroll
        unroll_candidate = []
        for axis_name in reordered_vars:
            if (axis_name != parallel_var and axis_name != vectorize_var \
                and axes_mode.get(axis_name, 0) == 0):
                unroll_candidate.append(axis_name)
        if len(unroll_candidate):
            unroll_var = unroll_candidate[0]
            for axis_name in unroll_candidate: # Pick the max size one to unroll
                if all_axes_size[axis_name] > all_axes_size[unroll_var]:
                    unroll_var = axis_name
            sch.LoopUnroll(unroll_var, all_axes_size[unroll_var])

    # Omp arguments
    import multiprocessing
    sch.SetThreadNum(multiprocessing.cpu_count())
    if config.get("omp", None) != None:
        sch.SetParallelChunk(config["omp"])
    else:
        sch.SetParallelChunk(32) # Default

    return sch


def Warm(
    schedule: Schedule,
    func: Build,
    agent_group: DQNAgentGroup,
    use_performance_model: bool = False,
    warm_trial: int = 10,
    population_size: int = 10,
    repeat_count:int = 5,
    eval_warm_times: int = 10,
    eval_round: int = 100,
    eval_timeout: float = None,
    eval_policy: str = "avg",
):
    """Warm find have there good program schedules.
    
    Parameters
    ----------
    schedule: Schedule
        It is pure object without any schedules.
    agent_group: DQNAgentGroup
    use_performance_model: bool

    """

    warm_ok = False
    best_value = float("inf")
    while not warm_ok:
        for tri in range(warm_trial):
            # Get batch random config 
            configs_dict = agent_group.RandomBatch(population_size)
            configs_entries = [{} for i in range(population_size)]
            configs_indicse = [{} for i in range(population_size)]
            for subspace_name, val in configs_dict.items(): 
                # every sub space item
                entries, indices = val[0], val[1]
                for i in range(population_size):
                    configs_entries[i][subspace_name] = entries[i]
                    configs_indicse[i][subspace_name] = indices[i]
            # Get config performance
            configs_performances = []
            for i in range(population_size):
                try:
                    schedule = AddSimpleSchedule(schedule.compute_tensor, configs_entries[i])
                except ValueError:
                    configs_performances.append(float('inf'))
                    continue
                if use_performance_model:
                    configs_performances.append(float('inf')) # Future todo.
                else:
                    configs_performances.append(
                        Evaluate(schedule, func, eval_warm_times,
                        eval_round, eval_timeout, eval_policy)
                    )
                if configs_performances[-1] < float("inf"):
                    agent_group.Record(configs_indicse[i], 
                                       configs_performances[-1], use_sa=False)
            
            best_per = min(configs_performances)
            print (f"[Autosparse.AutoTune] Warm in {tri} trial best: {best_per:.8f}, "
                   f"history best: {agent_group.Top1()[1]}")
            best_value = max(best_value, best_per)
            
        if not (agent_group.Top1()[1] < float("inf")):
            warm_trial += 1
            repeat_count -= 1
            # Can't find a solution
            print ("[AutoSparse.AutoTune] Warming: No valid schedule in warm up.")
            if repeat_count < 0:
                warm_ok = True
        else:
            warm_ok = True
    return best_value


def RandomSearching(
    schedule: Schedule,
    func: Build,
    agent_group: DQNAgentGroup,
    population_size: int = 100,
    use_performance_model: bool = False,
    trial: int = 100,
    eval_warm_times: int = 10,
    eval_round: int = 100,
    eval_timeout: float = None,
    eval_policy: str = "avg",
    **kwargs
):
    """Ramdom search.

    Returns
    -------
    indices: Dict {subspace_name: subspace_entry_index}
    value: float
        performance.
    
    """
    print("**************** Start Random Search ****************")
    print(f"population_size         ={population_size}")
    print(f"trial                   ={trial}")
    print(f"use_performance_model   ={use_performance_model}")
    print(f"eval_warm_times         ={eval_warm_times}")
    print(f"eval_round              ={eval_round}")
    print(f"eval_timeout            ={eval_timeout}")
    print(f"eval_policy             ={eval_policy}")
    print()

    writer = SummaryWriter('runs/random_search_complex_continue')
    for tri in range(trial):
        local_best = Warm(
            schedule=schedule,
            func=func,
            agent_group = agent_group,
            use_performance_model=use_performance_model,
            warm_trial=1,
            population_size=population_size,
            repeat_count=0,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy,
        )
        writer.add_scalar('Local Best Score', local_best, tri)
        writer.add_scalar('Global Best Score', agent_group.Top1()[1], tri)
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Random Search: Current time: {current_time} in trial {tri}", flush=True)

    return agent_group.Top1()


def QSearching(
    schedule: Schedule,
    func: Build,
    agent_group: DQNAgentGroup,
    population_size: int = 10,
    use_performance_model: bool = False,
    trial: int = 50,
    eval_warm_times: int = 10,
    eval_round: int = 100,
    eval_timeout: float = None,
    eval_policy: str = "avg",
    **kwargs
):
    """Only using DQN method"""
    pass

def SASearching(
    schedule: Schedule,
    func: Build,
    agent_group: DQNAgentGroup,
    population_size: int = 10,
    use_performance_model: bool = False,
    trial: int = 50,
    eval_warm_times: int = 10,
    eval_round: int = 100,
    eval_timeout: float = None,
    eval_policy: str = "avg",
    **kwargs
):
    """Only using SA method"""
    pass

def QSASearching(
    schedule: Schedule,
    func: Build,
    agent_group: DQNAgentGroup,
    population_size: int = 10,
    use_performance_model: bool = False,
    trial: int = 50,
    eval_warm_times: int = 10,
    eval_round: int = 100,
    eval_timeout: float = None,
    eval_policy: str = "avg",
    **kwargs
):
    """Using DQN and SA mixing method"""
    pass

def AutoTune(
    input: ComputeTensor,
    method: str = "random_searching",
    population_size: int = 100,
    agent_model_paths: Dict[str, str] = {},
    agent_data_paths: Dict[str, str] = {},
    use_performance_model: bool = False,
    save_performance_model: bool = False,
    performance_model_path: str = None,
    save_performance_data: bool = False,
    save_performance_data_filepath: str = None,
    trial: int = 100,
    eval_warm_times: int = 10,
    eval_round: int = 50,
    eval_timeout: float = None,
    eval_policy: str = "avg",
    save_searching: bool = True, 
    **kwargs
):
    """
    Parameters
    ----------
    input: ComputeTensor
    method: str optinal("random_searching)
        method include:
        "random_searching", "q_searching", "sa_searching", "q_sa_searching"
    population_size: int optinal(50)
        The size of the population during the search.
    agent_model_paths: Dict[str, str]
        Every agent's model path. The model have train in last searching.
    agent_data_paths: Dict[str, str]
        Every agent's data which is apper in last searching.
    use_performance_model: boll optinal(False)
        Wheather using cost model to predict program performance.
    performance_model_path: str
        Load the off line performance model path.
    save_performance_model: bool
    save_performance_data: bool
    save_performance_data_filepath: str
    trial:
        search trails.
    eval_round:
        Warm times when run a program to test excution time.
    test_round:
        Evaluating excution time will run round times program.
    eval_timeout: float
        Evaluation timeout for every program.
    eval_policy: str optinal("avg")
        "avg" mean return average test time set.
        "mid" mean return middile number of test time set.
        "best mean return best one of test time set.
    save_searching: bool optinal(True) 
        Is save the tune result, which include searching state and 
        schedule result.

    Return
    ------
    best_sch: Schedule
    """
    sch = None
    if isinstance(input, ComputeTensor):
        sch = CreateSchedule(input)
    elif isinstance(input, Schedule):
        sch = input
    else:
        raise ValueError("Input argument type error.")
    
    # Step1: Init schedule design space with all the subspace
    tune_space = Space()

    origin_axes = sch.origin_axes
    for name, axis in origin_axes.items():
        split_subspace = SplitSubSpace(axis[0].size, 2, policy="power2")
        tune_space.add_subspace(
            "splite_{}".format(name), split_subspace, "split"
        )
    
    reorder_subspace = ReorderSubSpace(len(origin_axes) * 2)
    tune_space.add_subspace(
        "reorder", reorder_subspace, "reorder"
    )

    for i, tensor in enumerate(sch.all_tensors_bk):
        if tensor.is_sparse:
            format_mode_subspace = FModeSubSpace(len(tensor.shape) * 2)
            tune_space.add_subspace(
                "format_mode_{}".format(sch.tensor_name_lst[i]),
                format_mode_subspace, "format_mode"
            )

    # Include parallel vecorize unroll
    parallel_subspace = ParallelSubspace(3) 
    tune_space.add_subspace(
        "parallel", parallel_subspace, "parallel"
    )

    omp_subspace = OpenMPSubspace()
    tune_space.add_subspace("omp", omp_subspace, "omp")

    print(f"[AutoSparse.AutoTuing] Space size = {len(tune_space)}")

    # Step2: Create Agent Group.
    agent_group = DQNAgentGroup(
        sch.GetScheduleName(), tune_space, decay = 0.9,
        lr = 0.02, epochs=20, train_batch_size=1000
    )

    # if use_performance_model:
        # Load performance model
    
    for name, filepath in agent_model_paths.items():
        agent_group.agent_group[name].model_path = filepath
        agent_group.agent_group[name].LoadModel()
    for name, filepath in agent_data_paths.items():
        agent_group.agent_group[name].data_path = filepath
        agent_group.agent_group[name].LoadData()
    
    # Step3: Tuning with searching
    func = Build(sch)
    print(f"[AutoSparse.AutoTuing] Origin format run time = {func.origin_time:.8f}", flush = True)
    if eval_timeout == None:
        eval_timeout = math.ceil(
            func.origin_time * 2 * (eval_warm_times + eval_round) / 1000)

    if method == "random_searching":
        indices, value = RandomSearching(
            schedule = sch,
            func=func,
            agent_group=agent_group,
            population_size=population_size,
            use_performance_model = use_performance_model,
            trial = trial,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy
        )
    elif method == "sa_searching":
        indices, value = SASearching(
            schedule = sch,
            func=func,
            agent_group=agent_group,
            population_size=population_size,
            use_performance_model = use_performance_model,
            trial = trial,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy
        )
    elif method == "q_searching":
        indices, value = QSearching(
            schedule = sch,
            func=func,
            agent_group=agent_group,
            population_size=population_size,
            use_performance_model = use_performance_model,
            trial = trial,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy
        )
    elif method == "q_sa_searching":
        indices, value = QSASearching(
            schedule = sch,
            func=func,
            agent_group=agent_group,
            population_size=population_size,
            use_performance_model = use_performance_model,
            trial = trial,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy
        )
    else:
        indices, value = {}, None
        assert False, \
            "[AutoSparse.AutoTune] Error search method。"

    if save_performance_data:
        agent_group.SavePerformanceData(save_performance_data_filepath)

    if save_searching:
        for name, filepath in agent_model_paths.items():
            agent_group.agent_group[name].SaveModel()
        for name, filepath in agent_data_paths.items():
            agent_group.agent_group[name].SaveData()
    config = agent_group.GetConfigFfromIndices(indices)
    return AddSimpleSchedule(sch.compute_tensor, config)