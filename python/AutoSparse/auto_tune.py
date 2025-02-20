from typing import Union
import signal
import time
import torch
import csv
import datetime
import random
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
        # print("Error ans")
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
            "[AutoTune.AddSimpleSchedule] Reorder dim error in config."
        reordered_vars = [sorted_splited_axes_names[i] for i in reordered_vars_indices]
        sch.LoopReorder(reordered_vars)
    else:
        reordered_vars = sorted_splited_axes_names
    

    # FormatReorder && Mode setting and checking
    axes_mode = {} # {axis_name: mode}
    for idx, tensor in enumerate(sch.all_tensors_bk[:-1]):
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
                    "[AutoTune.AddSimpleSchedule] Format mode count differ from tensor."
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
    
    
    is_out_sparse = sch.all_tensors_bk[-1].is_sparse
    if is_out_sparse:
        tensor = sch.all_tensors_bk[-1]
        # set order
        tensor_all_axes_name = tensor.format.axes_name.keys()
        new_ordered_axes = []
        for axis_name in reordered_vars:
            if axis_name in tensor_all_axes_name:
                new_ordered_axes.append(axis_name)
        if config.get("reorder", None) != None:
            new_tensor = sch.FormatReorder(tensor, new_ordered_axes)
        else:
            new_tensor = tensor
        
        # set mode
        format_mode_subspace_name = "format_mode_{}".format(sch.tensor_name_lst[-1])
        format_modes = config.get(format_mode_subspace_name, None)
        if format_modes != None:
            assert len(tensor.shape) == len(format_modes), \
                    "[AutoTune.AddSimpleSchedule] Format mode count differ from tensor."
            format_modes_dict = {}
            for i in range(len(new_ordered_axes)):
                format_modes_dict[new_ordered_axes[i]] = format_modes[i]
            if CheckMode(new_ordered_axes, all_axes_size, format_modes_dict) == False:
                raise ValueError()
            for i, axis_name in enumerate(new_ordered_axes):
                sch.FormatMode(new_tensor, axis_name, format_modes[i])
                # If there are multiple tensors with the same axis, take the largest mode. 
                axes_mode[axis_name] = max(axes_mode.get(axis_name, 0), format_modes[i])
        else: # mode same with other input sparse axis
            for i, axis_name in enumerate(new_ordered_axes):
                if axis_name in axes_mode.keys():
                    sch.FormatMode(new_tensor, axis_name, axes_mode[axis_name])
                else:
                    sch.FormatMode(new_tensor, axis_name, 0)
    
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
        sch.SetParallelChunk(config["omp"][0])
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
    **kwargs
):
    """Warm find have there good program schedules.
    
    Parameters
    ----------
    schedule: Schedule
        It is pure object without any schedules.
    agent_group: DQNAgentGroup
    use_performance_model: bool

    """

    GET_TIMES = [0., 0., 0., 0.]
    if kwargs.get('GET_TIMES', None) != None:
        GET_TIMES = kwargs.get('GET_TIMES')

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
                test_time0 = time.time()
                if use_performance_model:
                    configs_performances.append(float('inf')) # Future todo.
                else:
                    configs_performances.append(
                        Evaluate(schedule, func, eval_warm_times,
                        eval_round, eval_timeout, eval_policy)
                    )
                test_time1 = time.time()
                GET_TIMES[-1] += test_time1 - test_time0
                if configs_performances[-1] < float("inf"):
                    record_valid = agent_group.Record(configs_indicse[i], 
                                       configs_performances[-1], use_sa=False)
                    if record_valid:
                        agent_group.AddSchedule(
                            schedule.GenConfigCommand()[1], configs_performances[-1])
            
            best_per = min(configs_performances)
            print (f"[AutoTune] Warm in {tri} trial best: {best_per:.8f}, "
                   f"history best: {agent_group.Top1()[1]}")
            best_value = min(best_value, best_per)
            
        if not (agent_group.Top1()[1] < float("inf")):
            warm_trial += 1
            repeat_count -= 1
            # Can't find a solution
            print ("[AutoTune][Warming] No valid schedule in warm up.")
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
    prefix: str = "",
    save_best_trace: bool = False,
    save_dirpath: str = "",
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

    best_trace = []

    # writer = SummaryWriter('runs/random_search_' + prefix)
    start_time = time.time()

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
        best_trace.append([tri, time.time() - start_time, agent_group.Top1()[1]])
        # writer.add_scalar('Local Best Score', local_best, tri)
        # writer.add_scalar('Global Best Score', agent_group.Top1()[1], tri)
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Random Search: Current time: {current_time} in trial {tri}", flush=True)

    # Save best trace
    if save_best_trace:
        filepath = os.path.join(save_dirpath, prefix)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(filepath, "random_searching_{}".format(
            datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')
        ))
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Relative Time (ms)", "Value"])
            for item in best_trace:
                writer.writerow(item)

    return agent_group.Top1()


def PSearching(
    schedule: Schedule,
    func: Build,
    agent_group: DQNAgentGroup,
    population_size: int = 10,
    trial: int = 50,
    early_stop:int = 5,
    use_performance_model: bool = False,
    eval_warm_times: int = 10,
    eval_round: int = 100,
    eval_timeout: float = None,
    eval_policy: str = "avg",
    prefix: str = "",
    save_best_trace: bool = False,
    save_dirpath: str = "",
    **kwargs
):
    """Only using SA method"""

    print("**************** Start P Search ****************")
    print(f"population_size         ={population_size}")
    print(f"trial                   ={trial}")
    print(f"early_stop              ={early_stop}")
    print(f"use_performance_model   ={use_performance_model}")
    print(f"eval_warm_times         ={eval_warm_times}")
    print(f"eval_round              ={eval_round}")
    print(f"eval_timeout            ={eval_timeout}")
    print(f"eval_policy             ={eval_policy}")
    print()

    warm_trial = 5
    warm_population_size = int(population_size / 4)
    print(f"[AutoTune] Warm {warm_trial} trial, each run {warm_population_size} data.")
    global_best = float('inf')
    warm_try= 0
    while (agent_group.Top1Value() < float('inf')) == False and warm_try < 5:
        global_best = Warm(
            schedule=schedule,
            func=func,
            agent_group = agent_group,
            use_performance_model=use_performance_model,
            warm_trial=warm_trial,
            population_size=warm_population_size,
            repeat_count=warm_trial,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy,
        )
        warm_try += 1

    real_populations = agent_group.action_num
    print(f"[INFO] SAsearching get all {real_populations} directions in one trial.")

    trial = math.ceil(trial * population_size / real_populations)
    print(f"[INFO] Update to {trial} trial, and each run {real_populations} data.")

    early_stop_count = 0
    retired_indices = []

    # writer = SummaryWriter('runs/p_search_' + prefix)
    best_trace = []
    start_time = time.time()

    for tri in range(trial):
        # Ramdom get a good point.
        top_indices, top_value = agent_group.TopRandom(gamma=0.5)
        # Get all direction population
        next_data_lst = agent_group.SelectionFull(
            top_indices, top_value, no_repeat=True
        )
        
        # Evaluation performance
        configs_performances = [] 
        for idx, data in enumerate(next_data_lst):
            indices, _, name, direction, next_indices = data
            sch = None
            indices_saw_lst = [str(indices), str(indices)]
            # Find a ok schedule
            while True:
                next_config = agent_group.GetConfigFfromIndices(next_indices)
                try:
                    sch = AddSimpleSchedule(schedule.compute_tensor, next_config)
                    break
                except ValueError:
                    # Go on in this direction to get next entry
                    sch = None
                    next_indices = agent_group.SelectOneAction(
                        next_indices, name, direction, no_repeat=True)
                    if next_indices == None or str(next_indices) in indices_saw_lst: # repeat 
                        break
                    else:
                        indices_saw_lst.append(str(next_indices))
                        continue
            
            # Evaluation
            if sch is not None:
                if use_performance_model:
                    configs_performances.append(float('inf')) # Future todo.
                else:
                    configs_performances.append(
                        Evaluate(sch, func, eval_warm_times,
                        eval_round, eval_timeout, eval_policy)
                    )
                if configs_performances[-1] < float("inf"):
                    record_valid = agent_group.Record(next_indices, configs_performances[-1], 
                                       use_sa=True, gamma=0.05)
                    if record_valid:
                        agent_group.AddSchedule(
                            sch.GenConfigCommand()[1], configs_performances[-1])
        
        best_trace.append([tri, time.time() - start_time, global_best])
        
        if len(configs_performances):
            best_per = min(configs_performances)
        else:
            early_stop_count = math.ceil(early_stop_count / 1.5)
            best_per = float('inf')

        # writer.add_scalar('Local Best Score', best_per, tri)
        # writer.add_scalar('Global Best Score', agent_group.Top1()[1], tri)

        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print (f"[AutoTune] P searching Current time: {current_time} in {tri}"
                f" trial best: {best_per:.8f}, "
                f"history best: {global_best}",
                flush=True)

        # Can't find more better one, so retire best one to continue explore.
        if (best_per < global_best):
            global_best = best_per
            early_stop_count = 0
        else:
            top = agent_group.PopTop()
            retired_indices.append((top.indices, top.value))
            early_stop_count += 1
        
        # Early stop
        if early_stop_count > early_stop:
            print(
                f"[AutoTune] P searching early stop with repeats {early_stop} times."
            )
            break
    
    for item in retired_indices:
        agent_group.Record(item[0], item[1], use_sa=False)
    
    # Save best trace
    if save_best_trace:
        filepath = os.path.join(save_dirpath, prefix)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(filepath, "p_searching_{}".format(
            datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')
        ))
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Relative Time (ms)", "Value"])
            for item in best_trace:
                writer.writerow(item)

    return agent_group.Top1()

# Batch P method
def BatchPSearching(
    schedule: Schedule,
    func: Build,
    agent_group: DQNAgentGroup,
    population_size: int = 10,
    trial: int = 50,
    early_stop:int = 5,
    use_performance_model: bool = False,
    eval_warm_times: int = 10,
    eval_round: int = 100,
    eval_timeout: float = None,
    eval_policy: str = "avg",
    prefix: str = "",
    save_best_trace: bool = False,
    save_dirpath: str = "",
    **kwargs
):
    """Only using batch p method"""
    print("**************** Start Batch P Search ****************")
    print(f"population_size         ={population_size}")
    print(f"trial                   ={trial}")
    print(f"early_stop              ={early_stop}")
    print(f"use_performance_model   ={use_performance_model}")
    print(f"eval_warm_times         ={eval_warm_times}")
    print(f"eval_round              ={eval_round}")
    print(f"eval_timeout            ={eval_timeout}")
    print(f"eval_policy             ={eval_policy}")
    print()

    warm_trial = 5
    warm_population_size = int(population_size)
    print(f"[AutoTune] Warm {warm_trial} trial, each run {warm_population_size} data.")
    global_best = float('inf')
    warm_try= 0
    while (agent_group.Top1Value() < float('inf')) == False and warm_try < 5:
        global_best = Warm(
            schedule=schedule,
            func=func,
            agent_group = agent_group,
            use_performance_model=use_performance_model,
            warm_trial=warm_trial,
            population_size=warm_population_size,
            repeat_count=warm_trial,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy,
        )
        warm_try += 1
    
    top_k = math.ceil(population_size * 5 / agent_group.action_num)

    early_stop_count = 0
    retired_indices = []

    # writer = SummaryWriter('runs/p_search_' + prefix)
    best_trace = []
    start_time = time.time()

    for tri in range(trial):
        # Random get next batch data
        topk_indices_lst, topk_value_lst = agent_group.TopK(top_k)
        next_data_lst = []
        for idx in range(len(topk_indices_lst)):
            next_data_lst.extend(agent_group.SelectionFull(
                topk_indices_lst[idx], topk_value_lst[idx], no_repeat=True
            ))
        random.shuffle(next_data_lst)
        
        # Evaluation performance
        configs_performances = [] 
        for idx, data in enumerate(next_data_lst):
            indices, _, name, direction, next_indices = data
            sch = None
            indices_saw_lst = [str(indices), str(indices)]
            # Find a ok schedule
            while True:
                next_config = agent_group.GetConfigFfromIndices(next_indices)
                try:
                    sch = AddSimpleSchedule(schedule.compute_tensor, next_config)
                    break
                except ValueError:
                    # Go on in this direction to get next entry
                    sch = None
                    next_indices = agent_group.SelectOneAction(
                        next_indices, name, direction, no_repeat=True)
                    if next_indices == None or str(next_indices) in indices_saw_lst: # repeat 
                        break
                    else:
                        indices_saw_lst.append(str(next_indices))
                        continue
            
            # Evaluation
            if sch is not None:
                if use_performance_model:
                    configs_performances.append(float('inf')) # Future todo.
                else:
                    configs_performances.append(
                        Evaluate(sch, func, eval_warm_times,
                        eval_round, eval_timeout, eval_policy)
                    )
                if configs_performances[-1] < float("inf"):
                    record_valid = agent_group.Record(next_indices, configs_performances[-1], 
                                       use_sa=True, gamma=0.05)
                    if record_valid:
                        agent_group.AddSchedule(
                            sch.GenConfigCommand()[1], configs_performances[-1])
            
            if len(configs_performances) == population_size:
                break

        best_trace.append([tri, time.time() - start_time, global_best])

        if len(configs_performances):
            best_per = min(configs_performances)
        else:
            early_stop_count = math.ceil(early_stop_count / 1.5)
            best_per = float('inf')

        # writer.add_scalar('Local Best Score', best_per, tri)
        # writer.add_scalar('Global Best Score', agent_group.Top1()[1], tri)

        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print (f"[AutoTune] Batch P searching Current time: {current_time} in {tri}"
                f" trial best: {best_per:.8f}, "
                f"history best: {global_best}",
                flush=True)

        # Can't find more better one, so retire best one to continue explore.
        if (best_per < global_best):
            global_best = best_per
            early_stop_count = 0
        else:
            early_stop_count += 1
            # Attempt to abandon top1.
            top1_indices, top1_value = agent_group.Top1()
            # Get all direction population
            next_data_lst = agent_group.SelectionFull(
                top1_indices, top1_value, no_repeat=True
            )
            valid_config_cnt = 0
            for idx, data in enumerate(next_data_lst):
                indices, _, name, direction, next_indices = data
                sch = None
                indices_saw_lst = [str(indices), str(indices)]
                # Find a ok schedule
                while True:
                    next_config = agent_group.GetConfigFfromIndices(next_indices)
                    try:
                        sch = AddSimpleSchedule(schedule.compute_tensor, next_config)
                        valid_config_cnt += 1
                        break
                    except ValueError:
                        # Go on in this direction to get next entry
                        sch = None
                        next_indices = agent_group.SelectOneAction(
                            next_indices, name, direction, no_repeat=True)
                        if next_indices == None or str(next_indices) in indices_saw_lst: # repeat 
                            break
                        else:
                            indices_saw_lst.append(str(next_indices))
                            continue
            if valid_config_cnt < population_size * 0.1:
                top = agent_group.PopTop()
                retired_indices.append((top.indices, top.value))
        
        # Early stop
        if early_stop_count > early_stop:
            print(
                f"[AutoTune] Batch P searching early stop with repeats {early_stop} times."
            )
            break

    for item in retired_indices:
        agent_group.Record(item[0], item[1], use_sa=False)
    
    # Save best trace
    if save_best_trace:
        filepath = os.path.join(save_dirpath, prefix)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(filepath, "batch_p_searching_{}".format(
            datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')
        ))
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Relative Time (ms)", "Value"])
            for item in best_trace:
                writer.writerow(item)

    return agent_group.Top1()

# Batch P method
def SASearching(
    schedule: Schedule,
    func: Build,
    agent_group: DQNAgentGroup,
    population_size: int = 10,
    trial: int = 50,
    early_stop:int = 5,
    use_performance_model: bool = False,
    eval_warm_times: int = 10,
    eval_round: int = 100,
    eval_timeout: float = None,
    eval_policy: str = "avg",
    prefix: str = "",
    save_best_trace: bool = False,
    save_dirpath: str = "",
    **kwargs
):
    """Only using SA method"""
    print("**************** Start SA Search ****************")
    print(f"population_size         ={population_size}")
    print(f"trial                   ={trial}")
    print(f"early_stop              ={early_stop}")
    print(f"use_performance_model   ={use_performance_model}")
    print(f"eval_warm_times         ={eval_warm_times}")
    print(f"eval_round              ={eval_round}")
    print(f"eval_timeout            ={eval_timeout}")
    print(f"eval_policy             ={eval_policy}")
    print()

    warm_trial = 5
    warm_population_size = int(population_size)
    print(f"[AutoTune] Warm {warm_trial} trial, each run {warm_population_size} data.")
    global_best = float('inf')
    warm_try= 0
    while (agent_group.Top1Value() < float('inf')) == False and warm_try < 5:
        global_best = Warm(
            schedule=schedule,
            func=func,
            agent_group = agent_group,
            use_performance_model=use_performance_model,
            warm_trial=warm_trial,
            population_size=warm_population_size,
            repeat_count=warm_trial,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy,
        )
        warm_try += 1
    
    early_stop_count = 0
    retired_indices = []

    population_size = int(population_size / len(agent_group.agent_group.keys()) * 2)

    # writer = SummaryWriter('runs/sa_search_' + prefix)
    best_trace = []
    start_time = time.time()

    for tri in range(trial):
        # Random get next batch data
        topk_indices_lst, topk_value_lst = agent_group.TopK(population_size, modify=True)
        configs_performances = [] 
        for idx in range(len(topk_indices_lst)):
            all_next_data_lst = agent_group.SelectionFull(
                topk_indices_lst[idx], topk_value_lst[idx], no_repeat=True
            )
            # random.shuffle(all_next_data_lst)
            # next_data_lst = {}
            # for item in all_next_data_lst:
            #     if item[2] not in next_data_lst:
            #         next_data_lst[item[2]] = item
            # next_data_lst = list(next_data_lst.values())
            next_data_lst = random.sample(all_next_data_lst, k = len(agent_group.agent_group.keys()))

            # Evaluation performance
            for idx, data in enumerate(next_data_lst):
                indices, _, name, direction, next_indices = data
                sch = None
                indices_saw_lst = [str(indices), str(indices)]
                # Find a ok schedule
                while True:
                    next_config = agent_group.GetConfigFfromIndices(next_indices)
                    try:
                        sch = AddSimpleSchedule(schedule.compute_tensor, next_config)
                        break
                    except ValueError:
                        # Go on in this direction to get next entry
                        sch = None
                        next_indices = agent_group.SelectOneAction(
                            next_indices, name, direction, no_repeat=True)
                        if next_indices == None or str(next_indices) in indices_saw_lst: # repeat 
                            break
                        else:
                            indices_saw_lst.append(str(next_indices))
                            continue
                
                # Evaluation
                if sch is not None:
                    if use_performance_model:
                        configs_performances.append(float('inf')) # Future todo.
                    else:
                        configs_performances.append(
                            Evaluate(sch, func, eval_warm_times,
                            eval_round, eval_timeout, eval_policy)
                        )
                    if configs_performances[-1] < float("inf"):
                        record_valid = agent_group.Record(next_indices, configs_performances[-1], 
                                        use_sa=True, gamma=0.05)
                        if record_valid:
                            agent_group.AddSchedule(
                                sch.GenConfigCommand()[1], configs_performances[-1])

        best_trace.append([tri, time.time() - start_time, global_best])
        
        if len(configs_performances):
            best_per = min(configs_performances)
        else:
            early_stop_count = math.ceil(early_stop_count / 1.5)
            best_per = float('inf')

        # writer.add_scalar('Local Best Score', best_per, tri)
        # writer.add_scalar('Global Best Score', agent_group.Top1()[1], tri)

        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print (f"[AutoTune] SA searching Current time: {current_time} in {tri}"
                f" trial best: {best_per:.8f}, "
                f"history best: {global_best}",
                flush=True)

        # Can't find more better one, so retire best one to continue explore.
        if (best_per < global_best):
            global_best = best_per
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        # Early stop
        if early_stop_count > early_stop:
            print(
                f"[AutoTune] SA searching early stop with repeats {early_stop} times."
            )
            break

        # Append modified topk
        for idx in range(len(topk_indices_lst)):
            retired_indices.append((topk_indices_lst[idx], topk_value_lst[idx]))

        # Reload data
        if (tri + 1) % 10:
            retired_indices = sorted(retired_indices, key=lambda x: x[1])
            added_indices = retired_indices[:population_size]
            retired_indices = retired_indices[population_size:]
            for item in added_indices:
                agent_group.Record(item[0], item[1], use_sa=False)

    for item in retired_indices:
        agent_group.Record(item[0], item[1], use_sa=False)

    # Save best trace
    if save_best_trace:
        filepath = os.path.join(save_dirpath, prefix)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(filepath, "sa_searching_{}".format(
            datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')
        ))
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Relative Time (ms)", "Value"])
            for item in best_trace:
                writer.writerow(item)

    return agent_group.Top1()


def QSearching(
    schedule: Schedule,
    func: Build,
    agent_group: DQNAgentGroup,
    population_size: int = 10,
    use_performance_model: bool = False,
    trial: int = 50,
    update_target_gap:int = 4,
    early_stop:int = 5,
    eval_warm_times: int = 10,
    eval_round: int = 100,
    eval_timeout: float = None,
    eval_policy: str = "avg",
    prefix: str = "",
    save_best_trace: bool = False,
    save_dirpath: str = "",
    **kwargs
):
    """Only using DQN method"""
    return  QSASearching(
            schedule = schedule,
            func=func,
            agent_group=agent_group,
            population_size=population_size,
            use_performance_model = use_performance_model,
            trial = trial,
            use_sa = False,
            sa_gamma=1,
            update_target_gap=update_target_gap,
            early_stop = early_stop,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy,
            prefix=prefix,
            save_best_trace = save_best_trace,
            save_dirpath = save_dirpath
        )


def QSASearching(
    schedule: Schedule,
    func: Build,
    agent_group: DQNAgentGroup,
    population_size: int = 10,
    use_performance_model: bool = False,
    trial: int = 50,
    use_sa: bool = True,
    sa_gamma: int = 0.05,
    update_target_gap:int = 4,
    early_stop:int = 5,
    eval_warm_times: int = 10,
    eval_round: int = 100,
    eval_timeout: float = None,
    eval_policy: str = "avg",
    prefix: str = "",
    save_best_trace: bool = False,
    save_dirpath: str = "",
    **kwargs
):
    """Using DQN and SA mixing method"""

    print("**************** Start Q Search ****************")
    print(f"population_size         ={population_size}")
    print(f"trial                   ={trial}")
    print(f"use_sa                  ={use_sa}")
    print(f"sa_gamma                ={sa_gamma}")
    print(f"update_target_gap       ={update_target_gap}")
    print(f"early_stop              ={early_stop}")
    print(f"use_performance_model   ={use_performance_model}")
    print(f"eval_warm_times         ={eval_warm_times}")
    print(f"eval_round              ={eval_round}")
    print(f"eval_timeout            ={eval_timeout}")
    print(f"eval_policy             ={eval_policy}")
    print()

    GET_TIMES = [0., 0., 0., 0.]
    if kwargs.get('GET_TIMES', None) != None:
        GET_TIMES = kwargs.get('GET_TIMES')

    search_time1 = time.time()

    warm_trial = 3
    warm_population_size = max(int(population_size/2), 35)
    print(f"[AutoTune] Warm {warm_trial} trial, each run {warm_population_size} data.")
    global_best = float('inf')
    warm_try= 0
    while (agent_group.Top1Value() < float('inf')) == False and warm_try < 5:
        global_best = Warm(
            schedule=schedule,
            func=func,
            agent_group = agent_group,
            use_performance_model=use_performance_model,
            warm_trial=warm_trial,
            population_size=warm_population_size,
            repeat_count=warm_trial,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy,
            GET_TIMES = GET_TIMES
        )
        warm_try += 1
    
    early_stop_count = 0
    retired_indices = []

    train_cnt = 0

    # if use_sa:
    #     writer = SummaryWriter('runs/q_sa_search_' + prefix)
    # else:
    #     writer = SummaryWriter('runs/q_search_' + prefix)
    
    population_size = int(population_size / len(agent_group.agent_group.keys()) * 2)

    best_trace = []
    start_time = time.time()

    for tri in range(trial):
        # Get topk 
        topk_indices_lst, topk_value_lst = agent_group.TopK(population_size, modify=True)
        # Random get next batch data
        next_data_lst = agent_group.SelectAction(
            topk_indices_lst, topk_value_lst, trial=tri, 
            epsilon=0.8, gamma=0.01
        )
        
        # Evaluation performance
        configs_performances = [] 
        for idx, data in enumerate(next_data_lst):
            indices, value, name, direction, next_indices = data
            sch = None
            indices_saw_lst = [str(indices), str(indices)]
            # Find a ok schedule
            while True:
                next_config = agent_group.GetConfigFfromIndices(next_indices)
                try:
                    sch = AddSimpleSchedule(schedule.compute_tensor, next_config)
                    break
                except ValueError:
                    # Go on in this direction to get next entry
                    sch = None
                    next_indices = agent_group.SelectOneAction(
                        next_indices, name, direction, no_repeat=True)
                    if next_indices == None or str(next_indices) in indices_saw_lst: # repeat 
                        break
                    else:
                        indices_saw_lst.append(str(next_indices))
                        continue
            
            # Evaluation
            if sch is not None:
                test_time0 = time.time()
                if use_performance_model:
                    configs_performances.append(float('inf')) # Future todo.
                else:
                    configs_performances.append(
                        Evaluate(sch, func, eval_warm_times,
                        eval_round, eval_timeout, eval_policy)
                    )
                test_time1 = time.time()
                GET_TIMES[-1] += test_time1 - test_time0
                if configs_performances[-1] < float("inf"):
                    record_valid = agent_group.Record(next_indices, configs_performances[-1], 
                                       use_sa=use_sa, gamma=sa_gamma)
                    # Add agent train data
                    reward = np.tanh(max(value - configs_performances[-1], 0.0))
                    agent_group.AddData(
                        indices, name, direction, next_indices, reward
                    )
                    if record_valid:
                        agent_group.AddSchedule(sch.GenConfigCommand()[1], configs_performances[-1])

        best_trace.append([tri, time.time() - start_time, global_best])

        if len(configs_performances):
            best_per = min(configs_performances)
        else:
            early_stop_count = math.ceil(early_stop_count / 1.5)
            best_per = float('inf')

        # writer.add_scalar('Local Best Score', best_per, tri)
        # writer.add_scalar('Global Best Score', agent_group.Top1()[1], tri)

        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print (f"[AutoTune] Q searching Current time: {current_time} in {tri}"
                f" trial best: {best_per:.8f}, "
                f"history best: {global_best}",
                flush=True)

        if (best_per < global_best):
            global_best = best_per
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        # Early stop
        if early_stop_count > early_stop:
            print(
                f"[AutoTune] SA searching early stop with repeats {early_stop} times."
            )
            break
        
        # Append modified topk
        for idx in range(len(topk_indices_lst)):
            retired_indices.append((topk_indices_lst[idx], topk_value_lst[idx]))

        # Train model
        if list(agent_group.agent_group.values())[0].memory_size * len(agent_group.agent_group.keys()) > \
                list(agent_group.agent_group.values())[0].train_batch_size:
            test_time2 = time.time()
            agent_group.Train(save_model=False)
            # Update target model
            if (train_cnt + 1) % update_target_gap == 0:
                # Stable train
                agent_group.UpdateAgentTargetModel()
                print(f"[AutoTune] Update DQN target net in {tri} trial.")
            test_time3 = time.time()
            GET_TIMES[2] += test_time3 - test_time2
        else:
            early_stop_count = 0
        train_cnt += 1

        # Restart search from other random point.
        if early_stop_count >= 10:
            print("[AutoSparse][AutoTune] No change has been made in 10 rounds, so the search is restarted.")
            agent_group.ClearMemory()

            local_best = Warm(
                schedule = schedule,
                func = func,
                agent_group=agent_group,
                use_performance_model=use_performance_model,
                warm_trial=warm_trial,
                population_size=warm_population_size,
                repeat_count=warm_trial,
                eval_warm_times = eval_warm_times,
                eval_round = eval_round,
                eval_timeout = eval_timeout,
                eval_policy = eval_policy,
                GET_TIMES = GET_TIMES
            )
            global_best = min(local_best, global_best)
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Try Random Search: Current time: {current_time} in trial {tri}", flush=True)

            early_stop_count = 0

        # # Reload data
        # if (train_cnt + 1) % update_target_gap == 0 and len(retired_indices):
        #     retired_indices = sorted(retired_indices, key=lambda x: x[1])
        #     added_indices = retired_indices[:population_size]
        #     retired_indices = retired_indices[population_size:]
        #     for idx, item in enumerate(added_indices):
        #         if (idx / (len(retired_indices)+0.001)) < random.random():
        #             agent_group.Record(item[0], item[1], use_sa=False)

    search_time2 = time.time()
    GET_TIMES[1] = search_time2 - search_time1

    for item in retired_indices:
        agent_group.Record(item[0], item[1], use_sa=False)

    # Save best trace
    if save_best_trace:
        filepath = os.path.join(save_dirpath, prefix)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        if use_sa:
            filepath = os.path.join(filepath, "q_sa_searching_{}".format(
                datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')
            ))
        else:
            filepath = os.path.join(filepath, "q_searching_{}".format(
                datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')
            ))
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Relative Time (ms)", "Value"])
            for item in best_trace:
                writer.writerow(item)

    agent_group.RecoverMemory()

    return agent_group.Top1()


def AutoTune(
    input: ComputeTensor,
    method: str = "random_searching",
    population_size: int = 100,
    trial: int = 100,
    early_stop:int = 10,
    use_his_agent: bool = False,
    use_performance_model: bool = False,
    performance_model_path: str = None,
    save_agent_model: bool = False,
    save_agent_data: bool = False,
    save_performance_model: bool = False,
    save_memory_data: bool = False,
    save_schedule_data: bool = False,
    save_best_trace: bool = False,
    save_dirpath: str = os.getenv("AUTOSPARSE_HOME"),
    eval_warm_times: int = 10,
    eval_round: int = 50,
    eval_timeout: float = None,
    eval_policy: str = "avg",
    **kwargs
):
    """
    Parameters
    ----------
    input: ComputeTensor
    method: str optinal("random_searching)
        method include:
        "random_searching", "p_searching", "batch_p_searching",
        "sa_searching", "q_searching", "q_sa_searching"
    population_size: int optinal(50)
        The size of the population during the search.
    trial:
        search trails, which multiply with population_size can know how many
        point will be explorated.
    early_stop:
        Stop early when the same result is found several times.
    use_his_agent: boll optinal(False)
        Wheather using history agent model and data.
    use_performance_model: boll optinal(False)
        Wheather using cost model to predict program performance.
    performance_model_path: str
        Load the off line performance model path.
    save_agent_model: Dict[str, str]
        Every agent's model path. The model have train in last searching.
    save_agent_data: Dict[str, str]
        Every agent's data which is apper in last searching.
    save_performance_model: bool
    save_memory_data: bool
        Save the search process data so that the search can resume.
    save_schedule_data: bool
        Save explored space.
    save_best_trace: bool
        The convergence history during the search process is saved.
    save_dirpath: str
        All file need to save into the directory path.
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
    GET_TIMES = [0., 0., 0., 0.]
    if kwargs.get('GET_TIMES', None) != None:
        GET_TIMES = kwargs.get('GET_TIMES')

    total_start_time = time.time()

    sch = None
    if isinstance(input, ComputeTensor):
        sch = CreateSchedule(input)
    elif isinstance(input, Schedule):
        sch = input
    else:
        raise ValueError("Input argument type error.")
    
    sparse_prefix = ""
    for tensor in sch.all_tensors_bk:
        if tensor.is_sparse:
            assert tensor.data, \
                f"[Autotune] Tensor {tensor} have not load data."
            sparse_prefix += os.path.basename(tensor.data).split('.')[0]


    # Step1: Init schedule design space with all the subspace
    tune_space = Space()

    origin_axes = sch.origin_axes
    axis_splited_diensions = dict()
    for name, axis in origin_axes.items():
        split_subspace = SplitSubSpace(axis[0].size, 2, policy="power2")
        tune_space.add_subspace(
            "splite_{}".format(name), split_subspace, "split"
        )
        axis_splited_diensions[name] = 2
    
    sorted_axes_name = sorted(list(axis_splited_diensions.keys()))
    sorted_axis_splited_diensions = \
        [axis_splited_diensions[name] for name in sorted_axes_name]
    
    reorder_subspace = ReorderSubSpace(
        len(origin_axes) * 2, sorted_axis_splited_diensions)
    tune_space.add_subspace(
        "reorder", reorder_subspace, "reorder"
    )

    is_out_sparse = sch.all_tensors_bk[-1].is_sparse
    have_other_sparse_related = False

    for i, tensor in enumerate(sch.all_tensors_bk[:-1]):
        if tensor.is_sparse:
            format_mode_subspace = FModeSubSpace(len(tensor.shape) * 2)
            tune_space.add_subspace(
                "format_mode_{}".format(sch.tensor_name_lst[i]),
                format_mode_subspace, "format_mode"
            )
            if is_out_sparse:
                out_tensor_axes_name = [
                    axis.name for axis in sch.all_tensors_bk[-1].format.axes]
                for axis in tensor.format.axes:
                    if axis.name in out_tensor_axes_name:
                        have_other_sparse_related = True
    
    if is_out_sparse and have_other_sparse_related == False:
        tensor = sch.all_tensors_bk[-1]
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

    print(f"[AutoTuing] Space size = {len(tune_space)}")

    # Step2: Create Agent Group.
    agent_group = DQNAgentGroup(
        sch.GetScheduleName(), tune_space, decay = 0.9,
        lr = 0.02, epochs=5, train_batch_size=int(population_size/2)
    )

    # Load performance model
    # if use_performance_model:
        # Load performance model
    
    if (use_his_agent):
        # Load agent group performance data
        agent_group.memory = agent_group.LoadMemoryData(
            os.path.join(save_dirpath, sparse_prefix, "memory_data.pth")
        )
        # Load AgenGroup Model and data
        agent_group.LoadAgentModel(os.path.join(save_dirpath, sparse_prefix))
        agent_group.LoadAgentData(os.path.join(save_dirpath, sparse_prefix))
        print("[AutoSparse][AutoTune] Load history agent model and data.")
    
    # Step3: Tuning with searching
    func = Build(sch)
    print(f"[AutoTuing] Origin format run time = {func.origin_time:.8f}", flush = True)
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
            eval_policy = eval_policy,
            prefix=sparse_prefix,
            save_best_trace=save_best_trace,
            save_dirpath=save_dirpath
        )
    elif method == "p_searching":
        indices, value = PSearching(
            schedule = sch,
            func=func,
            agent_group=agent_group,
            population_size=population_size,
            use_performance_model = use_performance_model,
            trial = trial,
            early_stop = early_stop,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy,
            prefix=sparse_prefix,
            save_best_trace=save_best_trace,
            save_dirpath=save_dirpath
        )
    elif method == "batch_p_searching":
        indices, value = BatchPSearching(
            schedule = sch,
            func=func,
            agent_group=agent_group,
            population_size=population_size,
            use_performance_model = use_performance_model,
            trial = trial,
            early_stop = early_stop,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy,
            prefix=sparse_prefix,
            save_best_trace=save_best_trace,
            save_dirpath=save_dirpath
        )
    elif method == "sa_searching":
        indices, value = SASearching(
            schedule = sch,
            func=func,
            agent_group=agent_group,
            population_size=population_size,
            use_performance_model = use_performance_model,
            trial = trial,
            early_stop = early_stop,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy,
            prefix=sparse_prefix,
            save_best_trace=save_best_trace,
            save_dirpath=save_dirpath
        )
    elif method == "q_searching":
        indices, value = QSearching(
            schedule = sch,
            func=func,
            agent_group=agent_group,
            population_size=population_size,
            use_performance_model = use_performance_model,
            trial = trial,
            update_target_gap= math.ceil(trial * 0.08),
            early_stop = early_stop,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy,
            prefix=sparse_prefix,
            save_best_trace=save_best_trace,
            save_dirpath=save_dirpath
        )
    elif method == "q_sa_searching":
        indices, value = QSASearching(
            schedule = sch,
            func=func,
            agent_group=agent_group,
            population_size=population_size,
            use_performance_model = use_performance_model,
            trial = trial,
            use_sa=True,
            sa_gamma=0.05,
            update_target_gap= math.ceil(trial * 0.08),
            early_stop = early_stop,
            eval_warm_times = eval_warm_times,
            eval_round = eval_round,
            eval_timeout = eval_timeout,
            eval_policy = eval_policy,
            prefix=sparse_prefix,
            save_best_trace=save_best_trace,
            save_dirpath=save_dirpath,
            GET_TIMES = GET_TIMES
        )
    else:
        indices, value = {}, None
        assert False, \
            "[AutoTune] Error search method."

    # Save agent group performance data
    if save_memory_data:
        agent_group.SaveMemoryData(
            os.path.join(save_dirpath, sparse_prefix, "memory_data.pth")
    )

    # Load AgenGroup Model and data
    if (save_agent_model):
        agent_group.SaveAgentModel(os.path.join(save_dirpath, sparse_prefix))
        print("[AutoSparse][AutoTune] Save agent model.")
    if (save_agent_data):
        agent_group.SaveAgentData(os.path.join(save_dirpath, sparse_prefix))
        print("[AutoSparse][AutoTune] Save agent data.")
    
    # Save explored space.
    if save_schedule_data:
        agent_group.SaveScheduleData(os.path.join(save_dirpath, sparse_prefix, method+"schedule_data.pth"))
        agent_group.SaveScheduleDataWithTxt(os.path.join(save_dirpath, sparse_prefix+'.txt'))
        print("[AutoSparse][AutoTune] Save explored shcedule history and value.")

    config = agent_group.GetConfigFfromIndices(indices)

    total_end_time = time.time()
    GET_TIMES[0] = total_end_time - total_start_time
    return AddSimpleSchedule(sch.compute_tensor, config)