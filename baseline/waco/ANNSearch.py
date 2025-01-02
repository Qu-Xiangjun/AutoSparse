import os, sys
import numpy as np
import hnswlib
from AutoSparse import *
from AutoSparse.model import cuda_device_id
from tqdm import tqdm
import multiprocessing
import heapq

from buildKNN import TrainingScheduleDataset
from model import *

current_directory = os.path.dirname(os.path.abspath(__file__))

def SpMVTask(filename):
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    mtx_filepath = os.path.join(
        autosparse_prefix, "dataset", "demo_dataset", filename
    )
    num_row, num_col, num_nonezero = np.fromfile(
        mtx_filepath, count=3, dtype = '<i4'
    )
    print(f"num_row={num_row}, num_col={num_col}, num_nonezero={num_nonezero}")
    """Axis declarations"""
    i = Axis(int(num_row), ModeType.DENSE, "i")
    k = Axis(int(num_col), ModeType.COMPRESSED, "k")
    k_ = Axis(int(num_col), ModeType.DENSE, "k")
    """Tensor declaration"""
    A = Tensor((i, k), is_sparse=True)
    B = Tensor((k_, ), is_sparse=False)
    """Calculation declaration"""
    C = Compute(A@B)
    """Auto-Tune and excute"""
    A.LoadData(mtx_filepath)

    func = Build(C)
    return C, func

def SpMMTask(filename):
    # Create task by Autosparse
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    mtx_filepath = os.path.join(
        autosparse_prefix, "dataset", "demo_dataset", filename
    )
    num_row, num_col, num_nonezero = np.fromfile(
        mtx_filepath, count=3, dtype = '<i4'
    )
    print(f"num_row={num_row}, num_col={num_col}, num_nonezero={num_nonezero}")
    """Axis declarations"""
    i = Axis(int(num_row), ModeType.DENSE, "i")
    k = Axis(int(num_col), ModeType.COMPRESSED, "k")
    k_ = Axis(int(num_col), ModeType.DENSE, "k")
    j = Axis(128, ModeType.DENSE, "j")
    """Tensor declaration"""
    A = Tensor((i, k), is_sparse=True)
    B = Tensor((k_, j), is_sparse=False)
    """Calculation declaration"""
    C = Compute(A@B)
    """Auto-Tune and excute"""
    A.LoadData(mtx_filepath)

    func = Build(C)
    return C, func

def SDDMMTask(filename):
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    mtx_filepath = os.path.join(
        autosparse_prefix, "dataset", "demo_dataset", filename
    )
    num_row, num_col, num_nonezero = np.fromfile(
        mtx_filepath, count=3, dtype = '<i4'
    )
    print(f"{filename} num_row={num_row}, num_col={num_col}, num_nonezero={num_nonezero}")
    """Axis declarations"""
    i = Axis(int(num_row), ModeType.DENSE, "i")
    j = Axis(int(num_col), ModeType.COMPRESSED, "j")
    i_ = Axis(int(num_row), ModeType.DENSE, "i")
    k_ = Axis(128, ModeType.DENSE, "k")
    k__ = Axis(128, ModeType.DENSE, "k")
    j_ = Axis(int(num_col), ModeType.DENSE, "j")
    i__ = Axis(int(num_row), ModeType.DENSE, "i")
    j__ = Axis(int(num_col), ModeType.COMPRESSED, "j")
    """Tensor declaration"""
    A = Tensor((i, j), is_sparse=True)
    B = Tensor((i_, k_), is_sparse=False)
    C = Tensor((k__, j_), is_sparse=False)
    """Calculation declaration"""
    D = Compute(A*(B@C), is_sparse=True, format=(i__, j__))
    """Auto-Tune and excute"""
    A.LoadData(mtx_filepath)
    D.LoadData(mtx_filepath)

    func = Build(D)
    return D, func

def CreateSpMVSchedule(ct: ComputeTensor, superschedule: str):
    superschedule = superschedule.split()
    i1s = int(superschedule[0])
    k1s = int(superschedule[1])
    order = superschedule[2:6]
    for idx, item in enumerate(order):
        if (item == 'i0'):
            order[idx] = 'i1'
        elif (item == 'i1'):
            order[idx] = 'i0'
        elif (item == 'j0'):
            order[idx] = 'j1'
        elif (item == 'j1'):
            order[idx] = 'j0'
        elif (item == 'k0'):
            order[idx] = 'k1'
        elif (item == 'k1'):
            order[idx] = 'k0'
    i0f = int(superschedule[6]) ^ 1
    i1f = int(superschedule[7]) ^ 1
    k0f = int(superschedule[8]) ^ 1
    k1f = int(superschedule[9]) ^ 1
    parallel_var = superschedule[10]
    if parallel_var == 'i1':
        parallel_var = 'i0'
    parnum = int(superschedule[11])
    parchunk = int(superschedule[12])
    sch = CreateSchedule(ct)
    sch.FormatSplit('i', [math.ceil(sch.all_axes['i'][0].size / i1s) , i1s])
    sch.FormatSplit('k', [math.ceil(sch.all_axes['k'][0].size / k1s) , k1s])
    for tensor in sch.all_tensors_bk:
        if tensor.is_sparse == True:
            tensor_ = sch.FormatReorder(tensor, order)
            sch.FormatMode(tensor_, 'i0', i0f)
            sch.FormatMode(tensor_, 'i1', i1f)
            sch.FormatMode(tensor_, 'k0', k0f)
            sch.FormatMode(tensor_, 'k1', k1f)
    sch.LoopReorder(order)
    sch.LoopParallel(parallel_var)
    sch.SetThreadNum(multiprocessing.cpu_count())
    sch.SetParallelChunk(parchunk)

    return sch


def CreateSpMMSchedule(ct: ComputeTensor, superschedule: str):
    superschedule = superschedule.split()
    i1s = int(superschedule[0])
    k1s = int(superschedule[1])
    j1s = int(superschedule[2])
    order = superschedule[3:9]
    for idx, item in enumerate(order):
        if (item == 'i0'):
            order[idx] = 'i1'
        elif (item == 'i1'):
            order[idx] = 'i0'
        elif (item == 'j0'):
            order[idx] = 'j1'
        elif (item == 'j1'):
            order[idx] = 'j0'
        elif (item == 'k0'):
            order[idx] = 'k1'
        elif (item == 'k1'):
            order[idx] = 'k0'
    i0f = int(superschedule[9]) ^ 1
    i1f = int(superschedule[10]) ^ 1
    k0f = int(superschedule[11]) ^ 1
    k1f = int(superschedule[12]) ^ 1
    parallel_var = superschedule[13]
    if parallel_var == 'i1':
        parallel_var = 'i0'
    parnum = int(superschedule[14])
    parchunk = int(superschedule[15])
    sch = CreateSchedule(ct)
    sch.FormatSplit('i', [math.ceil(sch.all_axes['i'][0].size / i1s) , i1s])
    sch.FormatSplit('k', [math.ceil(sch.all_axes['k'][0].size / k1s) , k1s])
    sch.FormatSplit('j', [math.ceil(sch.all_axes['j'][0].size / j1s) , j1s])
    format_reordered = []
    for item in order:
        if 'i' in item or 'k' in item:
            format_reordered.append(item)
    for tensor in sch.all_tensors_bk:
        if tensor.is_sparse == True:
            tensor_ = sch.FormatReorder(tensor, format_reordered)
            sch.FormatMode(tensor_, 'i0', i0f)
            sch.FormatMode(tensor_, 'i1', i1f)
            sch.FormatMode(tensor_, 'k0', k0f)
            sch.FormatMode(tensor_, 'k1', k1f)
    sch.LoopReorder(order)
    sch.LoopParallel(parallel_var)
    sch.SetThreadNum(multiprocessing.cpu_count())
    sch.SetParallelChunk(parchunk)

    return sch

def CreateSDDMMSchedule(ct: ComputeTensor, superschedule: str):
    superschedule = superschedule.split()
    i1s = int(superschedule[0])
    k1s = int(superschedule[1])
    j1s = int(superschedule[2])
    order = superschedule[3:9]
    for idx, item in enumerate(order):
        if (item == 'i0'):
            order[idx] = 'i1'
        elif (item == 'i1'):
            order[idx] = 'i0'
        elif (item == 'j0'):
            order[idx] = 'j1'
        elif (item == 'j1'):
            order[idx] = 'j0'
        elif (item == 'k0'):
            order[idx] = 'k1'
        elif (item == 'k1'):
            order[idx] = 'k0'
    i0f = int(superschedule[9]) ^ 1
    i1f = int(superschedule[10]) ^ 1
    j0f = int(superschedule[11]) ^ 1
    j1f = int(superschedule[12]) ^ 1
    parallel_var = superschedule[13]
    if parallel_var == 'i1':
        parallel_var = 'i0'
    parnum = int(superschedule[14])
    parchunk = int(superschedule[15])
    sch = CreateSchedule(ct)
    sch.FormatSplit('i', [math.ceil(sch.all_axes['i'][0].size / i1s) , i1s])
    sch.FormatSplit('j', [math.ceil(sch.all_axes['j'][0].size / j1s) , j1s])
    sch.FormatSplit('k', [math.ceil(sch.all_axes['k'][0].size / k1s) , k1s])
    format_reordered = []
    for item in order:
        if 'i' in item or 'j' in item:
            format_reordered.append(item)
    for tensor in sch.all_tensors_bk:
        if tensor.is_sparse == True:
            tensor_ = sch.FormatReorder(tensor, format_reordered)
            sch.FormatMode(tensor_, 'i1', i1f)
            sch.FormatMode(tensor_, 'i0', i0f)
            sch.FormatMode(tensor_, 'j1', j1f)
            sch.FormatMode(tensor_, 'j0', j0f)
    sch.LoopReorder(order)
    sch.LoopParallel(parallel_var)
    sch.SetThreadNum(multiprocessing.cpu_count())
    sch.SetParallelChunk(parchunk)

    return sch

def RunSchedule(func: Build, sch: Schedule):
    # value = func.Run(sch, 10, 50)
    # if value < 0:
    #     print('ERROR ans')
    #     return float('inf')
    # return value
    return auto_tune.Evaluate(sch, func, 10, 50, eval_timeout=math.ceil(
            func.origin_time * 2 * (10 + 50) / 1000))

class MemEntry(object):
    def __init__(self, id, value):
        self.id = id
        self.value = value
    
    def __lt__(self, b):
        return self.value < b.value

def ANNS(task_name = "SpMM", matrix_filename = "nemspmm1_16x4_0.csr", warm_number = 100, trials = 500, k = 60):
    print(f"[WACO ANNS] Searching task {task_name} for matrix {matrix_filename}"
        f" with warm_number = {warm_number}, trials = {trials} and k = {k}")

    device = torch.device("cuda:" + str(cuda_device_id) if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    waco_prefix = os.path.join(autosparse_prefix, "baseline", "waco")

    schedules = TrainingScheduleDataset(os.path.join(waco_prefix, task_name, "TrainingData", "total.txt"), task_name)
    schedule_loader = torch.utils.data.DataLoader(schedules, batch_size=128, shuffle=False, num_workers=0)

    if (task_name == 'SpMM'):
        net = ResNet14SpMM(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    elif task_name == 'SpMV':
        net = ResNet14SpMV(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    elif task_name == 'SDDMM':
        net = ResNet14SDDMM(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    else:
        assert (False, "Error task name.")
    net = net.to(device)
    net.load_state_dict(torch.load(os.path.join(waco_prefix, task_name, 'resnet.pth')))
    net.eval()

    embeddings = [] 
    for batch_idx, (data, string) in tqdm(enumerate(schedule_loader), total=len(schedule_loader)):
        data = data.to(device)
        embedding = net.embed_super_schedule(data)
        embeddings.extend(embedding.detach().cpu().tolist())
    embeddings = np.array(embeddings)
    
    dim = len(schedules[0])
    num_elements = schedules.__len__()

    # Set KNN graph
    p = hnswlib.Index(space='l2', dim=dim)
    p.load_index(os.path.join(waco_prefix, task_name, "hnsw_schedule.bin"), max_elements = num_elements)
    p.set_ef(200) # ef should always be > k

    # Create task
    if task_name == "SpMV":
        st, func = SpMVTask(matrix_filename)
        createScheuleFunc = CreateSpMVSchedule
    elif task_name == "SpMM":
        st, func = SpMMTask(matrix_filename)
        createScheuleFunc = CreateSpMMSchedule
    elif task_name == "SDDMM":
        st, func = SDDMMTask(matrix_filename)
        createScheuleFunc = CreateSDDMMSchedule
    else:
        assert False, "Error task name."
    
    print(f"[WACO ANNS] CSR origin time = {func.origin_time:.8f} ms")

    memory: list[MemEntry] = [] # Record searched point.
    visited_set = set() # Store visited point id.
    retiered_item = []
    
    # Warm
    ids = list(random.sample(range(0, num_elements - 1), warm_number))
    print(f"[WACO ANNS] Warm up {warm_number} times.")
    for i, idx in enumerate(ids):
        if idx in visited_set:
            continue
        sch = createScheuleFunc(st, schedules[idx][1])
        value = RunSchedule(func, sch)
        heapq.heappush(memory, MemEntry(idx, value))
        visited_set.add(idx)
        print(f"[WACO ANNS] Warm up {i} times = {value:.8f} \t\t {schedules[idx][1]}", flush = True)
    print(f"[WACO ANNS] Find best id in warm up for {schedules[idx][1]} = {memory[0].value}", flush = True)

    early_stop = 20
    early_stop_cnt = 0

    global_best_value = memory[0].value

    for tri in range(trials):
        if len(memory) == 0:
            print("[WACO ANNS] Stop search when there have not any item in memory.")
            break
        
        cur_id = memory[0].id
        if (tri==0):
            print(schedules[cur_id][1])
        labels, dis = p.knn_query(embeddings[cur_id], k = k) # Find closest k neighbor.
        values = []
        run_ids = []
        for idx in labels.tolist()[0]:
            if idx in visited_set:
                continue
            sch = createScheuleFunc(st, schedules[int(idx)][1])
            if (tri==0):
                # print(schedules[int(idx)][0])
                print(schedules[int(idx)][1])
            value = RunSchedule(func, sch)
            visited_set.add(idx)
            values.append(value)
            run_ids.append(idx)

        if len(values):
            best_per = min(values)
        else:
            best_per = float('inf')
        print (f"[WACO ANNS] Search in {tri} trial best: {best_per:.8f}, "
                   f"history best: {global_best_value:.8f}", flush = True)
        
        if (best_per > global_best_value):
            if (best_per < float('inf')):
                early_stop_cnt += 1
            retiered_item.append(heapq.heappop(memory))
        else:
            early_stop_cnt = 0
            global_best_value = best_per
        
        if early_stop_cnt >= early_stop:
            print(f"[WACO ANNS] Early stop in reapte {early_stop} times.")
            break
        
        for idx, val in enumerate(values):
            heapq.heappush(memory, MemEntry(run_ids[idx], val))

    for item in retiered_item:
        heapq.heappush(memory, item)
        
    return memory[0].id, schedules[memory[0].id][1], memory[0].value


def ANNS2(task_name = "SpMM", matrix_filename = "nemspmm1_16x4_0.csr", warm_number = 100, trials = 500, k = 60):
    print(f"[WACO ANNS] Searching task {task_name} for matrix {matrix_filename}"
        f" with warm_number = {warm_number}, trials = {trials} and k = {k}")

    device = torch.device("cuda:" + str(cuda_device_id) if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    waco_prefix = os.path.join(autosparse_prefix, "baseline", "waco")

    schedules = TrainingScheduleDataset(os.path.join(waco_prefix, task_name, "TrainingData", "total.txt"), task_name)
    schedule_loader = torch.utils.data.DataLoader(schedules, batch_size=128, shuffle=False, num_workers=0)

    if (task_name == 'SpMM'):
        net = ResNet14SpMM(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    elif task_name == 'SpMV':
        net = ResNet14SpMV(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    elif task_name == 'SDDMM':
        net = ResNet14SDDMM(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    else:
        assert (False, "Error task name.")
    net = net.to(device)
    net.load_state_dict(torch.load(os.path.join(waco_prefix, task_name, 'resnet.pth')))
    net.eval()

    embeddings = [] 
    for batch_idx, (data, string) in tqdm(enumerate(schedule_loader), total=len(schedule_loader)):
        data = data.to(device)
        embedding = net.embed_super_schedule(data)
        embeddings.extend(embedding.detach().cpu().tolist())
    embeddings = np.array(embeddings)
    
    dim = len(schedules[0])
    num_elements = schedules.__len__()

    # Set KNN graph
    p = hnswlib.Index(space='l2', dim=dim)
    p.load_index(os.path.join(waco_prefix, task_name, "hnsw_schedule.bin"), max_elements = num_elements)
    p.set_ef(200) # ef should always be > k

    # Create task
    if task_name == "SpMV":
        st, func = SpMVTask(matrix_filename)
        createScheuleFunc = CreateSpMVSchedule
    elif task_name == "SpMM":
        st, func = SpMMTask(matrix_filename)
        createScheuleFunc = CreateSpMMSchedule
    elif task_name == "SDDMM":
        st, func = SDDMMTask(matrix_filename)
        createScheuleFunc = CreateSDDMMSchedule
    else:
        assert False, "Error task name."
    
    print(f"[WACO ANNS] CSR origin time = {func.origin_time:.8f} ms")

    memory: list[MemEntry] = [] # Record searched point.
    visited_set = set() # Store visited point id.
    retiered_item: list[MemEntry] = []
    
    # Warm
    ids = list(random.sample(range(0, num_elements - 1), warm_number))
    print(f"[WACO ANNS] Warm up {warm_number} times.")
    for i, idx in enumerate(ids):
        if idx in visited_set:
            continue
        sch = createScheuleFunc(st, schedules[idx][1])
        value = RunSchedule(func, sch)
        heapq.heappush(memory, MemEntry(idx, value))
        visited_set.add(idx)
        print(f"[WACO ANNS] Warm up {i} times = {value:.8f} \t\t {schedules[idx][1]}", flush = True)
    print(f"[WACO ANNS] Find best id in warm up for {schedules[idx][1]} = {memory[0].value}", flush = True)

    # early_stop = 20
    # early_stop_cnt = 0

    global_best_value = memory[0].value
    cur_id = memory[0].id
    
    for tri in range(trials):
        if len(memory) == 0:
            print("[WACO ANNS] Stop search when there have not any item in memory.")
            break
        
        if (tri==0):
            print(schedules[cur_id][1])
        labels, dis = p.knn_query(embeddings[cur_id], k = k) # Find closest k neighbor.
        values = []
        run_ids = []
        for idx in labels.tolist()[0]:
            if idx in visited_set:
                continue
            sch = createScheuleFunc(st, schedules[int(idx)][1])
            if (tri==0):
                # print(schedules[int(idx)][0])
                print(schedules[int(idx)][1])
            value = RunSchedule(func, sch)
            visited_set.add(idx)
            values.append(value)
            run_ids.append(idx)

        if len(values):
            best_per = min(values)
        else:
            best_per = float('inf')
        print (f"[WACO ANNS] Search in {tri} trial best: {best_per:.8f}, "
                   f"history best: {global_best_value:.8f}", flush = True)
        
        if (best_per > global_best_value):
            # if (best_per < float('inf')):
            #     early_stop_cnt += 1
            pass
        else:
            # early_stop_cnt = 0
            global_best_value = best_per
        
        if (best_per < float('inf')):
            cur_id = labels.tolist()[0][np.argmin(values)]
        else:
            retiered_item.append(heapq.heappop(memory))
            cur_id = retiered_item[-1].id

        # if early_stop_cnt >= early_stop:
        #     print(f"[WACO ANNS] Early stop in reapte {early_stop} times.")
        #     break
        
        for idx, val in enumerate(values):
            heapq.heappush(memory, MemEntry(run_ids[idx], val))

    for item in retiered_item:
        heapq.heappush(memory, item)
        
    return memory[0].id, schedules[memory[0].id][1], memory[0].value


def ANNS3(task_name = "SpMM", matrix_filename = "nemspmm1_16x4_0.csr", 
          warm_number = 100, trials = 500, k = 60, 
          save_res = False, save_dirpath: str = ""):
    print(f"[WACO ANNS] Searching task {task_name} for matrix {matrix_filename}"
        f" with warm_number = {warm_number}, trials = {trials} and k = {k}")

    device = torch.device("cuda:" + str(cuda_device_id) if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    waco_prefix = os.path.join(autosparse_prefix, "baseline", "waco")

    schedules = TrainingScheduleDataset(os.path.join(waco_prefix, task_name, "TrainingData", "total.txt"), task_name)
    schedule_loader = torch.utils.data.DataLoader(schedules, batch_size=128, shuffle=False, num_workers=0)

    if (task_name == 'SpMM'):
        net = ResNet14SpMM(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    elif task_name == 'SpMV':
        net = ResNet14SpMV(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    elif task_name == 'SDDMM':
        net = ResNet14SDDMM(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    else:
        assert (False, "Error task name.")
    net = net.to(device)
    pretrain_static_dict = torch.load(os.path.join(waco_prefix, task_name, 'resnet.pth'), map_location=device)
    new_state_dict = {}
    for key, value in pretrain_static_dict.items():
        if key in net.state_dict():
            new_state_dict[key] = value
    net.load_state_dict(new_state_dict)
    net.eval()

    embeddings = [] 
    for batch_idx, (data, string) in tqdm(enumerate(schedule_loader), total=len(schedule_loader)):
        data = data.to(device)
        embedding = net.embed_super_schedule(data)
        embeddings.extend(embedding.detach().cpu().tolist())
    embeddings = np.array(embeddings)
    
    dim = len(schedules[0])
    num_elements = schedules.__len__()

    # Set KNN graph
    p = hnswlib.Index(space='l2', dim=dim)
    p.load_index(os.path.join(waco_prefix, task_name, "hnsw_schedule.bin"), max_elements = num_elements)
    p.set_ef(200) # ef should always be > k

    # Create task
    if task_name == "SpMV":
        st, func = SpMVTask(matrix_filename)
        createScheuleFunc = CreateSpMVSchedule
    elif task_name == "SpMM":
        st, func = SpMMTask(matrix_filename)
        createScheuleFunc = CreateSpMMSchedule
    elif task_name == "SDDMM":
        st, func = SDDMMTask(matrix_filename)
        createScheuleFunc = CreateSDDMMSchedule
    else:
        assert False, "Error task name."
    
    print(f"[WACO ANNS] CSR origin time = {func.origin_time:.8f} ms")

    memory: list[MemEntry] = [] # Record searched point.
    visited_set = set() # Store visited point id.
    retiered_item: list[MemEntry] = []
    
    # Warm
    ids = list(random.sample(range(0, num_elements - 1), warm_number))
    print(f"[WACO ANNS] Warm up {warm_number} times.")
    for i, idx in enumerate(ids):
        if idx in visited_set:
            continue
        sch = createScheuleFunc(st, schedules[idx][1])
        value = RunSchedule(func, sch)
        heapq.heappush(memory, MemEntry(idx, value))
        visited_set.add(idx)
        print(f"[WACO ANNS] Warm up {i} times = {value:.8f} \t\t {schedules[idx][1]}", flush = True)
    print(f"[WACO ANNS] Find best id in warm up for {schedules[idx][1]} = {memory[0].value}", flush = True)

    # early_stop = 20
    # early_stop_cnt = 0

    global_best_value = memory[0].value
    cur_id = memory[0].id

    best_trace = []
    start_time = time.time()
    
    for tri in range(trials):
        if len(memory) == 0:
            print("[WACO ANNS] Stop search when there have not any item in memory.")
            break
        
        # if (tri==0):
            # print(schedules[cur_id][1])
        labels, dis = p.knn_query(embeddings[cur_id], k = k) # Find closest k neighbor.
        values = []
        run_ids = []
        for idx in labels.tolist()[0]:
            if idx in visited_set:
                continue
            sch = createScheuleFunc(st, schedules[int(idx)][1])
            # if (tri==0):
            #     # print(schedules[int(idx)][0])
            #     print(schedules[int(idx)][1])
            value = RunSchedule(func, sch)
            visited_set.add(idx)
            values.append(value)
            run_ids.append(idx)

        if len(values):
            best_per = min(values)
        else:
            best_per = float('inf')
        print (f"[WACO ANNS] Search in {tri} trial best: {best_per:.8f}, "
                   f"history best: {global_best_value:.8f}", flush = True)
        
        if not (best_per < float('inf')):
            retiered_item.append(heapq.heappop(memory))
        
        if (best_per > global_best_value):
            # if (best_per < float('inf')):
            #     early_stop_cnt += 1
            cur_id = memory[0].id
        else:
            # early_stop_cnt = 0
            global_best_value = best_per
            cur_id = labels.tolist()[0][np.argmin(values)]
        
        best_trace.append(
            [tri, time.time() - start_time, global_best_value]
        )
        
        # if early_stop_cnt >= early_stop:
        #     print(f"[WACO ANNS] Early stop in reapte {early_stop} times.")
        #     break
        
        for idx, val in enumerate(values):
            heapq.heappush(memory, MemEntry(run_ids[idx], val))

    for item in retiered_item:
        heapq.heappush(memory, item)
    
    # Save best trace
    if save_res:
        filedir = save_dirpath
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        csv_filepath = os.path.join(filedir, "anns_searching_{}".format(
            datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')
        ))
        with open(csv_filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Relative Time (ms)", "Value"])
            for item in best_trace:
                writer.writerow(item)
        
        # save all the point explored.
        schedule_data = [[schedules[int(item.id)][1], item.value]for item in memory]
        torch.save(schedule_data, os.path.join(
            filedir, "schedule_data_{}.pth".format(
                datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            ) 
        ))

    return memory[0].id, schedules[memory[0].id][1], memory[0].value

def test_spmm():
    schedules = [
        '2 8 64 i1 i0 k1 j1 k0 j0 0 1 0 1 i1 48 64', # 8.
        '8 32 32 i1 k1 j1 i0 k0 j0 0 1 0 1 i1 48 16', # 61.46027374 	
        '16 8 1 k1 i1 k0 i0 j1 j0 0 1 0 1 i1 48 64', # 344.70513916 	
        '4 16 128 i1 j1 k1 k0 i0 j0 0 0 1 1 i1 48 16', # 20512.62500000
        '4 1 4 i1 k1 j1 i0 k0 j0 0 1 0 1 i1 48 16', # 21167.08398438
        '4 1024 32 i1 k1 i0 j1 k0 j0 1 0 0 0 i1 48 256', # 63.75662994 	
        '512 8 128 k1 k0 i1 j1 i0 j0 1 1 1 1 i1 48 16', # error ans
        '128 64 16 i1 k1 i0 j1 k0 j0 1 0 0 0 i1 48 256', # 55.66510773 	
        '64 256 32 k1 i1 j1 k0 i0 j0 1 0 1 1 i1 48 16', # 5720.25048828 
        '32 16 8 i1 k1 k0 i0 j1 j0 1 0 1 1 i1 48 4', # 1726.12048340 
        '64 1024 4 k1 i1 i0 k0 j1 j0 0 0 0 0 i1 48 1', # 54.36511230
    ]

    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    waco_prefix = os.path.join(autosparse_prefix, "baseline", "waco")

    st, func = SpMMTask("nemspmm1_16x4_0.csr")
    createScheuleFunc = CreateSpMMSchedule

    print(f"[WACO] CSR origin time = {func.origin_time:.8f} ms")

    string = schedules[0]
    sch = createScheuleFunc(st, string)
    value = RunSchedule(func, sch)
    print(f"{string} = {value:.8f}")

    # string = schedules[1]
    # sch = createScheuleFunc(st, string)
    # value = RunSchedule(func, sch)
    # print(f"{string} = {value:.8f}")


if __name__ == "__main__":
    waco_prefix = os.getenv("AUTOSPARSE_HOME")
    platform = "xeon" # epyc xeon

    matrix_total_file = os.path.join(waco_prefix, "dataset", "total.txt")
    with open(matrix_total_file) as f:
        matrix_names = f.read().splitlines()

    matrix_names = [
        "bcsstk38",
        "mhd4800a",
        "conf5_0-4x4-18",
        "cca",
        "Trefethen_20000",
        "pf2177",
        "msc10848",
        "cfd1",
        "net100",
        "vanbody",
        "net150",
        "Chevron3_4x16_1",
        "vibrobox_1x1_0",
        "NACA0015_16x8_9",
        "nemspmm1_16x4_0",
        "Trec6_16x16_9",
        "crystk01_2x16_1",
        "t2dal_a_8x4_3",
        "EX1_8x8_4"
    ]

    for task_name in ["SDDMM"]: # "SpMM" "SpMV",
        for name in matrix_names:
            print(f"task {task_name} for matrix {name}")
            save_dirpath_prefix = os.path.join(
                waco_prefix, "baseline", "waco", task_name, platform+"_evaluation", name
            )
            if not os.path.exists(save_dirpath_prefix):
                os.makedirs(save_dirpath_prefix)

            result = ANNS3(
                task_name, name + '.csr', 100, 150, 60, 
                save_res=True, 
                save_dirpath=save_dirpath_prefix
            )
            res_f = open(os.path.join(waco_prefix, "baseline", "waco", platform+"_result_" + task_name + ".txt"), 'a')
            string = "{0} {1} {2} {3} {4} \n".format(
                task_name, name, result[0], result[1], result[2]
            )
            res_f.write(string)
            res_f.close()

    # task_name = 'SpMV'
    # id, schedules, value = ANNS3(task_name, "Trec6_16x16_9.csr")
    # print(schedules + f" {value:.8f}")

    # test_spmm()

# nohup python ANNSearch.py > ./log/epyc_evaluation_$(date +%Y%m%d%H%M).log 2>&1 & 
# nohup python ANNSearch.py > ./log/xeon_evaluation_$(date +%Y%m%d%H%M).log 2>&1 & 

# nohup python ANNSearch.py > ./log/xeon_evaluation_usages$(date +%Y%m%d%H%M).log 2>&1 & 
