import os, sys

import numpy as np
import scipy.sparse as sp
import tvm
import tvm.testing
from tvm import te, auto_scheduler, runtime, topi
from tvm.auto_scheduler import _ffi_api
from tvm.topi.utils import get_const_tuple
from tvm.topi.sparse.utils import random_bsr_matrix

platform = 'xeon'

if platform == 'epyc':
    os.environ["TVM_NUM_THREADS"] = "128"
else:
    os.environ["TVM_NUM_THREADS"] = "16"

current_directory = os.path.dirname(os.path.abspath(__file__))

@auto_scheduler.register_workload
def sparse_dense_pure(M, N, K, w_data_shape, w_indices_shape, w_indptr_shape, dtype):
    X = te.placeholder(shape=(M, K), dtype=dtype)
    W_data = te.placeholder(shape=w_data_shape, dtype=dtype)
    W_indices = te.placeholder(shape=w_indices_shape, dtype="int32")
    W_indptr = te.placeholder(shape=w_indptr_shape, dtype="int32")

    out = topi.nn.sparse_dense(X, W_data, W_indices, W_indptr)

    return [X, W_data, W_indices, W_indptr, out]


def meet_condition_func(search_policy, state, stage_id):
    state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if state.stages[stage_id].op.tag in [
        "sparse_dense_sp_rhs_bsrmm",
        "sparse_dense_sp_rhs_bsrmm_block",
    ]:
        return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST
    else:
        return auto_scheduler.PreloadCustomSketchRule.PASS


def apply_func(search_policy, state, stage_id):
    ret = []
    s0 = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if s0.stages[stage_id].op.tag == "sparse_dense_sp_rhs_bsrmm_block":
        return [s0.state_object, stage_id - 1]

    sparse_dense = s0.stages[stage_id].op
    sparse_dense_block = s0.stages[stage_id - 1].op
    assert sparse_dense.tag == "sparse_dense_sp_rhs_bsrmm"
    assert sparse_dense_block.tag == "sparse_dense_sp_rhs_bsrmm_block"

    # Set the default consumer of compute block
    consumer = sparse_dense

    # If sparse dense has a single elementwise consumer
    # We can compute inline the sparse_dense output stage
    consumers = _ffi_api.SearchPolicyUtilsGetConsumers(
        search_policy.search_task, s0.state_object, stage_id
    )
    if len(consumers) == 1:
        consumer_id = int(consumers.items()[0][0])
        if _ffi_api.SearchPolicyUtilsIsElementwiseMatch(
            search_policy.search_task, s0.state_object, stage_id, consumer_id
        ):
            consumer = s0.stages[consumer_id].op
            s0.compute_inline(sparse_dense)

    i, nb_j, j, row_offset, c = s0[sparse_dense_block].iters
    m, n = s0[consumer].iters
    i0, i1, i2 = s0.split(sparse_dense_block, i, [None, None])
    m0, m1 = s0.follow_split(consumer, m, len(s0.transform_steps) - 1, 1)
    j0, j1 = s0.split(sparse_dense_block, nb_j, [None])
    n0, n1 = s0.follow_split(consumer, n, len(s0.transform_steps) - 1, 1)
    s0.reorder(sparse_dense_block, [i0, j0, i1, j1, row_offset, i2, j, c])
    s0.reorder(consumer, [m0, n0, m1, n1])
    s0.compute_at(sparse_dense_block, consumer, n0)

    ret.append([s0.state_object, stage_id - 2])

    return ret

def from_csr(filepath) :
    csr = np.fromfile(filepath, dtype='<i4')
    num_row,num_col,nnz = csr[0],csr[1],csr[2]
    coo = np.zeros((nnz,2),dtype=int)
    coo[:,1] = csr[3+num_row+1:3+num_row+1+nnz] # col
    bins = np.array(csr[4:num_row+4]) - np.array(csr[3:num_row+3])
    coo[:,0] = np.repeat(range(num_row), bins)
    return num_row, num_col, nnz, coo

def tune_for_spmm(filename, M):
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    csr_filepath = os.path.join(autosparse_prefix, "dataset", "demo_dataset", filename+".csr")
    num_row, num_col, nnz, coo = from_csr(csr_filepath)
    print(f"{filename}: num_row = {num_row}, num_col = {num_col}, nnz = {nnz}")

    # Define the basic shapes of this sparse computation
    K = int(num_col)
    N = int(num_row)
    BS_R = 16
    BS_C = 16

    # Generate the test data with numpy
    X_np = np.random.randn(M, K).astype("float32")
    W_np_ = np.zeros((num_row, num_col), dtype="float32")
    W_np_[coo[:,0], coo[:,1]] = 1.0
    W_sp_np = sp.bsr_matrix(W_np_, blocksize=(BS_R, BS_C))
    # W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=1/40, dtype="float32")
    W_np = W_sp_np.todense()
    Y_np = X_np @ W_np.T  # Process the matrix multiplication

    # replace "llvm" below with "llvm -mcpu=core-avx2" to enable AVX2
    # replace "llvm" below with "llvm -mcpu=skylake-avx512" to enable AVX-512
    target = tvm.target.Target("llvm")

    # Register the sparse data to task inputs
    prefix = "sparse_dense_bsr_%d_%d_%d_%d_%d_%d_" % (
        N,
        K,
        BS_R,
        BS_C,
        W_sp_np.indices.shape[0],
        W_sp_np.indptr.shape[0],
    )
    task = tvm.auto_scheduler.SearchTask(
        func=sparse_dense_pure,
        args=(M, N, K, W_sp_np.data.shape, W_sp_np.indices.shape, W_sp_np.indptr.shape, "float32"),
        target=target,
        task_inputs={
            prefix + "W_data": runtime.ndarray.array(W_sp_np.data),
            prefix + "W_indices": runtime.ndarray.array(W_sp_np.indices),
            prefix + "W_indptr": runtime.ndarray.array(W_sp_np.indptr),
        },
        task_inputs_save_to_file=True,
    )

    log_file = filename + "_sparse_dense.json"
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
        # early_stopping=10,
    )

    search_policy = auto_scheduler.SketchPolicy(
        task,
        program_cost_model=auto_scheduler.XGBModel(),
        init_search_callbacks=[
            auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func, "SparseDense")
        ],
    )

    # Run auto-tuning (search)
    task.tune(tune_option, search_policy)

    # Apply the best schedule
    sch, args = task.apply_best(log_file)

    # We can lower the schedule to see the IR after auto-scheduling.
    # The auto-scheduler correctly performs optimizations including multi-level tiling,
    # layout transformation, parallelization, vectorization, unrolling, and operator fusion.
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))

    # Check correctness and evaluate performance
    # We build the binary and check its correctness and performance.
    func = tvm.build(sch, args, target)

    dev = tvm.cpu()

    X_tvm = tvm.nd.array(X_np, device=dev)
    W_data_tvm = tvm.nd.array(W_sp_np.data, device=dev)
    W_indices_tvm = tvm.nd.array(W_sp_np.indices, device=dev)
    W_indptr_tvm = tvm.nd.array(W_sp_np.indptr, device=dev)
    # B_tvm = tvm.nd.array(B_np, device=dev)
    Y_tvm = tvm.nd.empty(Y_np.shape, device=dev)

    func(X_tvm, W_data_tvm, W_indices_tvm, W_indptr_tvm, Y_tvm)

    # Check results
    print(Y_np.shape)
    print(Y_tvm.numpy().shape)
    tvm.testing.assert_allclose(Y_np, Y_tvm.numpy(), atol=1e-4, rtol=1e-4)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    print(
        "%s Execution time of this operator: %.3f ms"
        % (
            filename,
            np.median(
                evaluator(X_tvm, W_data_tvm, W_indices_tvm, W_indptr_tvm, Y_tvm).results
            )
            * 1000
        )
    )

def evaluate_best_record(filename, M):
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    csr_filepath = os.path.join(autosparse_prefix, "dataset", "demo_dataset", filename+".csr")
    num_row, num_col, nnz, coo = from_csr(csr_filepath)
    print(f"{filename}: num_row = {num_row}, num_col = {num_col}, nnz = {nnz}")

    # Define the basic shapes of this sparse computation
    
    K = int(num_col)
    N = int(num_row)
    BS_R = 16
    BS_C = 16
    assert (N % BS_R == 0)
    assert (K % BS_C == 0)

    X_np = np.random.randn(M, K).astype("float32")
    W_np_ = np.zeros((num_row, num_col), dtype="float32")
    W_np_[coo[:,0], coo[:,1]] = 1.0
    W_sp_np = sp.bsr_matrix(W_np_, blocksize=(BS_R, BS_C))
    W_np = W_sp_np.todense()
    Y_np = X_np @ W_np.T  # Process the matrix multiplication

    # replace "llvm" below with "llvm -mcpu=core-avx2" to enable AVX2
    # replace "llvm" below with "llvm -mcpu=skylake-avx512" to enable AVX-512
    target = tvm.target.Target("llvm")

    prefix = "sparse_dense_bsr_%d_%d_%d_%d_%d_%d_" % (
        N,
        K,
        BS_R,
        BS_C,
        W_sp_np.indices.shape[0],
        W_sp_np.indptr.shape[0],
    )
    task = tvm.auto_scheduler.SearchTask(
        func=sparse_dense_pure,
        args=(M, N, K, W_sp_np.data.shape, W_sp_np.indices.shape, W_sp_np.indptr.shape, "float32"),
        target=target,
        task_inputs={
            prefix + "W_data": runtime.ndarray.array(W_sp_np.data),
            prefix + "W_indices": runtime.ndarray.array(W_sp_np.indices),
            prefix + "W_indptr": runtime.ndarray.array(W_sp_np.indptr),
        },
        task_inputs_save_to_file=True,
    )

    log_file = filename + "_sparse_dense.json"
    
    # Apply the best schedule
    sch, args = task.apply_best(log_file)

    # We build the binary and check its correctness and performance.
    func = tvm.build(sch, args, target)
    dev = tvm.cpu()

    X_tvm = tvm.nd.array(X_np, device=dev)
    W_data_tvm = tvm.nd.array(W_sp_np.data, device=dev)
    W_indices_tvm = tvm.nd.array(W_sp_np.indices, device=dev)
    W_indptr_tvm = tvm.nd.array(W_sp_np.indptr, device=dev)
    Y_tvm = tvm.nd.empty(Y_np.shape, device=dev)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    print(
        "%s Execution time of this operator: %.3f ms"
        % (
            filename,
            np.median(
                evaluator(X_tvm, W_data_tvm, W_indices_tvm, W_indptr_tvm, Y_tvm).results
            )
            * 1000
        )
    )

if __name__ == "__main__":
    mtx_names = [
        'strides_mask',
        'encoder.layer.10.output.dense.weight', # 768 3072
        'encoder.layer.11.output.dense.weight',
        'encoder.layer.8.output.dense.weight',
        'encoder.layer.9.intermediate.dense.weight', # 3072 768
        'encoder.layer.9.output.dense.weight'
    ]
    for name in mtx_names:
        if name == 'strides_mask':
            tune_for_spmm(name, 256)
        elif 'intermediate' in name:
            tune_for_spmm(name, 3072)
        else:
            tune_for_spmm(name, 768)
    # for name in mtx_names:
    #     if name == 'strides_mask':
    #         evaluate_best_record(name, 256)
    #     elif 'intermediate' in name:
    #         evaluate_best_record(name, 3072)
    #     else:
    #         evaluate_best_record(name, 768)

# nohup python tune_sparse_x86_usages.py > ./log/xeon_evaluation_usgaes_$(date +%Y%m%d%H%M).log 2>&1 & 
