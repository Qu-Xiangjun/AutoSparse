import os

import numpy as np
import tvm
import tvm.testing
from tvm import te, auto_scheduler, runtime, topi
from tvm.auto_scheduler import _ffi_api
from tvm.topi.utils import get_const_tuple
from tvm.topi.sparse.utils import random_bsr_matrix


s = random_bsr_matrix(1024,1024,32,32,0.25,"float32")
"""
s是BCSR格式
data:
    s.data 存储了稀疏矩阵中所有非零的块。每个块都是一个小的密集矩阵，其形状为 (bs_r, bs_c)，即块的行数和列数。
    在这个上下文中，s.data.shape == (num_blocks, bs_r, bs_c) 表示 data 数组包含 num_blocks 个大小为 (bs_r, bs_c) 的块。
indices:
    s.indices 存储了每个块对应的列索引（在一个以块为单位的列视图中）。这个数组的长度等于非零块的数量，每个元素指示相应块在块列中的位置。
    s.indices.shape == (num_blocks,) 表示有 num_blocks 个这样的索引，每个索引对应一个块。
indptr:
    s.indptr 是一个指针数组，用于索引 data 和 indices 中的元素，以确定每行（按块行处理）的开始和结束位置。这有点类似于CSR格式中用于行的索引方式。
    s.indptr.shape == (m // bs_r + 1,) 表示这个数组的长度是块行数加一。原因是需要一个额外的元素来标示最后一个块行结束的位置。如果 m 是行的总数，m // bs_r 就是矩阵中完整块行的数量。
"""
print(s.data.shape)
print(s.indptr.shape) # pos
print(s.indices.shape) # idx

@auto_scheduler.register_workload
def sparse_dense_pure(M, N, K, w_data_shape, w_indices_shape, w_indptr_shape, dtype):
    X = te.placeholder(shape=(M, K), dtype=dtype)
    W_data = te.placeholder(shape=w_data_shape, dtype=dtype)
    W_indices = te.placeholder(shape=w_indices_shape, dtype="int32")
    W_indptr = te.placeholder(shape=w_indptr_shape, dtype="int32")

    out = topi.nn.sparse_dense(X, W_data, W_indices, W_indptr)

    return [X, W_data, W_indices, W_indptr, out]


# Define the basic shapes of this sparse computation
M = 4096
K = 8192
N = 256
BS_R = 16
BS_C = 16
density = 1/40

# Generate the test data with numpy
X_np = np.random.randn(M, K).astype("float32")
# X_np = np.maximum(np.zeros((M, K), dtype="float32"), X_np)  # Relu
W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
W_np = W_sp_np.todense()
Y_np = X_np @ W_np.T  # Process the matrix multiplication
# B_np = np.random.randn(M, N).astype("float32")
# Y_np = Y_np + B_np  # Bias add
# Y_np = np.maximum(np.zeros((M, N), dtype="float32"), Y_np)  # Relu

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

# # Inspect the computational graph
# print("Computational DAG:")
# print(task.compute_dag)

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


log_file = "sparse_dense.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

search_policy = auto_scheduler.SketchPolicy(
    task,
    program_cost_model=auto_scheduler.XGBModel(),
    init_search_callbacks=[
        auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func, "SparseDense")
    ],
)

def tune_and_evaluate(tune_option, search_policy):
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
        "Execution time of this operator: %.3f ms"
        % (
            np.median(
                evaluator(X_tvm, W_data_tvm, W_indices_tvm, W_indptr_tvm, Y_tvm).results
            )
            * 1000
        )
    )


# Notice: We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.
tune_and_evaluate(tune_option, search_policy)