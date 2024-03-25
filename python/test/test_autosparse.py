import os, sys

current = os.path.join(os.getcwd(), "python")
sys.path.append(current)

import AutoSparse as AS
from AutoSparse import Axis, Tensor, Format, ModeType

def test_schedule():
    k = Axis(128, ModeType.DENSE, 'k')
    j = Axis(256, ModeType.DENSE, 'j')

    A = Tensor(AS.CSR((64, 128), 'i', 'k'), is_sparse=True)
    B = Tensor((k, j), is_sparse=False)
    C = AS.Compute(A@B)

    sch = AS.CreateSchedule(C)

    # Add Schedule
    new_i_axes = sch.FormatSplit('i', [8, 8])
    print(new_i_axes)
    new_k_axes = sch.FormatSplit('k', [2, 64])
    print(new_k_axes)
    new_j_axes = sch.FormatSplit('j', [32, 8])
    print(new_k_axes)
    
    # i1, k0, i0, k1
    sch.FormatReorder(A, [new_k_axes[1], new_i_axes[1], new_k_axes[0], new_i_axes[0]])
    
    sch.FormatMode(A, new_i_axes[0], ModeType.SINGLETON)
    sch.FormatMode(A, new_i_axes[1], ModeType.COMPRESSED_UN)
    sch.FormatMode(A, new_k_axes[0], ModeType.COMPRESSED_UN)
    sch.FormatMode(A, new_k_axes[1], ModeType.COMPRESSED)

    # new_i1_axes = sch.LoopSplit(new_i_axes[1], [2, 8])
    # print(new_i1_axes)
    new_j0_axes = sch.LoopSplit(new_j_axes[0], [4, 8])
    print(new_j0_axes)

    # k1 i1 k0 i0 j00 j01 j1
    sch.LoopReorder([new_k_axes[1], new_i_axes[1], new_k_axes[0], new_i_axes[0], *new_j0_axes, new_j_axes[1]])

    sch.LoopParallel(new_i_axes[1])
    # sch.LoopVectorize(new_j_axes[1])
    # sch.LoopUnroll(new_j_axes[1], 16)

    A.LoadData("/home/qxj/AutoSparse/dataset/demo_dataset/__test_matrix.csr")

    print(A)

    func = AS.Build(sch)

    print(func.Run())

if __name__ == "__main__":
    test_schedule()