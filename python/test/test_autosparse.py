import os, sys

current = os.path.join(os.getcwd(), "python")
sys.path.append(current)

import AutoSparse as AS
from AutoSparse import Axis, Tensor, Format, ModeType

def test_schedule():
    k = Axis(64, ModeType.DENSE, 'k')
    j = Axis(256, ModeType.DENSE, 'j')

    A = Tensor(AS.CSR((32, 64), 'i', 'k'), is_sparse=True)
    B = Tensor((k, j), is_sparse=False)
    C = AS.Compute(A@B)

    sch = AS.CreateSchedule(C)

    # Add Schedule
    new_i_axes = sch.FormatSplit('i', [4, 8])
    print(new_i_axes)
    new_k_axes = sch.FormatSplit('k', [16, 4])
    print(new_k_axes)
    
    # i1, k0, i0, k1
    sch.FormatReorder(A, [new_i_axes[1], new_k_axes[0], new_i_axes[0], new_k_axes[1]])
    
    sch.FormatMode(A, new_i_axes[1], ModeType.SINGLETON_UN)
    sch.FormatMode(A, new_k_axes[0], ModeType.SINGLETON)
    sch.FormatMode(A, new_k_axes[1], ModeType.COMPRESSED)

    new_i1_axes = sch.LoopSplit(new_i_axes[1], [2, 4])
    print(new_i1_axes)
    new_j_axes = sch.LoopSplit('j', [8, 32])
    print(new_j_axes)

    sch.LoopReorder([*new_j_axes, new_i_axes[0], *new_i1_axes, *new_k_axes])

    sch.LoopVectorize(new_k_axes[-1])
    sch.LoopParallel(new_j_axes[0])
    sch.LoopUnroll(new_i_axes[0], 2)

    string = AS.Build(sch)

    print(string)

if __name__ == "__main__":
    test_schedule()