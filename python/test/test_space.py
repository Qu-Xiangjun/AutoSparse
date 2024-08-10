import os, sys
import pytest
import random

current = os.path.join(os.getcwd(), "python")
sys.path.append(current)

import AutoSparse as AS
from AutoSparse import space

@pytest.mark.parametrize("axis_size", [256, 769, 5978])
@pytest.mark.parametrize("dim", [2, 3, 4, 5])
@pytest.mark.parametrize("policy", ["power2", "factorization", "mixing"])
def test_splitsubspace(axis_size, dim , policy):
    ssspace = space.SplitSubSpace(axis_size, dim, policy)
    print("-------------- test_splitsubspace ---------------")
    print(f"axis_size = {axis_size}, dim = {dim}, policy = {policy}")
    print("SplitSubSpace all entries size = " + str(ssspace.size))
    print("SplitSubSpace directions num = " + str(ssspace.num_directions))
    current_pos = random.randint(0, ssspace.size - 1)
    next_direction = ssspace.GetDirection(
        random.randint(0, ssspace.num_directions - 1))
    print("SplitSubSpace Now position " + str(current_pos))
    print("SplitSubSpace Now entry " + str(ssspace.GetEntry(current_pos)))
    print("SplitSubSpace next direction " + str(next_direction))
    next_pos = ssspace.NextEntry(current_pos, next_direction)
    print("SplitSubSpace next position " + \
        str(next_pos))
    print("SplitSubSpace next entry" + str(ssspace.GetEntry(next_pos)))
    print()

@pytest.mark.parametrize("dim", [4, 6, 8])
def test_reordersubspace(dim):
    rsspace = space.ReorderSubSpace(dim)
    print("-------------- test_ReorderSubSpace ---------------")
    print(f"dim = {dim}")
    print("ReorderSubSpace all entries size = " + str(rsspace.size))
    print("ReorderSubSpace directions num = " + str(rsspace.num_directions))
    current_pos = random.randint(0, rsspace.size - 1)
    next_direction = rsspace.GetDirection(
        random.randint(0, rsspace.num_directions - 1))
    print("ReorderSubSpace Now position " + str(current_pos))
    print("ReorderSubSpace Now entry " + str(rsspace.GetEntry(current_pos)))
    print("ReorderSubSpace next direction " + str(next_direction))
    next_pos = rsspace.NextEntry(current_pos, next_direction)
    print("ReorderSubSpace next position " + \
        str(next_pos))
    print("ReorderSubSpace next entry" + str(rsspace.GetEntry(next_pos)))
    print()

@pytest.mark.parametrize("dim", [4, 6, 8])
def test_formatmodesubspace(dim):
    fmsspace = space.FModeSubSpace(dim)
    print("-------------- test_FModeSubSpace ---------------")
    print(f"dim = {dim}")
    print("FModeSubSpace all entries size = " + str(fmsspace.size))
    print("FModeSubSpace directions num = " + str(fmsspace.num_directions))
    current_pos = random.randint(0, fmsspace.size - 1)
    next_direction = fmsspace.GetDirection(
        random.randint(0, fmsspace.num_directions - 1))
    print("FModeSubSpace Now position " + str(current_pos))
    print("FModeSubSpace Now entry " + str(fmsspace.GetEntry(current_pos)))
    print("FModeSubSpace next direction " + str(next_direction))
    next_pos = fmsspace.NextEntry(current_pos, next_direction)
    print("FModeSubSpace next position " + \
        str(next_pos))
    print("FModeSubSpace next entry" + str(fmsspace.GetEntry(next_pos)))
    print()

@pytest.mark.parametrize("dim", [2, 3])
def test_parallelsubspace(dim):
    psspace = space.ParallelSubspace(dim)
    print("-------------- test_ParallelSubspace ---------------")
    print("ParallelSubspace all entries size = " + str(psspace.size))
    print("ParallelSubspace directions num = " + str(psspace.num_directions))
    current_pos = random.randint(0, psspace.size - 1)
    next_direction = psspace.GetDirection(
        random.randint(0, psspace.num_directions - 1))
    print("ParallelSubspace Now position " + str(current_pos))
    print("ParallelSubspace Now entry " + str(psspace.GetEntry(current_pos)))
    print("ParallelSubspace next direction " + str(next_direction))
    next_pos = psspace.NextEntry(current_pos, next_direction)
    print("ParallelSubspace next position " + \
        str(next_pos))
    print("ParallelSubspace next entry" + str(psspace.GetEntry(next_pos)))
    print()

if __name__ == "__main__":
    test_splitsubspace(5978, 4, "mixing")
    test_reordersubspace(8)
    test_formatmodesubspace(4)
    test_parallelsubspace(3)
