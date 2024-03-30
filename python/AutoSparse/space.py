""" Define schedule space."""
import numpy as np
import copy
import random

from .utils import *
import itertools

class SubSpace(object):
    def __init__(self) -> None:
        self.dim = 0 # space dimension
        self.all_entries = [] # All the entries in space
        self.all_entries_dict = dict()
        self.directions = [] # Expansion direction in space
        self.type_key = ""

    @property
    def size(self):
        if (len(self.all_entries) == 0):
            return 1
        return len(self.all_entries)
    
    @property
    def num_directions(self):
        return len(self.directions)

    def __len__(self):
        return self.size
    
    def RandomEntry(self):
        """Random select a entry"""
        return np.random.choice(self.all_entries)

    def NextEntry(self, *args, **kwargs):
        """ Get next entry by direction argument in space. """
        raise NotImplementedError()
    
    def GetEntry(self, pos: int):
        """ Get the entry in the space position. """
        return self.all_entries[pos]
    
    def GetDirection(self, dir_idx):
        """ Get direction from deirections list. """
        return self.directions[dir_idx % self.num_directions]

    def GetBatchEntry(self, batch_indices):
        ret_entities = []
        for index in batch_indices:
            ret_entities.append(self.GetEntry(index))
        return ret_entities


class SplitSubSpace(SubSpace):
    """The sub space contain format split and loop split."""
    def __init__(self, axis_size: int, dim: int = 2, policy: str = "power2"):
        """
        Parameters
        ----------
        axis_size: int
            The origin axis size, which need to be splited.
        dim: int optional(2)
            New axes count splited from axis.
        policy: str optinal("power2")
            Factorization policy, there have "power2" (only factorize to number
            based 2), "factorization", and "mixing" (contain 2 method).
        """
        super(SplitSubSpace, self).__init__()
        assert dim > 0
        self.axis_size = axis_size
        self.dim = dim
        self.all_entries = SplitWithFactorization(axis_size, dim, policy)
        self.all_entries_dict = {tuple(val): idx for idx, val in enumerate(self.all_entries)}
        # 3D change
        # i, j indicate selecting two elements resplit, and last item 
        # indicate wheather exchange two elements
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    self.directions.append((i, j, 0))
                    self.directions.append((i, j, 1))
        self.policy = policy
        self.type_key = "split"

    def NextEntry(self, pos: int, direction: Union[int, Tuple]):
        """
        Return
        ------
        next_pos: int
            The next entry position in all entries.
        """
        if (isinstance(direction, int)):
            direction = [direction]
        if len(direction) == 1: # 1D
            return (pos + direction[0]) % self.size
        elif (len(direction) == 2 or len(direction) == 3): # 2D or 3D
            first_pos, second_pos = direction[0], direction[1]
            assert 0 <= first_pos < self.dim and 0 <= second_pos < self.dim
            cur_entry = self.all_entries[pos]
            ret = copy.deepcopy(cur_entry)
            value = ret[first_pos] * ret[second_pos]
            next_pos = -1
            # while next_pos < 0:
            #     tmp = ret[first_pos] + 1
            #     while (tmp <= value):
            #         if self.policy == "power2" and IsPowerX(2, tmp):
            #             break
            #         elif self.policy == "factorization" and value % tmp == 0:
            #             break
            #         elif self.policy == "mixing" and \
            #             (IsPowerX(2, tmp) or value % tmp == 0):
            #             break 
            #         else:
            #             tmp += 1
            #     tmp = min(tmp, value)
            #     ret[first_pos] = tmp
            #     ret[second_pos] = math.ceil(value / tmp)
            #     if (len(direction) == 3 and direction[2]):
            #         ret[first_pos], ret[second_pos] = ret[second_pos], ret[first_pos]
            #     try:
            #         next_pos = self.all_entries.index(ret)
            #     except ValueError:
            #         next_pos = -1

            # Reorganization
            candidate_list = SplitWithFactorization(value, 2, self.policy)
            sorted_candidate_list = sorted(candidate_list)
            for item in sorted_candidate_list[::-1]:
                if item[0] > ret[first_pos]: # If all less it, will not change
                    ret[first_pos]  = item[0]
                    ret[second_pos] = item[1]
            if (len(direction) == 3 and direction[2]):
                ret[first_pos], ret[second_pos] = ret[second_pos], ret[first_pos]
            
            next_pos = self.all_entries_dict.get(tuple(ret), -1)

            assert (next_pos >= 0)
            return next_pos
        else:
            raise NotImplementedError(
                f"Not support more than 3 dimensions direction: {direction}."
            )

class ReorderSubSpace(SubSpace):
    def __init__(self, dim: int) -> None:
        """
        Reorder contain format and loop schedules. There will shuffle
        all axis order.
        Parameter
        ---------
        dim: int
            The axes number need to reorder.
        """
        super(ReorderSubSpace, self).__init__()
        assert dim >= 2
        self.dim = dim
        self.all_entries = Permute([i for i in range(dim)])
        self.all_entries_dict = {tuple(val): idx for idx, val in enumerate(self.all_entries)}
        # 2D indicate selecting id i and j of axes to exchange position.
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                self.directions.append((i, j))
        self.type_key = "reorder"

    def NextEntry(self, pos: int, direction: Tuple):
        """
        Return
        ------
        next_pos: int
            The next entry position in all entries.
        """
        assert isinstance(direction, tuple) and len(direction) == 2
        ret = copy.deepcopy(self.all_entries[pos])
        first_idx = ret.index(direction[0])
        second_idx = ret.index(direction[1])
        ret[first_idx], ret[second_idx] = ret[second_idx], ret[first_idx]
        next_pos = self.all_entries_dict.get(tuple(ret), -1)
        assert next_pos >= 0
        return next_pos

class FModeSubSpace(SubSpace):
    def __init__(self, dim) -> None:
        super(FModeSubSpace, self).__init__()
        assert dim > 0
        self.dim = dim
        possible_mode_values = range(5)
        self.all_entries = list(itertools.product(possible_mode_values, repeat=dim))
        self.all_entries_dict = {tuple(val): idx for idx, val in enumerate(self.all_entries)}
        self.directions = list(itertools.product(range(2), repeat=dim))
        self.type_key = "format_mode"
    
    def NextEntry(self, pos: int, direction: Tuple):
        """
        Return
        ------
        next_pos: int
            The next entry position in all entries.
        """
        assert len(direction) == len(self.all_entries[pos]), \
            "[AutoSparse.Space][Error] direction shape dimensions error."
        ret = list(copy.deepcopy(self.all_entries[pos]))
        for i in range(len(direction)):
            ret[i] = (ret[i] + direction[i]) % 5
        next_pos = self.all_entries_dict.get(tuple(ret), -1)
        assert next_pos >= 0
        return next_pos

class ParallelSubspace(SubSpace):
    """Contain parallel, vectorization, unroll sub space."""
    def __init__(self, dim) -> None:
        super(ParallelSubspace, self).__init__()
        assert dim > 0
        self.dim = dim
        possible_mode_values = range(2)
        self.all_entries = list(itertools.product(possible_mode_values, repeat=self.dim))
        self.all_entries_dict = {tuple(val): idx for idx, val in enumerate(self.all_entries)}
        self.directions = list(itertools.product(possible_mode_values, repeat=self.dim))
        self.type_key = "parallel"
    
    def NextEntry(self, pos: int, direction: Tuple):
        """
        Return
        ------
        next_pos: int
            The next entry position in all entries.
        """
        assert len(direction) == len(self.all_entries[pos]), \
            "[AutoSparse.Space][Error] direction shape dimensions error."
        ret = list(copy.deepcopy(self.all_entries[pos]))
        for i in range(len(direction)):
            ret[i] = (ret[i] + direction[i]) % 2
        next_pos = self.all_entries_dict.get(tuple(ret), -1)
        assert next_pos >= 0
        return next_pos
    
class Space(object):
    def __init__(self) -> None:
        self.subspaces = {} # all the subspace object, name: object
        self.valid_type_keys = [
            "parallel", "format_mode", "split", "reorder"
        ]
        # Record all the subspace with it's schedule type 
        self.types = {key: [] for key in self.valid_type_keys}
        self.dim = 0 # Toatal space dimension
    
    def add_subspace(self, name: str, subspace: SubSpace, type_key):
        """
        Parameters
        ----------
        name str
            subspace name
        subspace SubSpace
            Object for SubSpace
        type_key 
            space type name in ["parallel", "format_mode", "split", "reorder"]
        """
        assert name in self.subspaces, \
            f"[AutoSparce.Space][Error] The space already exist the subspace {name}."
        assert type_key in self.valid_type_keys, \
            f"[AutoSparce.Space][Error] Type key is invalid for {type_key}."
        self.subspaces[name] = subspace
        self.types[type_key].append(name)
        self.dim += subspace.dim

    def items(self):
        """Return all the subspace name and object"""
        return self.subspaces.items()

    def __len__(self):
        """Get the total space size, which indicate all the point count in space."""
        sz = 1
        for _, subspace in self.items():
            sz *= len(subspace)
        return sz
