import numpy as np
import copy
from typing import *

import AutoSparse
from .format import Axis, Format

DtypeSet = [
    "bool", "uint8", "uint16", 
    "uint32", "uint64", "uchar", "ushort", "uint", "ulong", 
    "ulonglong", "int8", "int16", "int32", "int64", "char", 
    "short", "int", "long", "longlong", "float", "double", 
    "complexfloat", "complexdouble"
]          

class Operator(object):
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any):
        """ Construct ComputeTensor. """
        dtype_set = set()
        for item in args:
            dtype_set.add(item.dtype)
        assert len(dtype_set) == 1, \
            "[AutoSparse.Operator] Operands have inconsistent data types."
        format, inputs, dtype = self.Compute(*args)
        return ComputeTensor.make_from_op(self, inputs, format, dtype)
    
    def Compute(self, *args: Tuple["Value"]) -> Tuple[Union[Format, Tuple], List["Value"], str]:
        """ Extract ComputeTensor info by input value. """
        raise NotImplementedError()
    
    def __str__(self):
        """ Get einsum expression. """
        return ""
    

class Value(object):
    """ A value in computation graph """
    op: Optional[Operator] # Need to get EinsumExpr from op
    inputs: List["Value"] # trace of computation graph
    cached_data: Union[np.ndarray, str] # ndarray of dense or CSR of sparse
    dtype : str 
    format : Format
    is_sparse : bool

    def _init(
        self,
        format: Union[Format, Tuple["Axis"]],
        op = None,
        inputs = [],
        cached_data = None,
        dtype = "float",
        is_sparse = False
    ) -> None:
        """
        Parameters
        ----------
        format: Format or Tuple["Axis"]
        """
        assert dtype in DtypeSet, \
                "[AutoSparse.Value] dtype must be one of these: " + str(DtypeSet)
        if isinstance(format, Format):
            self.format = format
        elif isinstance(format, Tuple):
            self.format = Format(format)
        else:
            assert False, \
                "[AutoSparse.Value] format data type dosen't meet the requirements."
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.dtype = dtype
        self.is_sparse = is_sparse
    
    def __str__(self) -> str:
        return "AutoSparse.Value(" + str(self.format)[11:] + ", dtype = " + \
            self.dtype + ", is_saprse = " + str(self.is_sparse) + ")"
        
    def is_tensor(self):
        return self.op is None
    
    def is_compute_tensor(self):
        return self.op is not None
    
    @property
    def shape(self):
        return self.format.shape

    @property
    def data(self):
        return self.cached_data

    @data.setter
    def data(self, value):
        raise NotImplementedError()
    
    def LoadData(self, CSR_filepath, dataset_type = '2D'):
        """ Init cached_data from other tensor or numpy or local file. """
        if dataset_type == '2D':
            num_row, num_col, num_nonezero = np.fromfile(
            CSR_filepath, count=3, dtype = '<i4'
            )
            assert (num_row, num_col) == self.shape, \
                f"[AutoSparse.LoadData] Shape of csr file data is different from {self.shape}."
            self.cached_data = CSR_filepath
        else:
            assert False,\
                f"[AutoSparse.LoadData] Pnly support 2D data load in now."
            raise NotImplementedError()
    
    def __add__(self, other):
        if isinstance(other, Value):
            return AutoSparse.ops.EWiseAdd()(self, other)
        else:
            return AutoSparse.ops.ScalarAdd(other)(self)

    def __mul__(self, other):
        if isinstance(other, Value):
            return AutoSparse.ops.EWiseMul()(self, other)
        else:
            return AutoSparse.ops.ScalarMul(other)(self)

    def __sub__(self, other):
        if isinstance(other, Value):
            return AutoSparse.ops.EWiseSub()(self, other)
        else:
            return AutoSparse.ops.ScalarSub(other)(self)

    def __truediv__(self, other):
        if isinstance(other, Value):
            return AutoSparse.ops.EWiseDiv()(self, other)
        else:
            return AutoSparse.ops.ScalarDiv(other)(self)

    def __matmul__(self, other):
        return AutoSparse.ops.Matmul()(self, other)

    def sum(self, axes = None):
        return AutoSparse.ops.Summation(axes)(self)

    def __neg__(self):
        return AutoSparse.ops.Negate()(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__



class Tensor(Value):
    """ Define tensor. """
    def __init__(
        self, 
        format: Union[Format, Tuple["Axis"]],
        *,
        is_sparse = False,
        dtype = "float",
        **kwargs
    ) -> None:
        self._init(
            format,
            None,
            [], 
            dtype = dtype,
            is_sparse = is_sparse
        )
    
    def __str__(self) -> str:
        return "AutoSparse.Tensor(" + str(self.format)[11:] + ", dtype = " + \
            self.dtype + ", is_saprse = " + str(self.is_sparse) +  \
            ", data = " + str(self.cached_data) + ")"


class ComputeTensor(Value):
    """ The tensor is computed by operators from other Tensor. """

    def __init__(
        self,
        format: Union[Format, Tuple["Axis"]],
        op,
        inputs,
        *,
        is_sparse = False,
        dtype = "float",

    ) -> None:
        self._init(
            format,
            op,
            inputs,
            dtype = dtype,
            is_sparse = is_sparse
        )

    @property
    def origin_inputs_list(self):
        tensors = FindTopoSort(self)
        ret_lst = []
        for tensor in tensors:
            if isinstance(tensor, Tensor):
                ret_lst.append(tensor)
        return ret_lst

    @property
    def reduce_axes(self):
        ret_axes = {}
        tensors = FindTopoSort(self)
        origin_inputs_list = []
        left_axes = {axis.name: False for axis in self.format.axes}
        for tensor in tensors:
            if isinstance(tensor, Tensor):
                origin_inputs_list.append(tensor)
        for tensor in origin_inputs_list:
            for axis in tensor.format.axes:
                if axis.name in self.format.axes_name.keys():
                    left_axes[axis.name] = True
                else:
                    if ret_axes.get(axis.name, None) == None:
                        ret_axes[axis.name] = [copy.deepcopy(axis)]
                    else:
                        ret_axes[axis.name].append(copy.deepcopy(axis))
        for key, value in left_axes:
            if value == False:
                ret_axes[key] = [copy.deepcopy(self.format.axes_name[key])]
        return ret_axes


    @property
    def spatial_axes(self):
        ret_axes = {}
        tensors = FindTopoSort(self)
        origin_inputs_list = []
        for tensor in tensors:
            if isinstance(tensor, Tensor):
                origin_inputs_list.append(tensor)
        for tensor in origin_inputs_list:
            for axis in tensor.format.axes:
                if axis.name in self.format.axes_name.keys():
                    if ret_axes.get(axis.name, None) == None:
                        ret_axes[axis.name] = [copy.deepcopy(axis)]
                    else:
                        ret_axes[axis.name].append(copy.deepcopy(axis))
        return ret_axes
    
    def __str__(self) -> str:
        return "AutoSparse.ComputeTensor("+ str(self.format)[11:] + ", dtype = " + \
            self.dtype + ", is_saprse = " + str(self.is_sparse) + ", op = "\
            + str(self.op) + ", data = " + str(self.cached_data) + ")"
    
    @staticmethod
    def make_from_op(
        op: Operator, inputs: List["Value"], format: Format, dtype: str
    ):
        """ Construct ComputeTensor by operator. """
        tensor = ComputeTensor.__new__(ComputeTensor)
        tensor.__init__(format, op, inputs, dtype = dtype)
        return tensor

    def GetEinsumExpr(self):
        """ Get compute einsum expression by traversing computation graph. """
        raise NotImplementedError()


def ComputeHelp1(
    compute_tensor: ComputeTensor, 
    format: Union[Format, Tuple["Axis"]] = None,
    is_sparse = False
):
    format_flag = True
    if isinstance(format, Tuple):
        format = Format(format)
    if format != None:
        if len(format.shape) != len(compute_tensor.format.shape):
            format_flag = False
        else:
            for i in range(len(format.shape)):
                if format.axes[i].size != compute_tensor.format.axes[i].size:
                    format_flag = False
                    break
                elif format.axes[i].name != compute_tensor.format.axes[i].name:
                    format_flag = False
                else:
                    compute_tensor.format.axes[i] = format.axes[i]
    assert format_flag, \
        "[AutoSparse.Compute] input format don't match with ComputeTensor."

    compute_tensor.is_sparse = is_sparse
    return compute_tensor


def ComputeHelp2(
    op: Operator, 
    inputs: List, 
    format: Union[Format, Tuple["Axis"]] = None,
    is_sparse = False
):
    scalar_inputs = []
    ewise_inputs = []
    for item in inputs:
        if isinstance(item, Value):
            ewise_inputs.append(item)
        else:
            scalar_inputs.append(item)
    if len(scalar_inputs):
        return ComputeHelp1(op(*scalar_inputs)(*ewise_inputs), format, is_sparse)
    return ComputeHelp1(op()(*ewise_inputs), format, is_sparse)

def Compute(*args: Any, **kwds: Any):
    """Construct ComputeTensor by operator.
    Parameters
    ----------
    args : List of arg
        If size is 1: compute_tensor : ComputeTensor
        Elif size is 2: (op, inputs) : (Operator, List["Value"])
    kwds : Dict or default args
        format : optional(None) Union[Format, Tuple["Axis"]]
            The ComputeTensor format.
        is_sparse : optional(Flase) 
            Is or not the ComputeTensor sparse.
    """
    format = kwds.get("format", None)
    is_sparse = kwds.get("is_sparse", False)
    if (len(args) == 1):
        return ComputeHelp1(args[0], format, is_sparse)
    elif (len(args) == 2):
        return ComputeHelp2(args[0], args[1], format, is_sparse)
    else:
        assert False, \
            "[AutoSparse.Compute] Error input args."

def FindTopoSort(node_list: Union[List[Value], Value]) -> List[Value]:
    """
    Given a list of nodes, return a topological sort list of nodes with post-order.
    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS.
    """
    if isinstance(node_list, Value):
        node_list = [node_list]
    visited = set()
    topo_order = list()
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node: Value, visited: set, topo_order: List):
    """Post-order DFS"""
    # First check is visited.
    if node in visited:
        return

    visited.add(node)
    # Recursive depth traversal for each inputs node.
    for i in node.inputs:
        if i not in visited:
            topo_sort_dfs(i, visited, topo_order)
    
    # Add current node after all the inputs node added.
    topo_order.append(node)