from typing import *
from copy import deepcopy
import multiprocessing
import torch

from .tensor import Value, Tensor, ComputeTensor, FindTopoSort
from .format import Axis, Format, ModeType
from . import format
from .utils import GetAlphabet26BaseNumber

class Schedule(object):
    """Define schedules."""
    compute_tensor: ComputeTensor
    origin_input_tensors: List['Value']
    all_tensors_bk: List["Value"] # Contain left and right tensor handle.
    
    fsplit_record: Dict[str, Tuple] # Such as {'i1': ['i10', 'i11', 32, 8]}
    lsplit_record: Dict[str, Tuple] # Such as {'i1': ['i10', 'i11', 32, 8]}
    lreordered_vars: List[str]
    parallel_var: str
    vectorize_var: str
    unroll_args: Tuple[str, int]
    thread_num: int
    prechunk: int

    # Is this entering the stage of modifying the loop schedule, 
    # having completed the format modification?
    flag_in_loop_schedule: bool

    def __init__(self, compute_tensor: ComputeTensor) -> None:
        self.compute_tensor = compute_tensor
        self.origin_input_tensors = compute_tensor.origin_inputs_list
        self.all_tensors_bk = deepcopy(self.origin_input_tensors)
        self.all_tensors_bk.append(deepcopy(self.compute_tensor))

        self.fsplit_record = dict()
        self.lsplit_record = dict()
        self.lreordered_vars = []
        self.parallel_var = None
        self.vectorize_var = None
        self.unroll_args = (None, 0)
        self.thread_num = multiprocessing.cpu_count() # default
        self.parchunk = 64 # default
        
        self.flag_in_loop_schedule = False # Is 

        # Set tesnor name for every tensor
        self.tensor_name_lst = []
        for i in range(len(self.all_tensors_bk)):
            self.tensor_name_lst.append(GetAlphabet26BaseNumber(i, is_upper=True))

    @property
    def all_axes(self) -> Dict[str, List["Axis"]]:
        """Axis name point to Axis object in all_tensors_bk and it don't
        contain loop splited axis
        """
        all_axes = dict()
        for tensor in self.all_tensors_bk:
            for axis in tensor.format.axes:
                if all_axes.get(axis.name, None) == None:
                    all_axes[axis.name] = [axis]
                else:
                    all_axes[axis.name].append(axis)
        return all_axes
    
    @property
    def origin_axes(self) -> Dict[str, List["Axis"]]:
        """Get all axes in origin axes, which don't contain splited axes."""
        all_axes = dict()
        for tensor in self.origin_input_tensors + [self.compute_tensor]:
            for axis in tensor.format.axes:
                if all_axes.get(axis.name, None) == None:
                    all_axes[axis.name] = [axis]
                else:
                    all_axes[axis.name].append(axis)
        return all_axes
    
    @property
    def spatial_axes(self) -> Dict[str, "Axis"]:
        """Get all spatial axes in origin axes, which don't contain splited axes."""
        axes = dict()
        for axis in self.compute_tensor.format.axes:
            axes[axis.name] = axis
        return axes
    
    @property
    def reduce_axes(self) -> Dict[str, "Axis"]:
        """Get all reduce axes in origin axes, which don't contain splited axes."""
        spatial_axes_name_lst = self.spatial_axes.keys()
        axes = dict()
        for tensor in self.origin_input_tensors:
            for axis in tensor.format.axes:
                if axis.name not in spatial_axes_name_lst:
                    axes[axis.name] = axis
        return axes


    def GetAllAxeAfterSplited(self) -> Tuple[List[str], List[int]]:
        """Get all the axis name, which contain loop splited axis."""
        all_axes = self.all_axes
        all_axes_size = []
        for axes in all_axes.values():
            all_axes_size.append(axes[0].size)
        all_axes = list(all_axes.keys())
        for axis, lst in self.lsplit_record.items():
            new_axes_lst = lst[: int(len(lst) / 2)]
            new_axes_size_lst = lst[int(len(lst) / 2):]
            idx = all_axes.index(axis)
            all_axes.remove(axis)
            all_axes.extend(new_axes_lst)
            all_axes_size.pop(idx)
            all_axes_size.extend(new_axes_size_lst)
        return all_axes, all_axes_size

    def _HaveTensor(self, tensor: Value):
        ret_tensor = None
        if tensor is self.compute_tensor:
            ret_tensor = self.all_tensors_bk[-1]
        else:
            for idx, item in enumerate(self.origin_input_tensors):
                if tensor is item:
                    ret_tensor = self.all_tensors_bk[idx]
            for item in self.all_tensors_bk:
                if tensor is item:
                    ret_tensor = item
        return ret_tensor
    
    def _HaveAxisInTensor(self, tensor: Value, axis_name: str):
        if axis_name in tensor.format.axes_name.keys():
            return True
        return False


    def FormatSplit(self, axis_name: str, axes_size_lst: List):
        """Split a axis of format in sparse tensor. Notice only using 
        once for every axis of origin tensor.

        Parameters
        ----------
        axis_name: str
            The axis name in tensor.format.axes
        axes_size_lst: List
            New axes size list, and length of list is splited axis count.
        
        Return
        ------
        new_axes_names: List[str] or None
            If None indicate loop spliting operation occur error, else
            return new axes name list.
        """
        if self.flag_in_loop_schedule:
            print("[Warning][AutoSparse.Schedule] Can't change split format when"
                  " during loop shcedules modifacation phase. ")
            return None

        if axis_name in self.fsplit_record.keys():
            print(
                f"[Warning][AutoSparse.Schedule] '{axis_name}' of FormatSplit "
                "already have split record, so that there can't add new record."
            )
            return None
        
        related_tensor_lst = []
        for tensor in self.all_tensors_bk:
            if self._HaveAxisInTensor(tensor, axis_name):
                related_tensor_lst.append(tensor) 
        if len(related_tensor_lst) == 0:
            print(
                f"[Warning][AutoSparse.Schedule] Can't find axis "
                f"`{axis_name}` in FormatSplit."
            )
            return None

        total_size = 1
        for item in axes_size_lst:
            total_size *= item
        if total_size != related_tensor_lst[0].format.axes_name[axis_name].size:
            print(
                f"[Warning][AutoSparse.Schedule] Accumulate multiplies of split "
                "axes must equal with origin axis size in FormatSplit."
            )
            return None
        
        # Change all the related tensor's format.
        new_axes_names = [axis_name + str(i) for i in range(len(axes_size_lst))]
        for tensor in related_tensor_lst:
            new_axes = []
            for i in range(len(axes_size_lst)):
                new_axes.append(Axis(axes_size_lst[i], name = new_axes_names[i]))
            axes_name = [] # The name of all axes
            for item in tensor.format.axes:
                axes_name.append(item.name)
            axis_idx = axes_name.index(axis_name)
            new_format_axes = list(tensor.format.axes[:axis_idx]) + new_axes + \
                                list(tensor.format.axes[axis_idx + 1:])
            new_format_order = []
            for i in range(len(tensor.format.order)):
                if i == axis_idx:
                    new_format_order.extend(
                        [tensor.format.order[axis_idx] + j 
                        for j in range(len(axes_size_lst))]
                    )
                elif tensor.format.order[i] > tensor.format.order[axis_idx]:
                    new_format_order.append(
                        tensor.format.order[i] + len(axes_size_lst) - 1
                    )
                else:
                    new_format_order.append(tensor.format.order[i])
            tensor.format = Format(tuple(new_format_axes), tuple(new_format_order))
        
        # Record to fsplit_record
        value = tuple(new_axes_names + axes_size_lst)
        self.fsplit_record[axis_name] = value

        return new_axes_names
    
    def FormatReorder(self, tensor: Value, axes_name_lst: List[str]):
        """Reorder storage format axis

        Parameters
        ----------
        tensor: Value
            Which tensor in this computational graph of `ComputeTensor` need FS.
            Notice, this tensor can be original input tenor in `ComputeTensor`,
            or it can be entry in the changed `all_inputs_tensor`.
        axes_name_lst: List[str]
            The order list.
        
        Return
        ------
        ret_tensor: Value
            Changed Value. If this is None, there is an error where the 
            tensor is not found.
        """
        ret_tensor = self._HaveTensor(tensor)
        if ret_tensor == None:
            print("[Warning][AutoSparse.Schedule] Can't find tensor in FormatReorder.")
            return None

        if len(axes_name_lst) != len(ret_tensor.format.shape):
            print(
                f"[Warning][AutoSparse.Schedule] `axes_name_lst` of FormatMode"
                f" must conatin all the axes."
            )
        
        for axis_name in axes_name_lst:
            if self._HaveAxisInTensor(ret_tensor, axis_name) == False:
                print(
                f"[Warning][AutoSparse.Schedule] Can't find axis "
                f"`{axis_name}` in FormatReorder."
                )
                return None

        new_order = []
        for idx, axis in enumerate(ret_tensor.format.axes):
            idx = axes_name_lst.index(axis.name)
            new_order.append(idx)

        ret_tensor.format.order = tuple(new_order)

        return ret_tensor


    def FormatMode(self, tensor: Value, axis_name: str, mode: Union[ModeType, int]):
        """Change storage format axis mode.

        Parameters
        ----------
        tensor: Value
            Which tensor in this computational graph of `ComputeTensor` need FS.
            Notice, this tensor can be original input tenor in `ComputeTensor`,
            or it can be entry in the changed `all_inputs_tensor`.
        axis_name: str
            Which axis in tensor need to change mode.
        mode: ModeType
        
        Return
        ------
        ret_tensor: Value
            Changed Value. If this is None, there is an error where the 
            tensor is not found.
        """
        ret_tensor = self._HaveTensor(tensor)
        if ret_tensor == None:
            print("[Warning][AutoSparse.Schedule] Can't find tensor in "
                  "FormatMode.")
            return None
        
        if self._HaveAxisInTensor(ret_tensor, axis_name) == False:
            print(
                f"[Warning][AutoSparse.Schedule] Can't find axis "
                f"`{axis_name}` in FormatMode."
            )
            return None
        
        if isinstance(mode, int):
            assert mode < len(format.FormatMode)
            mode = format.FormatMode[mode]

        ret_tensor.format.axes_name[axis_name].mode = mode
        return ret_tensor
    
    def LoopSplit(self, axis_name: str, axes_size_lst: List[int]):
        """Split loop axism, and notice only using once for every axis of 
        fsplited or origin tensor.

        Parameters
        ----------
        axis_name: str
            The axis name in computation.
        axes_size_lst: List
            New axes size list, and length of list is splited axis count.
        
        Return
        ------
        new_axes_name_lst: List[str] or None
            If None indicate loop spliting operation occur error, else
            return new axes name list.
        """
        all_axes = self.all_axes
        if axis_name not in all_axes.keys():
            print(
                f"[Warning][AutoSparse.Schedule] Can't find axis "
                f"`{axis_name}` in LoopSplit."
            )
            return None

        if axis_name in self.lsplit_record.keys():
            print(
                f"[Warning][AutoSparse.Schedule] '{axis_name}' of LoopSplit "
                "already have split record, so that there can't add new record."
            )
            return None
        
        total_size = 1
        for item in axes_size_lst:
            total_size *= item
        if total_size != all_axes[axis_name][0].size:
            print(
                f"[Warning][AutoSparse.Schedule] Accumulate multiplies of split "
                "axes must equal with origin axis size in LoopSplit."
            )
            return None
        
        new_axes_name_lst = [
            axis_name + str(i) for i in range(len(axes_size_lst))
        ]
        value = tuple(new_axes_name_lst + axes_size_lst)
        self.lsplit_record[axis_name] = value

        self.flag_in_loop_schedule = True
        return new_axes_name_lst
        
    def LoopReorder(self, axes_name_lst: List[str]):
        """Reorder loop axes
        Parameter
        ---------
        axes_name_lst: List[str]
            The order list. Notice the list must correspond with 
        
        Return
        ------
        excute_ok: bool
            False indicate there have some error, and True mean reorder
            loop successfully.
        """
        all_axes_name, _ = self.GetAllAxeAfterSplited()
        if len(axes_name_lst) != len(all_axes_name):
            print(
                f"[Warning][AutoSparse.Schedule] `axes_name_lst` of "
                "LoopReorder must contain all axis."
            )
            return False
        for axis_name in axes_name_lst:
            if axis_name not in all_axes_name:
                print(
                    f"[Warning][AutoSparse.Schedule] '{axis_name}' of "
                    "LoopReorder does not appear in all axes record."
                )
                return False

        self.lreordered_vars = axes_name_lst
        self.flag_in_loop_schedule = True
        return True
    
    def LoopVectorize(self, axis_name: str):
        """Vectorize loop axis.

        Parameters
        ----------
        axis_name: str
            The axis name in computation.
        
        Return
        ------
        excute_ok: bool
            False indicate there have some error, and True mean reorder
            loop successfully.
        """
        all_axes, _ = self.GetAllAxeAfterSplited()
        if axis_name not in all_axes:
            print(
                f"[Warning][AutoSparse.Schedule] Can't find axis "
                f"`{axis_name}` in LoopUnroll."
            )
            return False
        
        if axis_name == self.parallel_var or axis_name == self.unroll_args[0]:
            print(
                f"[Warning][AutoSparse.Schedule] Axis can't same with parallized"
                f"` or vectorized axis."
            )
            return False

        self.vectorize_var = axis_name

        self.flag_in_loop_schedule = True
        return True
    
    def LoopParallel(self, axis_name: str):
        """Parallelize loop axis
        
        Parameters
        ----------
        axis_name: str
            The axis name in computation.
        
        Return
        ------
        excute_ok: bool
            False indicate there have some error, and True mean reorder
            loop successfully.
        """
        all_axes, _ = self.GetAllAxeAfterSplited()
        if axis_name not in all_axes:
            print(
                f"[Warning][AutoSparse.Schedule] Can't find axis "
                f"`{axis_name}` in LoopUnroll."
            )
            return False
        
        if axis_name == self.vectorize_var or axis_name == self.unroll_args[0]:
            print(
                f"[Warning][AutoSparse.Schedule] Axis can't same with parallized"
                f"` or vectorized axis."
            )
            return False

        self.parallel_var = axis_name

        self.flag_in_loop_schedule = True
        return True
        
    def LoopUnroll(self, axis_name: str, factor: int):
        """Unroll loop axis with factor
        
        Parameters
        ----------
        axis_name: str
            The axis name in computation.
        factor: int
            unroll factor, which must be less than the length of the axis
        
        Return
        ------
        excute_ok: bool
            False indicate there have some error, and True mean reorder
            loop successfully.
        """
        all_axes, all_axes_size = self.GetAllAxeAfterSplited()
        if axis_name not in all_axes:
            print(
                f"[Warning][AutoSparse.Schedule] Can't find axis "
                f"`{axis_name}` in LoopUnroll."
            )
            return False
        
        if axis_name == self.parallel_var or axis_name == self.vectorize_var:
            print(
                f"[Warning][AutoSparse.Schedule] Axis can't same with parallized"
                f"` or vectorized axis."
            )
            return False

        axis_size = all_axes_size[all_axes.index(axis_name)]
        if factor > axis_size:
            print(
                f"[Warning][AutoSparse.Schedule] Unroll factor {factor} "
                f"can't be more than axis size {axis_size}"
            )
            return False

        self.unroll_args = (axis_name, factor)

        self.flag_in_loop_schedule = True
        return True

    def SetThreadNum(self, num):
        """Set OpenMP thread number argument."""
        self.thread_num = num

    def SetParallelChunk(self, num):
        """Set OpenMP parallel chunk size"""
        self.parchunk = num

    def GetScheduleName(self):
        """"""
        pure_comp_desc = []
        # Tensor describe
        tensor_count = len(self.origin_input_tensors) + 1
        pure_comp_desc.append(str(tensor_count))
        for idx, tensor in enumerate(self.origin_input_tensors + [self.compute_tensor]):
            pure_comp_desc.append(self.tensor_name_lst[idx])
            pure_comp_desc.append(str(int(tensor.is_sparse)))
            pure_comp_desc.append(str(len(tensor.shape)))
            for axis in tensor.format.axes:
                pure_comp_desc.append(axis.name)
                pure_comp_desc.append(str(axis.size))
                pure_comp_desc.append(str(axis.mode.value))
        return '_'.join(pure_comp_desc)

    def GenConfigCommand(self):
        """Describe computation einsum expression of origin tensor.
        Describe Schedule and format information.
        Return
        ------
        (pure_comp_desc, schedules)
            ("tensor_count, tensor_list[tensor_name, is_sparse, axes_count, 
            axes_list[axis_name, size, mode]],dtype", 
            "fsplit_count, 
            fsplit_list[axis_name, new_axes_count, new_axes_list[axis_name, size]],
            sparse_tensor_count, 
            freorder_list[tensor_name, axes_size, axes_list[axis_name]], 
            fmode_list[tensor_name, axes_size, axes_list[axis_name, fmode]],
            lsplit_count, 
            lsplit_list[axis_name, new_axes_count, new_axes_list[axis_name, size]],
            lreorder_vars_count, lreorder_list[axis_name]
            parallize_axis_name, vectorize_axis_name, unroll_axis_name, unroll_factor, 
            thread_num, parchunk")
            Notice: If the item of 'x_count' is 0, there will no next item of 
                'x_list'. 'dtype' can't be none. 'parallize' and 'vectorize'
                'unroll' item will fill None string if there have not action. 
                 And 'unroll_factor' will be ignored if 'unroll' is None.
        """
        pure_comp_desc = []
        schedules = []
        # Tensor describe
        tensor_count = len(self.origin_input_tensors) + 1
        pure_comp_desc.append(str(tensor_count))
        for idx, tensor in enumerate(self.origin_input_tensors + [self.compute_tensor]):
            pure_comp_desc.append(self.tensor_name_lst[idx])
            pure_comp_desc.append(str(int(tensor.is_sparse)))
            pure_comp_desc.append(str(len(tensor.shape)))
            for axis in tensor.format.axes:
                pure_comp_desc.append(axis.name)
                pure_comp_desc.append(str(axis.size))
                pure_comp_desc.append(str(axis.mode.value))
        
        # dtype
        pure_comp_desc.append(self.compute_tensor.dtype)

        # fsplit
        fsplit_count = len(self.fsplit_record)
        schedules.append(str(fsplit_count))
        for key, value in self.fsplit_record.items():
            schedules.append(key) # axis_name
            new_axes_count = int(len(value) / 2)
            assert new_axes_count * 2 == len(value)
            schedules.append(str(new_axes_count))
            for i in range(new_axes_count):
                schedules.append(value[i])
                schedules.append(str(value[i + new_axes_count]))
        
        # Notice maybe origin tensor have format config
        # freorder and fmode
        sparse_tensor_count = 0
        freorder_lst = []
        fmode_lst = []
        for idx, tensor in enumerate(self.all_tensors_bk):
            if tensor.is_sparse:
                sparse_tensor_count += 1
                # freorder
                freorder_lst.append(self.tensor_name_lst[idx]) # tensor name
                freorder_lst.append(str(len(tensor.shape))) # tensor dimentions
                reordered_vars = ["" for i in range(len(tensor.format.order))]
                for axis_idx, reordered_idx in enumerate(tensor.format.order):
                    reordered_vars[reordered_idx] = tensor.format.axes[axis_idx].name
                freorder_lst.extend(reordered_vars)
                # fmode
                fmode_lst.append(self.tensor_name_lst[idx]) # tensor name
                fmode_lst.append(str(len(tensor.shape))) # tensor dimentions
                for axis in tensor.format.axes:
                    fmode_lst.append(axis.name)
                    fmode_lst.append(str(axis.mode.value))
        schedules.append(str(sparse_tensor_count))
        schedules.extend(freorder_lst)
        schedules.extend(fmode_lst)

        # lsplit
        lsplit_count = len(self.lsplit_record)
        schedules.append(str(lsplit_count))
        for key, value in self.lsplit_record.items():
            schedules.append(key) # axis_name
            new_axes_count = int(len(value) / 2)
            assert new_axes_count * 2 == len(value)
            schedules.append(str(new_axes_count))
            for i in range(new_axes_count):
                schedules.append(value[i])
                schedules.append(str(value[i + new_axes_count]))
        
        # lreordered_vars
        schedules.append(str(len(self.lreordered_vars)))
        schedules.extend(self.lreordered_vars)

        # parallel
        schedules.append(str(self.parallel_var))
        # vectorize
        schedules.append(str(self.vectorize_var))
        # unroll
        schedules.append(str(self.unroll_args[0]))
        schedules.append(str(self.unroll_args[1]))
        # OpenMP thread and chunk
        schedules.append(str(self.thread_num))
        schedules.append(str(self.parchunk))
    
        return (' '.join(pure_comp_desc), ' '.join(schedules))

    def SaveScheduleConfigCommand(self, filepath):
        # Save all the tensor format order.
        pure_comp_desc, schedules = self.GenConfigCommand()
        params = {
            "pure_comp_desc" : pure_comp_desc,
            "schedules" : schedules
        }
        torch.save(params, filepath)
    
    def LoadScheduleConfigCommand(self, filepath):
        checkpoint = torch.load(self.filepath)
        pure_comp_desc = checkpoint['pure_comp_desc']
        schedules = checkpoint['schedules']
        return pure_comp_desc, schedules


def CreateSchedule(ctensor: ComputeTensor):
    return Schedule(ctensor)