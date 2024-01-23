import random
import os, sys
import math
import itertools
import numpy as np
import multiprocessing
from functools import partial

current_filepath = os.path.abspath(__file__)
work_dir = os.path.dirname(os.path.dirname(current_filepath))
sys.argv = [current_filepath,
            os.path.join(work_dir, 'dataset', 'total.txt')]

def check_mode(freordered_vars : list, vars_mode : dict, dimensions : dict):
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

def test_check_mode():
    freordered_vars = ['i0', 'k0', 'k1', 'i1']
    vars_mode = {}
    dimensions = {}
    vars_mode['i0'] = 4
    vars_mode['k0'] = 2
    vars_mode['k1'] = 1
    vars_mode['i1'] = 2
    dimensions['i0'] = 1
    dimensions['k0'] = 1
    dimensions['k1'] = 64
    dimensions['i1'] = 128
    a = check_mode(freordered_vars, vars_mode, dimensions)
    print(a)
test_check_mode()

def check_lreorder(vars_mode : dict, freordered_vars_lsplit : list, lreordered_vars : list):
    # The order between sparse array axes is fixed.
    for idx in range(1, len(freordered_vars_lsplit)):
        item0 = freordered_vars_lsplit[idx - 1]
        item1 = freordered_vars_lsplit[idx]
        item0_index = lreordered_vars.index(item0)
        item1_index = lreordered_vars.index(item1)
        if item0_index > item1_index:
            return False
    return True

def random_generate_config(mtx_name, autosparse_prefix):
    mtx_name += ".csr"
    mtx_filepath = os.path.join(
        autosparse_prefix, "dataset", "suitsparse", mtx_name
    )
    # 从文件读3个数据，数据的类型为小端法（<）的4字节整数（i4）
    num_row, num_col, num_nonezero = np.fromfile(
        mtx_filepath, count=3, dtype = '<i4'
    )
    configs = set()
    while(len(configs) <= 100000):
        dimensions = {}
        ### Set format
        i_fsplit = random.choice([1<<p for p in range(int(math.log(num_row,2)))])
        j_fsplit = random.choice([1<<p for p in range(int(math.log(256,2)))])
        k_fsplit = random.choice([1<<p for p in range(int(math.log(num_col,2)))])
        i1 = math.ceil(num_row / i_fsplit)
        i0 = i_fsplit
        k1 = math.ceil(num_col / k_fsplit)
        k0 = k_fsplit
        j1 = math.ceil(256 / j_fsplit)
        j0 = j_fsplit
        dimensions['i1'] = i1
        dimensions['i0'] = i0
        dimensions['k1'] = k1
        dimensions['k0'] = k0
        dimensions['j1'] = j1
        dimensions['j0'] = j0
        freordered_vars = ['i1', 'i0', 'k1', 'k0'] # reorder all the axis for sparse axes.
        random.shuffle(freordered_vars)
        vars_mode = {}
        for item in (freordered_vars + ['j1', 'j0']):
            vars_mode[item] = 0
        for idx, item in enumerate(freordered_vars):
            if dimensions[item] == 1:
                vars_mode[item] = random.choice([i for i in range(5)])
            elif idx == 0:
                # The condition make level mode can't be singleton.
                # Beacuse first level and
                # Only if the dimensions is 1, the mode can be singleton.
                vars_mode[item] = random.choice([0, 1, 2])
            elif vars_mode[freordered_vars[idx - 1]] == 0: # dense
                # (un_)singleton follow dense level will make computation error, 
                # only if dimensions is 1.
                vars_mode[item] = random.choice([0, 1, 2])
            elif vars_mode[freordered_vars[idx - 1]] == 1: # sparse
                vars_mode[item] = random.choice([0, 1, 2])
            elif vars_mode[freordered_vars[idx - 1]] == 2: # un_sparse
                # only last axis can contain singleton followed un_sparse.
                if idx == len(freordered_vars) - 1: 
                    # Notice last one cant be 4
                    vars_mode[item] = random.choice([i for i in range(1, 4)])
                else: 
                    vars_mode[item] = random.choice([1, 2, 4])
            elif vars_mode[freordered_vars[idx - 1]] == 3: # singleton
                vars_mode[item] = random.choice([1, 2])
            elif vars_mode[freordered_vars[idx - 1]] == 4: # un_singleton
                # only last axis can contain singleton followed un_sparse.
                if idx == len(freordered_vars) - 1: 
                    # Notice last one cant be 4
                    vars_mode[item] = random.choice([i for i in range(1, 4)])
                else: 
                    vars_mode[item] = random.choice([1, 2, 4])
        if check_mode(freordered_vars, vars_mode, dimensions) == False:
            continue
        i1f = vars_mode['i1'] 
        i0f = vars_mode['i0'] 
        k1f = vars_mode['k1'] 
        k0f = vars_mode['k0'] 

        ### Set schedule
        ### lsplit
        freordered_vars_lsplit = freordered_vars.copy()
        lreordered_vars = []
        lreordered_vars_temp = []

        i1_lsplit = random.choice([1<<p for p in range(int(math.log(i1, 2)) + 1)])
        i0_lsplit = random.choice([1<<p for p in range(int(math.log(i0, 2)) + 1)])
        if vars_mode['i1'] < 2 and i1_lsplit > 1:
            dimensions['i11'] = math.ceil(i1 / i1_lsplit)
            dimensions['i10'] = math.ceil(i1_lsplit)
            # lreordered_vars += ['i11', 'i10']
            idx = freordered_vars_lsplit.index('i1')
            freordered_vars_lsplit[idx] = 'i11'
            freordered_vars_lsplit.insert(idx + 1, 'i10')
        else:
            dimensions['i1'] = i1
            # lreordered_vars.append('i1')
            i1_lsplit = 0
        if vars_mode['i0'] < 2 and i0_lsplit > 1:
            dimensions['i01'] = math.ceil(i0 / i0_lsplit)
            dimensions['i00'] = math.ceil(i0_lsplit)
            # lreordered_vars += ['i01', 'i00']
            idx = freordered_vars_lsplit.index('i0')
            freordered_vars_lsplit[idx] = 'i01'
            freordered_vars_lsplit.insert(idx + 1, 'i00')
        else:
            dimensions['i0'] = i0
            # lreordered_vars.append('i0')
            i0_lsplit = 0

        k1_lsplit = random.choice([1<<p for p in range(int(math.log(k1, 2)) + 1)])
        k0_lsplit = random.choice([1<<p for p in range(int(math.log(k0, 2)) + 1)])
        if vars_mode['k1'] < 2 and k1_lsplit > 1:
            dimensions['k11'] = math.ceil(k1 / k1_lsplit)
            dimensions['k10'] = math.ceil(k1_lsplit)
            # lreordered_vars += ['k11', 'k10']
            idx = freordered_vars_lsplit.index('k1')
            freordered_vars_lsplit[idx] = 'k11'
            freordered_vars_lsplit.insert(idx + 1, 'k10')
        else:
            dimensions['k1'] = k1
            # lreordered_vars.append('k1')
            k1_lsplit = 0
        if vars_mode['k0'] < 2 and k0_lsplit > 1:
            dimensions['k01'] = math.ceil(k0 / k0_lsplit)
            dimensions['k00'] = math.ceil(k0_lsplit)
            # lreordered_vars += ['k01', 'k00']
            idx = freordered_vars_lsplit.index('k0')
            freordered_vars_lsplit[idx] = 'k01'
            freordered_vars_lsplit.insert(idx + 1, 'k00')
        else:
            dimensions['k0'] = k0
            # lreordered_vars.append('k0')
            k0_lsplit = 0

        j1_lsplit = random.choice([1<<p for p in range(int(math.log(j1, 2)) + 1)])
        j0_lsplit = random.choice([1<<p for p in range(int(math.log(j0, 2)) + 1)])
        if j1_lsplit > 1:
            dimensions['j11'] = math.ceil(j1 / j1_lsplit)
            dimensions['j10'] = math.ceil(j1_lsplit)
            lreordered_vars_temp += ['j11', 'j10']
        else:
            dimensions['j1'] = j1
            lreordered_vars_temp.append('j1')
            j1_lsplit = 0
        if j0_lsplit > 1:
            dimensions['j01'] = math.ceil(j0 / j0_lsplit)
            dimensions['j00'] = math.ceil(j0_lsplit)
            lreordered_vars_temp += ['j01', 'j00']
        else:
            dimensions['j0'] = j0
            lreordered_vars_temp.append('j0')
            j0_lsplit = 0
        
        ### lreorder
        lreordered_vars = freordered_vars_lsplit.copy()
        random.shuffle(lreordered_vars_temp)
        for item in lreordered_vars_temp:
            idx = random.randint(0, len(lreordered_vars))
            lreordered_vars.insert(idx, item)
        for item in lreordered_vars: # add new splited axis mode.
            if len(item) == 3:
                vars_mode[item] = vars_mode.get(item[0:-1], 0)
        if check_lreorder(vars_mode, freordered_vars_lsplit, lreordered_vars) == False:
            continue
        
        ### prallel and vectorize
        # Notice only apply for dense axis
        parallel = "None"
        # for item in lreordered_vars:
        #     if (vars_mode[item] == 0):
        #         parallel = item
        #         break
        vectorize = "None"
        # for item in lreordered_vars[::-1]:
        #     if (vars_mode[item] == 0):
        #         vectorize = item
        #         break
        
        ### unroll
        unroll_candidate = []
        # for item in lreordered_vars:
        #     if (item != parallel and parallel != vectorize and dimensions[item] > 1 and vars_mode[item] == 0):
        #         unroll_candidate.append(item)
        if (len(unroll_candidate)):
            unroll = random.choice(unroll_candidate)
            unroll_factor = random.choice([dimensions[unroll]>>p for p in range(int(math.log(dimensions[unroll], 2)))])
        else:
            unroll = "None"
            unroll_factor = 0
        
        ### Precompute. 
        # Notiece can't apply for `singletone` related axis.
        precompute_candidate = []
        # for c1 in ['i', 'j']:
        #     for c2 in ['1', '0']:
        #         if(vars_mode.get(c1 + c2, 0) < 3):
        #             for c3 in ['', '1', '0']:
        #                 if c1 + c2 + c3 in lreordered_vars:
        #                     precompute_candidate.append(c1 + c2 + c3)
        if len(precompute_candidate):
            precompute = random.choice(precompute_candidate)
        else:
            precompute = "None"

        ### print
        num_cores = multiprocessing.cpu_count()
        thread_num = random.choice([int(num_cores), int(num_cores * 2), int(num_cores / 2)])
        parchunk = random.choice([1 << p for p in range(9)])
        configs.add("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
            i_fsplit, k_fsplit, j_fsplit, " ".join(freordered_vars), i1f, i0f, k1f, k0f,
            i1_lsplit, i0_lsplit, k1_lsplit, k0_lsplit, j1_lsplit, j0_lsplit,
            len(lreordered_vars), " ".join(lreordered_vars), 
            parallel, vectorize, unroll, unroll_factor, precompute,
            thread_num, parchunk
        ))

    output_filename = os.path.join(work_dir, 'program_sampling', 'config', mtx_name[:-4]+'.txt')
    with open(output_filename, 'w') as f:
        f.writelines(f"{item}\n" for item in configs)
    print(output_filename, " is ok")


if __name__ == '__main__':
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    if autosparse_prefix == None:
        print("Err : environment variable WACO_HOME is not defined")
        quit()
    
    with open(sys.argv[1]) as f:
        lines = f.read().splitlines()
    mtx_name_list = []
    for mtx_name in lines:
        mtx_name_list.append(mtx_name)
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # Using functools.partial to pass extra arguments.
    partial_random_generate_config = partial(
        random_generate_config, autosparse_prefix=autosparse_prefix
    )
    # Using processes pooling to handle every elemets in list.
    results = pool.map(partial_random_generate_config, mtx_name_list)
    # close processes pooling
    pool.close()
    pool.join()
