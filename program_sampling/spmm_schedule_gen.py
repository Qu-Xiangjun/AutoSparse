import random
import os, sys
import math
import itertools
import numpy as np

def check_lreorder(vars_mode : dict, freordered_vars : list, lreordered_vars : list):
    freordered_vars_lsplit = []
    for item in freordered_vars:
        freordered_vars_lsplit += [item + '1', item + '0']
    for idx, item0 in enumerate(freordered_vars_lsplit):
        if vars_mode.get(item0, 0)== 0: # Dense axis don't rely any axis.
            continue
        item0_index = freordered_vars_lsplit.index(item0)
        for item1 in freordered_vars_lsplit[idx + 1 :]:
            if vars_mode.get(item1, 0) == 0:
                continue
            item1_index = freordered_vars_lsplit.index(item1)
            if item0_index > item1_index:
                return False
    return True

current_filepath = os.path.abspath(__file__)
work_dir = os.path.dirname(os.path.dirname(current_filepath))
sys.argv = [current_filepath,
            os.path.join(work_dir, 'dataset', 'total.txt')]

if __name__ == '__main__':
    autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
    if autosparse_prefix == None:
        print("Err : environment variable WACO_HOME is not defined")
        quit()
    
    with open(sys.argv[1]) as f:
        lines = f.read().splitlines()

    for mtx_name in lines:
        mtx_name += ".csr"
        mtx_filepath = os.path.join(
            autosparse_prefix, "dataset", "suitsparse", mtx_name
        )
        # 从文件读3个数据，数据的类型为小端法（<）的4字节整数（i4）
        num_row, num_col, num_nonezero = np.fromfile(
            mtx_filepath, count=3, dtype = '<i4'
        )

        configs = set()
        while(len(configs) <= 10000):
            # Set format
            i_fsplit = random.choice([1<<p for p in range(int(math.log(num_row,2)))])
            j_fsplit = random.choice([1<<p for p in range(int(math.log(256,2)))])
            k_fsplit = random.choice([1<<p for p in range(int(math.log(num_col,2)))])
            freordered_vars = ['i1', 'i0', 'k1', 'k0']
            random.shuffle(freordered_vars)
            vars_mode = {}
            for item in (freordered_vars + ['j1', 'j0']):
                vars_mode[item] = 0
            i1f = random.choice([i for i in range(5)]) # 0 ~ 4 : indicate mode_type enum idnex.
            i0f = random.choice([i for i in range(5)])
            k1f = random.choice([i for i in range(5)])
            k0f = random.choice([i for i in range(5)])
            vars_mode['i1'] = i1f
            vars_mode['i0'] = i0f
            vars_mode['k1'] = k1f
            vars_mode['k0'] = k0f

            # Set schedule
            # lsplit
            dimensions = {}
            lreordered_vars = []
            i1 = math.ceil(num_row / i_fsplit)
            i0 = i_fsplit
            i1_lsplit = random.choice([1<<p for p in range(int(math.log(i1, 2)) + 1)])
            i0_lsplit = random.choice([1<<p for p in range(int(math.log(i0, 2)) + 1)])
            if vars_mode['i1'] < 3 and i1_lsplit > 1:
                dimensions['i11'] = math.ceil(i1 / i1_lsplit)
                dimensions['i10'] = math.ceil(i1_lsplit)
                lreordered_vars += ['i11', 'i10']
            else:
                dimensions['i1'] = i1
                lreordered_vars.append('i1')
                i1_lsplit = 0
            if vars_mode['i0'] < 3 and i0_lsplit > 1:
                dimensions['i01'] = math.ceil(i0 / i0_lsplit)
                dimensions['i00'] = math.ceil(i0_lsplit)
                lreordered_vars += ['i01', 'i00']
            else:
                dimensions['i0'] = i0
                lreordered_vars.append('i0')
                i0_lsplit = 0
            k1 = math.ceil(num_col / k_fsplit)
            k0 = k_fsplit
            k1_lsplit = random.choice([1<<p for p in range(int(math.log(k1, 2)) + 1)])
            k0_lsplit = random.choice([1<<p for p in range(int(math.log(k0, 2)) + 1)])
            if vars_mode['k1'] < 3 and k1_lsplit > 1:
                dimensions['k11'] = math.ceil(k1 / k1_lsplit)
                dimensions['k10'] = math.ceil(k1_lsplit)
                lreordered_vars += ['k11', 'k10']
            else:
                dimensions['k1'] = k1
                lreordered_vars.append('k1')
                k1_lsplit = 0
            if vars_mode['k0'] < 3 and k0_lsplit > 1:
                dimensions['k01'] = math.ceil(k0 / k0_lsplit)
                dimensions['k00'] = math.ceil(k0_lsplit)
                lreordered_vars += ['k01', 'k00']
            else:
                dimensions['k0'] = k0
                lreordered_vars.append('k0')
                k0_lsplit = 0
            j1 = math.ceil(256 / j_fsplit)
            j0 = j_fsplit
            j1_lsplit = random.choice([1<<p for p in range(int(math.log(j1, 2)) + 1)])
            j0_lsplit = random.choice([1<<p for p in range(int(math.log(j0, 2)) + 1)])
            if j1_lsplit > 1:
                dimensions['j11'] = math.ceil(j1 / j1_lsplit)
                dimensions['j10'] = math.ceil(j1_lsplit)
                lreordered_vars += ['j11', 'j10']
            else:
                dimensions['j1'] = j1
                lreordered_vars.append('j1')
                j1_lsplit = 0
            if j0_lsplit > 1:
                dimensions['j01'] = math.ceil(j0 / j0_lsplit)
                dimensions['j00'] = math.ceil(j0_lsplit)
                lreordered_vars += ['j01', 'j00']
            else:
                dimensions['j0'] = j0
                lreordered_vars.append('j0')
                j0_lsplit = 0
            # lreorder
            random.shuffle(lreordered_vars)
            for item in lreordered_vars:
                if len(item) == 3:
                    vars_mode[item] = vars_mode[item[0:-1]]
            if check_lreorder(vars_mode, freordered_vars, lreordered_vars) == False:
                continue
            # prallel and vectorize
            # Notice only apply for dense axis
            parallel = "None"
            for item in lreordered_vars:
                if (vars_mode[item] == 0):
                    parallel = item
                    break
            vectorize = "None"
            for item in lreordered_vars[::-1]:
                if (vars_mode[item] == 0):
                    vectorize = item
                    break
            # unroll
            unroll_candidate = []
            for item in lreordered_vars:
                if (item != parallel and parallel != vectorize and dimensions[item] > 1):
                    unroll_candidate.append(item)
            if (len(unroll_candidate)):
                unroll = random.choice(unroll_candidate)
                unroll_factor = random.choice([dimensions[unroll]>>p for p in range(int(math.log(dimensions[unroll], 2)))])
            else:
                unroll = "None"
            # Precompute. Notiece can' apply for `singletone` related axis.
            precompute_candidate = []
            for c1 in ['i', 'j']:
                for c2 in ['1', '0']:
                    if(vars_mode[c1 + c2] < 3):
                        for c3 in ['1', '0']:
                            precompute_candidate.append(c1 + c2 + c3)
            if len(precompute_candidate):
                precompute = random.choice(precompute_candidate)
            else:
                precompute = "None"

            
            import multiprocessing
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



