import random
import os, sys
import math
import itertools
import numpy as np

def check_lreorder(vars_mode : dict, freordered_vars : list, lreordered_vars : list):
    freordered_vars_lsplit = []
    for item in freordered_vars:
        freordered_vars_lsplit += [item + '1', item + '0']
    for idx, item in enumerate(freordered_vars_lsplit):
        if idx == 0 or vars_mode[item] == 0: 
            # First is always after head.
            # Dense axis don't rely any axis.
            continue 
        last_var = freordered_vars_lsplit[idx - 1]
        last_index = lreordered_vars.index(last_var)
        now_index = lreordered_vars.index(item)
        if last_index > now_index:
            return False
    return True

work_dir = os.getcwd()
sys.argv = [os.path.abspath(__file__),
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
            i1 = math.ceil(num_row / i_fsplit)
            i0 = i_fsplit
            i1_lsplit = random.choice([1<<p for p in range(int(math.log(i1, 2)) + 1)])
            i0_lsplit = random.choice([1<<p for p in range(int(math.log(i0, 2)) + 1)])
            dimensions['i11'] = math.ceil(i1 / i1_lsplit)
            dimensions['i10'] = math.ceil(i1_lsplit)
            dimensions['i01'] = math.ceil(i0 / i0_lsplit)
            dimensions['i00'] = math.ceil(i0_lsplit)
            k1 = math.ceil(num_col / k_fsplit)
            k0 = k_fsplit
            k1_lsplit = random.choice([1<<p for p in range(int(math.log(k1, 2)) + 1)])
            k0_lsplit = random.choice([1<<p for p in range(int(math.log(k0, 2)) + 1)])
            dimensions['k11'] = math.ceil(k1 / k1_lsplit)
            dimensions['k10'] = math.ceil(k1_lsplit)
            dimensions['k01'] = math.ceil(k0 / k0_lsplit)
            dimensions['k00'] = math.ceil(k0_lsplit)
            j1 = math.ceil(256 / j_fsplit)
            j0 = j_fsplit
            j1_lsplit = random.choice([1<<p for p in range(int(math.log(j1, 2)) + 1)])
            j0_lsplit = random.choice([1<<p for p in range(int(math.log(j0, 2)) + 1)])
            dimensions['j11'] = math.ceil(j1 / j1_lsplit)
            dimensions['j10'] = math.ceil(j1_lsplit)
            dimensions['j01'] = math.ceil(j0 / j0_lsplit)
            dimensions['j00'] = math.ceil(j0_lsplit)
            # lreorder
            lreordered_vars = ['i11', 'i10', 'i01', 'i00', 'k11', 'k10', 'k01', 'k00', 'j11', 'j10', 'j01', 'j00']
            random.shuffle(lreordered_vars)
            for item in lreordered_vars:
                vars_mode[item] = vars_mode[item[0:-1]]
            if check_lreorder(vars_mode, freordered_vars, lreordered_vars) == False:
                continue
            # prallel, unroll and vectorize
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
            unroll_candidate = []
            for item in lreordered_vars:
                if (vars_mode[item] == 0 and item != parallel and parallel != vectorize and dimensions[item] > 1):
                    unroll_candidate.append(item)
            if (len(unroll_candidate)):
                unroll = random.choice(unroll_candidate)
                unroll_factor = random.choice([dimensions[unroll]>>p for p in range(int(math.log(dimensions[unroll], 2)))])
            else:
                unroll = "None"
            # Precompute
            precompute = random.choice(['i', 'j']) + random.choice(['1', '0']) + random.choice(['1', '0'])
            
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            thread_num = random.choice([num_cores, num_cores * 2, num_cores / 2])
            parchunk = random.choice([1 << p for p in range(9)])
            configs.add("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                i_fsplit, k_fsplit, j_fsplit, " ".join(freordered_vars), i1f, i0f, k1f, k0f,
                i1_lsplit, i0_lsplit, k1_lsplit, k0_lsplit, j1_lsplit, j0_lsplit,
                " ".join(lreordered_vars), parallel, vectorize, unroll, unroll_factor, precompute,
                thread_num, parchunk
            ))

        output_filename = os.path.join(work_dir, 'program_sampling', 'config', mtx_name[:-4]+'.txt')
        with open(output_filename, 'w') as f:
            f.writelines(f"{item}\n" for item in configs)
        print(output_filename, " is ok")



