import random
import os
import itertools
import time
import sys
import math
import numpy as np

sys.argv = ['/home/qxj/Workload-Aware-Co-Optimization/WACO/training_data_generator/SpMM_SuperSchedule_Generator.py',
            '/home/qxj/Workload-Aware-Co-Optimization/WACO/SpMM/TrainingData/total.txt']

result = ""
currinput = ""
num_row = 0
num_col = 0
num_nonzero = 0


def filter(l, j):
    """
        过滤不合法的参数 
        保证_1 在 _0 之前;
        保证j 或者至少j0在末尾;
        保障i1 在 j1之前;
        """
    i1 = l.index('i1')
    k1 = l.index('k1')
    j1 = l.index('j1')
    i0 = l.index('i0')
    k0 = l.index('k0')
    j0 = l.index('j0')

    if j < 8:
        lastj = j1 == 4 and j0 == 5
    else:
        lastj = j0 == 5

    return (i1 < i0) and (k1 < k0) and (j1 < j0) and lastj and (j1 > i1)


if __name__ == '__main__':
    waco_prefix = os.getenv("WACO_HOME")
    if waco_prefix == None:
        print("Err : environment variable WACO_HOME is not defined")
        quit()

    with open(sys.argv[1]) as f:
        lines = f.read().splitlines()

    for idx, mtx_name in enumerate(lines):
        # 从文件读3个数据，数据的类型为小端法（<）的4字节整数（i4）
        csr = np.fromfile(
            waco_prefix + "/dataset/{0}.csr".format(mtx_name), count=3, dtype='<i4')
        num_row, num_col, num_nonzero = csr[0], csr[1], csr[2]
        
        cfgs = set()
        cfg = {}
        while (len(cfgs) < 32):
            # random.choice([1<<p for p in range(int(math.log(num_row,2)))])
            cfg['isplit'] = random.choice([1, 2, 4, 8, 16, 32])
            # random.choice([1<<p for p in range(int(math.log(num_col,2)))])
            cfg['ksplit'] = random.choice([1, 2, 4, 8, 16, 32])
            cfg['jsplit'] = random.choice(
                [1 << p for p in range(int(math.log(256, 2)))])
            cfg['rankorder'] = ['i1', 'k1', 'j1', 'i0', 'k0', 'j0']
            cfg['i1f'] = random.choice([0, 1])
            cfg['i0f'] = 1  # random.choice([0,1])
            cfg['k1f'] = 0  # random.choice([0,1])
            cfg['k0f'] = 1  # random.choice([0,1])
            cfg['paridx'] = random.choice(['i1'])
            cfg['parnum'] = random.choice([48])
            cfg['parchunk'] = random.choice([1 << p for p in range(9)])
            isplit, ksplit, jsplit = cfg['isplit'], cfg['ksplit'], cfg['jsplit']
            rankorder = " ".join(cfg['rankorder'])
            i1f, i0f, k1f, k0f = cfg['i1f'], cfg['i0f'], cfg['k1f'], cfg['k0f']
            paridx, parnum, parchunk = cfg['paridx'], cfg['parnum'], cfg['parchunk']
            cfgs.add("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n".format(
                isplit, ksplit, jsplit, rankorder, i1f, i0f, k1f, k0f, paridx, parnum, parchunk))

        while (len(cfgs) < 100):
            # 对各个轴进行split
            # random.choice([1<<p for p in range(int(math.log(num_row,2)))])
            cfg['isplit'] = random.choice([1, 2, 4, 8, 16, 32])
            # random.choice([1<<p for p in range(int(math.log(num_col,2)))])
            cfg['ksplit'] = random.choice([1, 2, 4, 8, 16, 32])
            cfg['jsplit'] = random.choice(
                [1 << p for p in range(int(math.log(256, 2)))])
            # 重排序循环顺序
            cfg['rankorder'] = random.choice([p for p in list(itertools.permutations(
                ['i1', 'i0', 'k1', 'k0', 'j1', 'j0'])) if filter(p, cfg['jsplit'])])
            # 设置 稀疏矩阵轴的格式
            cfg['i1f'] = random.choice([0, 1])
            cfg['i0f'] = random.choice([0, 1])
            cfg['k1f'] = random.choice([0, 1])
            cfg['k0f'] = random.choice([0, 1])
            # 设置 parallel
            cfg['paridx'] = random.choice(['i1'])
            cfg['parnum'] = random.choice([48])
            cfg['parchunk'] = random.choice([1 << p for p in range(9)])
            isplit, ksplit, jsplit = cfg['isplit'], cfg['ksplit'], cfg['jsplit']
            rankorder = " ".join(cfg['rankorder'])
            i1f, i0f, k1f, k0f = cfg['i1f'], cfg['i0f'], cfg['k1f'], cfg['k0f']
            paridx, parnum, parchunk = cfg['paridx'], cfg['parnum'], cfg['parchunk']
            cfgs.add("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n".format(
                isplit, ksplit, jsplit, rankorder, i1f, i0f, k1f, k0f, paridx, parnum, parchunk))

        f = open("./config/{}.txt".format(mtx_name), 'w')
        for sched in cfgs:
            f.write(sched)
        f.close()
