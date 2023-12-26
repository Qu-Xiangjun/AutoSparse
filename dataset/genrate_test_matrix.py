import numpy as np
import random
import os
import struct

def write_matrix_2_csr_file(matirx : np.array, filepath: str, filename: str):
    """
    Write a matirx as csr format in file as postfix 'csr'.
    The file will store three int type number for row, col and nonezores numbers firstly,
    and store pos, crd, val array.

    Parameters
    ----------
    arg1 : matrix np.array
        A 2D matirx indense format.
    arg2 : filepath 
        The output csr file path.
    arg3 : filename
        The output csr file name without postfix.
    """
    row_indices, col_indices = np.nonzero(matirx)
    num_nonzero = len(row_indices)
    A_crd = col_indices.tolist()
    A_val = a[row_indices, col_indices].tolist()

    # 存储到文件
    filename = os.path.join(filepath, filename+'.csr')
    with open(filename, 'wb') as csr_file:
        num_row, num_col = a.shape
        csr_file.write(num_row.to_bytes(4, 'little'))  # 存储行数, 小端存储
        csr_file.write(num_col.to_bytes(4, 'little'))  # 存储列数
        csr_file.write(num_nonzero.to_bytes(4, 'little'))  # 存储非零数量
        
        A_pos = [0]  # 初始位置
        for i in range(1, num_row + 1):
            A_pos.append(A_pos[i - 1] + A_crd.count(i - 1))  # 计算每行的非零元素数

        for pos in A_pos:
            csr_file.write(pos.to_bytes(4, 'little'))  # 存储 A_pos

        for crd in A_crd:
            csr_file.write(crd.to_bytes(4, 'little'))  # 存储 A_crd
        
        for val in A_val:
            csr_file.write(struct.pack('<f', val))  # 使用 struct 模块将浮点数转换为字节流


# a * b -> c
random.seed(0)
a = np.ones((32, 32))
b = np.ones((32, 256))

sparsity = 1.0 / (2**4)
for i in range(32):
    # zero_count = 256 - int(256 * sparsity)
    # index = [i for i in range(256)]
    # random.shuffle(index)
    # index = index[:zero_count]
    # for j in index:
    #     a[i, j] = 0

    for j in range(32):
        if i != j:
            a[i, j] = 0

c = np.matmul(a, b) 
print(c)

# 存下来a
write_matrix_2_csr_file(a, "./dataset", 'test_matrix')
