import torch
import numpy as np
import random
import os
import struct

def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        # 对角线及以下的元素置为1，其余置为0，形成全局的注意力遮罩
        b = torch.tril(torch.ones(n, n))
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        # 仅保留局部关注上下文范围内的元素，超出范围的置为0
        ctx = min(n - 1, bandwidth - 1)
        b = torch.tril(torch.ones(n, n), diagonal=ctx)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        # 创建一个 n x n 的矩阵，表示每个位置的索引
        x = torch.arange(n).unsqueeze(1)
        y = x.t()
        # 生成一个大小为 n x n 的全零矩阵
        z = torch.zeros(n, n, dtype=torch.int32)
        # 求出每个位置的 q 和 k 值
        q = z + x
        k = z + y
        # 判断条件 1：q >= k
        c1 = q >= k
        # 判断条件 2：(q - k) 与 stride 的余数为 0
        c2 = torch.remainder(q - k, stride) == 0
        # 组合两个条件
        c3 = c1 & c2
        # 将结果转换为浮点数，形成 strided 注意力遮罩
        b = c3.float()
    else:
        raise ValueError('Not yet implemented')
    
    # 在最后两个维度上添加两个维度，以符合 PyTorch 的张量形状
    b = b
    return b



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
    row_indices, col_indices = np.nonzero(matirx.numpy())
    num_nonzero = len(row_indices)
    A_crd = col_indices.tolist()
    A_val = matirx[row_indices, col_indices].tolist()

    # 存储到文件
    filename = os.path.join(filepath, filename+'.csr')
    with open(filename, 'wb') as csr_file:
        num_row, num_col = matirx.shape
        csr_file.write(num_row.to_bytes(4, 'little'))  # 存储行数, 小端存储
        csr_file.write(num_col.to_bytes(4, 'little'))  # 存储列数
        csr_file.write(num_nonzero.to_bytes(4, 'little'))  # 存储非零数量
        
        A_pos = [0]  # 初始位置
        for i in range(1, num_row + 1):
            A_pos.append(A_pos[i - 1] + row_indices.tolist().count(i - 1))  # 计算每行的非零元素数

        for pos in A_pos:
            csr_file.write(pos.to_bytes(4, 'little'))  # 存储 A_pos

        for crd in A_crd:
            csr_file.write(crd.to_bytes(4, 'little'))  # 存储 A_crd
        
        for val in A_val:
            csr_file.write(struct.pack('<f', val))  # 使用 struct 模块将浮点数转换为字节流

stirded_matrix = get_attn_mask(4096, 'strided', 32)
write_matrix_2_csr_file(stirded_matrix, "", 'strides_mask')