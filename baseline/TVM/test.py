import os

import numpy as np
import scipy.sparse as sp

def from_csr(filepath) :
    csr = np.fromfile(filepath, dtype='<i4')
    num_row,num_col,nnz = csr[0],csr[1],csr[2]
    coo = np.zeros((nnz,2),dtype=int)
    coo[:,1] = csr[3+num_row+1:3+num_row+1+nnz] # col
    bins = np.array(csr[4:num_row+4]) - np.array(csr[3:num_row+3])
    coo[:,0] = np.repeat(range(num_row), bins)
    return num_row, num_col, nnz, coo

filename = "Trec6_16x16_9"

autosparse_prefix = os.getenv("AUTOSPARSE_HOME")
csr_filepath = os.path.join(autosparse_prefix, "dataset", "demo_dataset", filename+".csr")
num_row, num_col, nnz, coo = from_csr(csr_filepath)
print(f"{filename}: num_row = {num_row}, num_col = {num_col}, nnz = {nnz}")

M = num_row
K = num_col
N = 256
BS_R = 16
BS_C = 16

# Generate the test data with numpy
X_np = np.random.randn(M, K).astype("float32")
W_np = np.zeros((num_row, num_col), dtype="float32")
W_np[coo[:,0], coo[:,1]] = 1.0
W_sp_np = sp.bsr_matrix(W_np, blocksize=(BS_R, BS_C))

print(W_sp_np.data.shape)
print(W_sp_np.indptr.shape[0])
print(W_sp_np.indices.shape[0])

W_np_test = W_sp_np.todense()

# 逐元素比较
close_elements = np.isclose(W_np, W_np_test, rtol=1e-6, atol=1e-9)
print(close_elements)  # 输出结果

# Y_np = X_np @ W_np.T  # Process the matrix multiplication


