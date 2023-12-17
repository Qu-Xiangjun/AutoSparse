import pytaco as pt
from pytaco import compressed, dense
import numpy as np

# Define formats for storing the sparse matrix and dense vectors
csr = pt.format([dense, compressed])
dv  = pt.format([dense])

# Load a sparse matrix stored in the matrix market format) and store it 
# as a CSR matrix.  The matrix in this example can be downloaded from:
# https://www.cise.ufl.edu/research/sparse/MM/Boeing/pwtk.tar.gz
A = pt.read("pwtk.mtx", csr)

# Generate two random vectors using NumPy and pass them into TACO
x = pt.from_array(np.random.uniform(size=A.shape[1]))
z = pt.from_array(np.random.uniform(size=A.shape[0]))

# Declare the result to be a dense vector
y = pt.tensor([A.shape[0]], dv)

# Declare index vars
i, j = pt.get_index_vars(2)

# Define the SpMV computation
y[i] = A[i, j] * x[j] + z[i]

# Perform the SpMV computation and write the result to file
pt.write("y.tns", y)