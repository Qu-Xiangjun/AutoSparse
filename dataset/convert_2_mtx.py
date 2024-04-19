import os, sys

from AutoSparse.utils import get_coo_from_csr_file, write_mtx_from_coo

current_directory = os.path.dirname(os.path.abspath(__file__))

matrix_total_file = os.path.join(current_directory, "total.txt")
with open(matrix_total_file) as f :
    matrix_names = f.read().splitlines()
    for name in matrix_names:
        num_row, num_col, nnz, coo = get_coo_from_csr_file(
            os.path.join(current_directory, "demo_dataset", name + ".csr")
        )
        print(f"{name} : {num_row} {num_col} {nnz}")
        write_mtx_from_coo(
            num_row, num_col, coo, 
            os.path.join(current_directory, "mtx_demo_dataset", name + ".mtx")
        )