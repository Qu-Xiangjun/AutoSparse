import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, coo_matrix
import os
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))

def get_all_files_in_directory(directory):
    file_names = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_names.append(filename)
    return file_names

def mtx_2_csr(file_path):
    matrix = loadmat(file_path)['Problem']['A'][0, 0]
    csr_matrix_format = csr_matrix(matrix) # Convert to csr format.
    num_rows, num_cols = csr_matrix_format.shape
    num_nonzero = csr_matrix_format.nnz
    pos = csr_matrix_format.indptr
    crd = csr_matrix_format.indices
    val = csr_matrix_format.data
    # Print info
    # print("Number of Rows:", num_rows)
    # print("Number of Columns:", num_cols)
    # print("Number of Nonzero Values:", num_nonzero)
    # print("Row Offset Array (pos):", pos)
    # print("Column Index Array (crd):", crd)
    # print("Nonzero Value Array (val):", val)

    return num_rows, num_cols, num_nonzero, pos, crd, val

def mtx_2_coo(file_path):
    matrix = loadmat(file_path)['Problem']['A'][0, 0]
    num_rows, num_cols = matrix.shape
    num_nonzero = matrix.nnz
    coo_matrix_format = coo_matrix(matrix) # Convert to coo format.
    rows = coo_matrix_format.row
    cols = coo_matrix_format.col
    data = coo_matrix_format.data
    # Print info
    # print("COO Matrix Shape:", coo_matrix_format.shape)
    # print("COO Matrix Nonzero Count:", coo_matrix_format.nnz)
    # print("COO Matrix Data Type:", coo_matrix_format.dtype)

    return num_rows, num_cols, num_nonzero, rows, cols, data

def from_csr(file_path):
    csr = np.fromfile(file_path, dtype="<i4")
    num_row, num_col, nnz = csr[0], csr[1], csr[2]
    coo = np.zeros((nnz, 2), dtype=int)
    coo[:,1] = csr[3 + num_row + 1:]
    bins = np.array(csr[4:num_row+4]) - np.array(csr[3:num_row+3])
    coo[:,0] = np.repeat(range(num_row), bins)
    return num_row, num_col, nnz, coo

if __name__ == "__main__":
    mat_data_path = os.path.join(current_dir, "mat_data")
    file_names = get_all_files_in_directory(mat_data_path)
    csr_data_path = os.path.join(current_dir, "csr_data")
    if os.path.exists(csr_data_path) == False:
        os.mkdir(csr_data_path)

    exists_files = []
    for idx, file_name in enumerate(file_names):
        file_path = os.path.join(mat_data_path, file_name)
        output_file_path = os.path.join(csr_data_path, file_name.split(".")[0] + ".csr")
        if os.path.exists(output_file_path):
            continue
        try:
            num_rows, num_cols, num_nonzero, pos, crd, _ = mtx_2_csr(file_path)
        except Exception as e:
            print(file_name, " is error.")
            os.remove(file_path) # Delete error .mat file.
            continue
        exists_files.append(file_name) # Correct mat file.
        with open(output_file_path, 'wb') as csr_file:
            csr_file.write(num_rows.to_bytes(4, 'little'))  # little endding 
            csr_file.write(num_cols.to_bytes(4, 'little'))  
            csr_file.write(num_nonzero.to_bytes(4, 'little')) 
            pos.tofile(csr_file)
            crd.tofile(csr_file)
        print(str(idx)+"/"+str(len(file_names)), " ", file_name, " successed convert to csr.")
        # num_row, num_col, nnz, coo = from_csr(output_file_path)

    # Delete error file info in statistical csv.
    csv_file_path = os.path.join(current_dir, 'matrix_info.csv')
    read_data = []
    with open(csv_file_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        for row in csv_reader:
            if row[-1] not in exists_files:
                continue # Delete error file
            read_data.append(row)
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 写入 CSV 文件的标题行
        csv_writer.writerow([
            "Id", "Group", "Name", "Num Rows", "Num Cols", "Num Nonzero", 
            "Sparsity", "DType", "Is 2D/3D", "Is SPD", "Pattern Symmetry",
            "Numerical Symmetry", "Kind", "File Path"
        ])
        for data_row in read_data:
            csv_writer.writerow(data_row)

