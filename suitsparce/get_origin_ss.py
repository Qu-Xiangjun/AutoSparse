# Only run in linux enviroment.
import ssgetpy
import os, sys
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))

def download_data():
    selected_matrix = ssgetpy.search(
        rowbounds=(0, 131072), colbounds=(0, 131072), nzbounds=(0, 10_000_000),
        is2d3d = False, limit = 10000
    )

    matrix_info = [] # Get all the matrix info
    for matrix in selected_matrix:
        matrix_info.append(
            [
                matrix.id,
                matrix.group,
                matrix.name,
                matrix.rows,
                matrix.cols,
                matrix.nnz,
                matrix.nnz * 1.0 /  matrix.rows / matrix.cols,
                matrix.dtype,
                matrix.is2d3d,
                matrix.isspd,
                matrix.psym,
                matrix.nsym,
                matrix.kind,
                matrix.name + ".mat"
            ]
        )

    format = "MAT"
    destpath = os.path.join(current_dir, "mat_data")
    selected_matrix.download(format = format, destpath = destpath, extract = True)
    # for idx, matrix in enumerate(selected_matrix):
    #     filepath = os.path.join(destpath, matrix.name + ".mat")
    #     if os.path.exists(filepath) == False:
    #         matrix.download(format = format, destpath = destpath, extract = True)
    #         print(str(idx) + "/" + str(len(selected_matrix)), "Successed Download ", matrix.name + ".mat")
    print("Successed Download MAT data.")

    # Save info of matirx
    csv_file_path = os.path.join(
        current_dir, 'matrix_info.csv'
    )
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 写入 CSV 文件的标题行
        csv_writer.writerow([
            "Id", "Group", "Name", "Num Rows", "Num Cols", "Num Nonzero", 
            "Sparsity", "DType", "Is 2D/3D", "Is SPD", "Pattern Symmetry",
            "Numerical Symmetry", "Kind", "File Path"
        ])
        for data_row in matrix_info:
            csv_writer.writerow(data_row)

if __name__ == "__main__":
    download_data()








