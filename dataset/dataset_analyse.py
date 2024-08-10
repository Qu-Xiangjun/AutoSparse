import os, sys
import numpy as np
import csv

dataset_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.dirname(dataset_dir)
dataset_dir = os.path.join(dataset_dir, 'pretrained', 'dataset')

def get_all_files_in_directory(directory):
    file_names = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_names.append(filename)
    return file_names

files_in_directory = get_all_files_in_directory(dataset_dir)

dataset_info = []
for file_name in files_in_directory:
    mtx_filepath = os.path.join(dataset_dir, file_name)
    num_row, num_col, num_nonezero = np.fromfile(
        mtx_filepath, count=3, dtype = '<i4'
    )
    dataset_info.append([
        file_name.split(".")[0], num_row, num_col, 
         num_nonezero, num_nonezero*1.0/num_row/num_col
    ])

csv_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'dataset_analyse.csv'
)

with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "File Name", "Num Rows", "Num Cols", "Num Nonzero", "Sparsity"
    ])
    for data_row in dataset_info:
        csv_writer.writerow(data_row)

### Select dataset
seletced_dataset_info = []
for data_row in dataset_info:
    if data_row[1] <= 131072 and data_row[3] < 10_000_000:
        seletced_dataset_info.append(data_row)

csv_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'selected_dataset_analyse.csv'
)
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "File Name", "Num Rows", "Num Cols", "Num Nonzero", "Sparsity"
    ])
    for data_row in seletced_dataset_info:
        csv_writer.writerow(data_row)

output_filename_total = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'autosparse_total.txt'
)
output_filename_validation = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'autosparse_validation.txt'
)
output_filename_train = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'autosparse_total.txt'
)
