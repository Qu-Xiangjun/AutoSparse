import os, sys
import numpy as np
import csv
from tqdm import tqdm
import time
import re


dataset_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.dirname(dataset_dir)

def get_all_files_in_directory(directory):
    file_names = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_names.append(filename)
    return file_names


dataset_info = []
csv_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'dataset_analyse.csv'
)
if os.path.exists(csv_file_path) == False:
    print("[INFO] analyse dataset info.")
    files_in_directory = get_all_files_in_directory(dataset_dir)
    for file_name in tqdm(
        files_in_directory, total = len(files_in_directory),
        desc = "Analyse csr file"
    ):
        if ".csr" not in file_name:
            continue
        mtx_filepath = os.path.join(dataset_dir, file_name)
        # 从文件读3个数据，数据的类型为小端法（<）的4字节整数（i4）
        num_row, num_col, num_nonezero = np.fromfile(
            mtx_filepath, count=3, dtype = '<i4'
        )
        
        dataset_info.append([
            file_name.split(".")[0], num_row, num_col, 
            num_nonezero, num_nonezero*1.0/num_row/num_col,
            os.path.getsize(mtx_filepath)
        ])

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 写入 CSV 文件的标题行
        csv_writer.writerow([
            "File Name", "Num Rows", "Num Cols", "Num Nonzero", 
            "Sparsity", "Filesize"
        ])
        for data_row in tqdm(
            dataset_info, total = len(dataset_info), desc="Write analyse csv file"
        ):
            csv_writer.writerow(data_row)
else:
    print("[INFO] Load existed dataset analyse file.")
    with open(csv_file_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        for row in tqdm(csv_reader, desc="Reading CSV"):
            dataset_info.append(row)


### get origin file, which isn't augmented.
origin_file = []
argumented_file = []
pattern = re.compile(r'\d{1,2}x\d{1,2}')
for data_row in dataset_info:
    filename = data_row[0].split(".")[0]
    splited_name = filename.split("_")
    if len(splited_name) > 2 and pattern.search(splited_name[-2]):
        argumented_file.append(data_row)
    else:
        origin_file.append(data_row)


# ### dataset partition
# dataset_info = sorted(dataset_info, key = lambda x: x[-1])
# seletced_dataset_info = []
# for data_row in dataset_info:
#     # filter.
#     if num_row < 100 and num_col < 100: # mini data
#         continue
#     if num_row * 100 < num_col or num_row > num_col * 100:
#         # row col diff so much in size
#         continue
#     if data_row[1] <= 131072 and data_row[3] < 10_000_000:
#         seletced_dataset_info.append(data_row)

# csv_file_path = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 'selected_dataset_analyse.csv'
# )
# with open(csv_file_path, 'w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     # 写入 CSV 文件的标题行
#     csv_writer.writerow([
#         "File Name", "Num Rows", "Num Cols", "Num Nonzero", "Sparsity"
#     ])
#     for data_row in seletced_dataset_info:
#         csv_writer.writerow(data_row)

# output_filename_total = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 'autosparse_total.txt'
# )
# output_filename_validation = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 'autosparse_validation.txt'
# )
# output_filename_train = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 'autosparse_total.txt'
# )
