import os, sys
import numpy as np
import csv
from tqdm import tqdm
import random 

prefix = os.getenv("AUTOSPARSE_HOME")
pretrained_dataset_dir = os.path.join(prefix, 'pretrained', 'dataset')
dataset_dir = os.path.join(prefix, 'waco_dataset')

def get_all_files_in_directory(directory):
    file_names = []
    for filename in os.listdir(directory):
        if '.csr' in str(filename) and os.path.isfile(os.path.join(directory, filename)):
            file_names.append(filename)
    return file_names

# pretrained_files_in_directory = get_all_files_in_directory(pretrained_dataset_dir)

# pretrained_dataset_mtx_names = []
# for file_name in pretrained_files_in_directory:
#     mtx_filepath = os.path.join(dataset_dir, file_name)
#     # 从文件读3个数据，数据的类型为小端法（<）的4字节整数（i4）
#     num_row, num_col, num_nonezero = np.fromfile(
#         mtx_filepath, count=3, dtype = '<i4'
#     )
#     if num_row <= 131072 and num_row >= 1024 and num_col >= 1024 and \
#         num_nonezero >= 100_000 and num_nonezero <= 5_000_000:
#         pretrained_dataset_mtx_names.append(file_name[:-4])

# random.shuffle(pretrained_dataset_mtx_names)
# pretrained_dataset_mtx_names = pretrained_dataset_mtx_names[0:350]

# test_dataset_names = pretrained_dataset_mtx_names[:100]
# total_data_nemes = pretrained_dataset_mtx_names[100:]



files_in_directory = get_all_files_in_directory(dataset_dir)

origin_data_info = []
handle_dataset_info = []

modes = ['1x2', '2x4', '1x4', '8x16', '8x8', '4x4', '2x2', '16x16', '4x8', '8x4', '16x8', '2x16', '4x2', 
        '4x1', '8x2', '3x1', '1x3', '4x16', '16x2', '2x8', '16x4', '16x1', '1x8', '2x1', '1x1', '1x16', '8x1']

for file_name in tqdm(files_in_directory,total=len(files_in_directory)):
    is_origin_matrix = True
    for md in modes:
        if md in file_name:
            is_origin_matrix = False
            break
    mtx_filepath = os.path.join(dataset_dir, file_name)
    # 从文件读3个数据，数据的类型为小端法（<）的4字节整数（i4）
    num_row, num_col, num_nonezero = np.fromfile(
        mtx_filepath, count=3, dtype = '<i4'
    )
    if num_row <= 131072 and num_row >= 1024 and num_col >= 1024 and \
        num_nonezero >= 100_000 and num_nonezero <= 5_000_000:
        if is_origin_matrix:
            origin_data_info.append([
                file_name[:-4], num_row, num_col, 
                num_nonezero, num_nonezero*1.0/num_row/num_col
            ])
        else:
            handle_dataset_info.append([
                file_name[:-4], num_row, num_col, 
                num_nonezero, num_nonezero*1.0/num_row/num_col
            ])

random.shuffle(origin_data_info)
origin_data_info = origin_data_info[:530]
origin_mtx_names = [item[0] for item in origin_data_info]

seletced_dataset_info = origin_data_info


for item in handle_dataset_info:
    mtx_name = item[0]
    if '_'.join(mtx_name.split('_')[:-2]) not in origin_mtx_names:
        continue
    if '4x4' not in mtx_name and '8x4' not in mtx_name and '4x8' not in mtx_name and\
        '8x8' not in mtx_name and '16x8' not in mtx_name and '8x16' not in mtx_name and\
        '4x16' not in mtx_name and '16x4' not in mtx_name and '2x16' not in mtx_name and\
        '16x2' not in mtx_name and '8x2' not in mtx_name and '2x8' not in mtx_name and\
        '16x16' not in mtx_name:
        continue
    seletced_dataset_info.append(item)


csv_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'selected_dataset_analyse.csv'
)
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "File Name", "Num Rows", "Num Cols", "Num Nonzero", "Sparsity"
    ])
    for data_row in tqdm(seletced_dataset_info, total = len(seletced_dataset_info)):
        csv_writer.writerow(data_row)


test_mtx_names= origin_mtx_names[:200]
total_mtx_names = [item[0] for item in seletced_dataset_info[200:]]
random.shuffle(total_mtx_names)

train_data = total_mtx_names[0:int(len(total_mtx_names)*0.8)]
validation_data = total_mtx_names[int(len(total_mtx_names)*0.8):]

output_filename_total = os.path.join(
    prefix, 'dataset', 'total.txt'
)
with open(output_filename_total, 'w', newline='') as fp:
    for item in total_mtx_names:
        fp.write(item+'\n')

output_filename_validation = os.path.join(
    prefix, 'dataset', 'validation.txt'
)
with open(output_filename_validation, 'w', newline='') as fp:
    for item in validation_data:
        fp.writelines(item+'\n')

output_filename_train = os.path.join(
    prefix, 'dataset', 'train.txt'
)
with open(output_filename_train, 'w', newline='') as fp:
    for item in train_data:
        fp.writelines(item+'\n')

output_filename_test = os.path.join(
    prefix, 'dataset', 'test.txt'
)
with open(output_filename_test, 'w', newline='') as fp:
    for item in test_mtx_names:
        fp.writelines(item+'\n')