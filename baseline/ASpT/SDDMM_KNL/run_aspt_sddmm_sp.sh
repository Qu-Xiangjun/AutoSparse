#!/bin/bash

# SpMM程序的路径
SPMM_BIN="./SDDMM_ASpT_SP.x"

# 数据集目录的路径
DATASET_DIR="../../../dataset/mtx_demo_dataset/"

# 结果文件的路径
RESULT_FILE="result"

# 检查SpMM程序是否存在
if [ ! -f "$SPMM_BIN" ]; then
    echo "SpMM binary not found at $SPMM_BIN"
    exit 1
fi

# 遍历DATASET_DIR目录下的所有.mtx文件
find "$DATASET_DIR" -type f -name "*.mtx" | while read mtx_file; do
    echo "Processing $mtx_file"

    # 运行SpMM程序并将输出追加到结果文件
    ./SDDMM_ASpT_SP.x "$mtx_file" >> "$RESULT_FILE"
    
    # 检查程序运行是否成功
    if [ $? -ne 0 ]; then
        echo "SpMM failed on $mtx_file"
    fi
done

echo "All .mtx files have been processed."