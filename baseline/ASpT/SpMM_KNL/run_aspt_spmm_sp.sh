#!/bin/bash

# # SpMM程序的路径
# SPMM_BIN="./SpMM_ASpT_SP.x"

# # 数据集目录的路径
# DATASET_DIR="../../../dataset/mtx_demo_dataset/

# # 结果文件的路径
# RESULT_FILE="result"

# # 检查SpMM程序是否存在
# if [ ! -f "$SPMM_BIN" ]; then
#     echo "SpMM binary not found at $SPMM_BIN"
#     exit 1
# fi

# # 遍历DATASET_DIR目录下的所有.mtx文件
# find "$DATASET_DIR" -type f -name "*.mtx" | while read mtx_file; do
#     echo "Processing $mtx_file"

#     # 运行SpMM程序并将输出追加到结果文件
#     ./SpMM_ASpT_SP.x "$mtx_file" >> "$RESULT_FILE"
    
#     # 检查程序运行是否成功
#     if [ $? -ne 0 ]; then
#         echo "SpMM failed on $mtx_file"
#     fi
# done

# echo "All .mtx files have been processed."

#!/bin/bash

# 循环执行命令 100 次
# for i in {1..100}
# do
    # ./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/
# done


./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/mhd4800a.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/conf5_0-4x4-18.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/cca.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/Trec6_16x16_9.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/bcsstk38.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/Trefethen_20000.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/nemspmm1_16x4_0.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/pf2177.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/crystk01_2x16_1.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/EX1_8x8_4.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/msc10848.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/t2dal_a_8x4_3.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/cfd1.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/net100.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/vanbody.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/net150.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/NACA0015_16x8_9.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/vibrobox_1x1_0.mtx
./SpMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/Chevron3_4x16_1.mtx