#!/bin/bash

# 获取当前时间，格式为时分秒
current_time=$(date +"%H%M%S")

# 定义日志文件名
log_file="log_${current_time}.txt"

# 执行命令并将输出重定向到日志文件
./bin/spmm ../../dataset/suitsparse/test_matrix.csr ../config/test_matrix.txt ../config_waco/test_matrix.txt >> "$log_file" 2>&1
