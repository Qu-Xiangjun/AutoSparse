首先配置好AUTOSPARSE的环境跟着主仓库的README.md


# WACO 配置环境
1. 下载WACO仓库中pretrained压缩包，然后将其中 WACO目录下的SpMM, SpMV, SDDMM 的 TraningData, resnet.pth 复制到本目录下的waco文件夹
2. 运行buildKNN.py文件，自动从 TraningData 构建KNN网络，其中需要用到resnet.pth的预训练网络
3. 最后运行ANNSearch.py文件搜索目标稀疏矩阵的最佳配置，任务稀疏矩阵在 ../dataset/total.txt中


# TVM-S 配置环境
1. pip install apache-tvm
2. 运行 tune_sparse_x86.py 文件搜索最佳调度为任务，任务稀疏矩阵在 ../dataset/total.txt中


# ASpT 配置环境
1. 安装icpc编译器 https://hackmd.io/@nadhifmr/H1HcwPeUj
2. 安装MKL库 
```bash
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/2f3a5785-1c41-4f65-a2f9-ddf9e0db3ea0/l_onemkl_p_2024.1.0.695_offline.sh
sudo sh ./l_onemkl_p_2024.1.0.695_offline.sh -a --cli 
```
3. 拉取ASpT仓库，按照其readme 修改makefile.in中编译器位置和MKL库位置，
   - 注意2024.1版本很多文件与其原有的已经不匹配，注意修改一些地址的路径位置
   - 如果报错.so库找不到，就从MKL库的路径中添加到.bashrc ，如 `export LD_LIBRARY_PATH=/path/to/libiomp5:$LD_LIBRARY_PATH`
   - 注意如果没有AVX-512 删除 各个文件夹下makefile中 avx512和向量指令的编译命令
   - 进入各个文件夹单独make，然后按照run_KNL.sh 脚本中的执行方式单独执行每个可执行文件