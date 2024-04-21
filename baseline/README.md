You are already config main env in AutoSparse README.md.

The next topics are config every baseline env.

# WACO
0. install hnswlib 
```bash
cd $AUTOSPARSE_HOME/baseline/waco/hnswlib
pip install .
```
1. building a KNN graph
```bash
python $AUTOSPARSE_HOME/baseline/waco/buildKNN.py
```
2. search using ANNS
```bash
python $AUTOSPARSE_HOME/baseline/waco/ANNSearch.py
```
result will be writed in `$AUTOSPARSE_HOME/baseline/waco/result.txt`

# TVM-S
0. `pip install apache-tvm`
1. run `tune_sparse_x86.py` and result will write to cmd.

# MKL lib in intel oneAPI
0. install MKL lib
```bash
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/2f3a5785-1c41-4f65-a2f9-ddf9e0db3ea0/l_onemkl_p_2024.1.0.695_offline.sh
sudo sh ./l_onemkl_p_2024.1.0.695_offline.sh -a --cli 
# Config env
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2024.1/lib:$LD_LIBRARY_PATH
```
1. compile test file
```bash
cd $AUTOSPARSE_HOME/baseline/MKL
make
```
2. run test file

```bash
./bin/mkl_test
```
Result will be writed in `./result.txt`

# ASpT
0. already install icc in before step.
1. compile SpMM and run
```bash
cd $AUTOSPARSE_HOME/baseline/ASpT/SpMM_KNL
make
bash run_aspt_spmm_sp.sh
```
result will write to `./result`
2. compile SDDMM and run
```bash
cd $AUTOSPARSE_HOME/baseline/ASpT/SDDMM_KNL
make
bash run_aspt_sddmm_sp.sh
```
result will write to `./result`