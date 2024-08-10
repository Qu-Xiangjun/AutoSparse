# AutoSparse
Source-2-Source automatically generate high-performance sparse tensor program in CPU with storage format and schedule exploration. 

AutoSparse has been accepted as a regular paper by ICCD2024 for ["AutoSparse: AutoSparse: A Source-to-Source Format and Schedule Auto-Tuning Framework for Sparse Tensor Program "].

# Introductions
Sparse tensor computation plays a crucial role in modern deep learning workloads, and its expensive computational cost leads to a strong demand for high-performance operators. However, developing high-performance sparse operators is exceptionally challenging and tedious. Existing vendor operator libraries fail to keep pace with the evolving trends in new algorithms. Sparse tensor compilers simplify the development and optimization of operator, but existing work either requires significant engineering effort for tuning or suffers from limitations in search space and search strategies, which creates unavoidable cost and efficiency issues. 

In this paper, we propose AutoSparse, a source-to-source tuning framework that targets sparse format and schedule for sparse tensor program.  Firstly, AutoSparse designs a sparse tensor DSL based on dynamic computational graph at the front-end, and proposes a sparse tensor program computational pattern extraction and automatic design space generation scheme based on it. Second, AutoSparse's back-end designs an adaptive exploration strategy based on reinforcement learning and heuristic algorithm to find the optimal format and schedule configuration in a large-scale design space. Compared to prior work, developers using AutoSparse do not need to specify tuning design space relied on any compilation or hardware knowledge. we use the SuiteSparse dataset to compare with four state-of-the-art baselines, namely, the high-performance operator library MKL, the manually-based optimisation scheme ASpT, the auto-tuning-based framework TVM-S and WACO. The results demonstrate that AutoSparse achieves average speedups of 1.92-2.48$\times$, 1.19-6.34$\times$, and 1.47-2.23$\times$ for the SpMV, SpMM, and SDDMM operators, respectively.

# Tutorial
### Requirement
You can use gcc or intel c++ compiler (icc, icpc) to compile `AutoSparse` framework.

Python third-party libraries list in './requirements.txt', which need to pip install.

### Installation
0. Clone repo and set env var
   
```bash
git clone https://https://github.com/Qu-Xiangjun/AutoSparse.git
vim ~/.bashrc
export AUTOSPARSE_HOME=~/AutoSparse
```

1. Install intel c++ compiler
```bash
# Install guide in https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-0/install-with-command-line.html

# Intsall Intel oneAPI Base ToolKit
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18236/l_BaseKit_p_2021.4.0.3422.sh
sudo sh l_BaseKit_p_2021.4.0.3422.sh -a --silent --eula accept

# Install Intel oneAPI HPC ToolKit
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18211/l_HPCKit_p_2021.4.0.3347.sh
sudo sh l_HPCKit_p_2021.4.0.3347.sh -a --silent --eula accept

# Configure Env Var
vim ~/.bashrc
export INTEL_ONEAPI_ROOT=/opt/intel/oneapi
export PATH=$PATH:$INTEL_ONEAPI_ROOT/compiler/2021.4.0/env:$INTEL_ONEAPI_ROOT/compiler/2021.4.0/linux/bin/intel64

# Create iccvar.sh
sudo cp $INTEL_ONEAPI_ROOT/compiler/2021.4.0/env/vars.sh $INTEL_ONEAPI_ROOT/compiler/2021.4.0/env/iccvars.sh
# Verify installation
icc --version
icpc --version
which iccvars.sh
```

2. Install TACO
```bash
cd $AUTOSPARSE_HOME/taco
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

3. Install AutoSparse
```bash
export NUMCORE=$(nproc --all)
export PYTHONPATH=$AUTOSPARSE_HOME/python:$PYTHONPATH
cd $AUTOSPARSE_HOME/python
make
```
> Note: if you use `gcc`, you will changed `-DUSE_ICC=ON` in `$AUTOSPARSE_HOOME/poython/Makefile` to be `OFF`

### How to use
Please see `$AUTOSPARSE_HOOME/poython/tutorial.py` and `$AUTOSPARSE_HOOME/poython/test`