# AutoSparse
Source-2-Source automatically generate sparse tensor program in CPU.

# Requirement
You can use gcc or intel c++ compiler (icc, icpc) to compile `AutoSparse` framework.

Python third-party libraries list in './requirements.txt', which need to pip install.

# Installation
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
sudo cp /opt/intel/oneapi/compiler/2021.4.0/env/vars.sh /opt/intel/oneapi/compiler/2021.4.0/env/iccvars.sh
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
cd $AUTOSPARSE_HOME/python
make
```
> Note: if you use `gcc`, you will changed `-DUSE_ICC=ON` in `$AUTOSPARSE_HOOME/poython/Makefile` to be `OFF`

1. How to use
Please see `$AUTOSPARSE_HOOME/poython/tutorial.py` and `$AUTOSPARSE_HOOME/poython/test`