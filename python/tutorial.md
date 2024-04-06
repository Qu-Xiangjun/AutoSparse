编译的时候，记得在Makefile中修改@cd build; cmake -DUSE_ICC=OFF .. 是否开启英特尔的ICPC编译器，否则使用GNU的GCC

export AUTOSPARSE_HOME=/home/qxj/AutoSparse
export PYTHONPATH=$AUTOSPARSE_HOME/python:$PYTHONPATH