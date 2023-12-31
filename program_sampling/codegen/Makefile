NUMCORE=$$(nproc --all)

gccspmm :	
	g++ -std=c++17 -O3 -march=native -fopenmp -DGCC -DNUMCORE=${NUMCORE} spmm.cc -o spmm -lm -ldl
iccspmm :	
	icpc -std=c++17 -O3 -march=native -qopenmp -DICC -DNUMCORE=${NUMCORE} spmm.cc -o spmm -lm -ldl 

gcctest_kernel :	
	g++ -std=c++17 -O3 -march=native -fopenmp -DGCC -DNUMCORE=${NUMCORE} test_kernel.cc -o test_kernel -lm -ldl
icctest_kernel :	
	icpc -std=c++17 -O3 -march=native -qopenmp -DICC -DNUMCORE=${NUMCORE} test_kernel.cc -o test_kernel -lm -ldl

gcc : gccspmm gcctest_kernel
icc : iccspmm icctest_kernel