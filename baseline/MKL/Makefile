iccmkl:
	icpc -std=c++17 -O3 -march=native -qopenmp -DICC  mkl_test.cc -o ./bin/mkl_test -lm -ldl -I"/opt/intel/oneapi/mkl/2024.1/include" -I. -Wl,--start-group "/opt/intel/oneapi/mkl/2024.1/lib/libmkl_intel_lp64.a" "/opt/intel/oneapi/mkl/2024.1/lib/libmkl_intel_thread.a" "/opt/intel/oneapi/mkl/2024.1/lib/libmkl_core.a" -Wl,--end-group -pthread -m64 -parallel -dynamic -Ofast -simd -vec -qopt-prefetch=5 -qopt-prefetch-distance=0,4

icc: iccmkl