// #include <iostream>
// #include <algorithm>
// #include <vector>
// #include <string.h>

// #include "backend_api.hpp"

// using namespace std;

// int main()
// {
//     string computation_desc = "3 A 1 2 i 4096 0 k 8192 1 B 0 2 k 8192 0 j 256 0 C 0 2 i 4096 0 j 256 0 float";
//     string schedule = "3 i 2 i0 2048 i1 2 k 2 k0 1024 k1 8 j 2 j0 4 j1 64 1 A 4 i0 i1 k0 k1 A 4 i0 1 i1 0 k0 1 k1 0 0 6 i0 i1 k0 j0 k1 j1 i0 None None 0 12 64";
//     vector<string> filepaths;
//     filepaths.push_back("/home/qxj/AutoSparse/dataset/demo_dataset/nemspmm1_16x4_0.csr");

//     BackEndAPI device(computation_desc, filepaths);

//     cout << device.origin_time << endl;

//     double excute_time = device.Compute(schedule, 100, 200);

//     cout << excute_time << endl;
//     return 0;
// }

// // icpc -std=c++17 -O3 -march=native -qopenmp -DICC -DNUMCORE=${NUMCORE} test_backend_api.cpp -o ./bin/test_backend_api -lm -ldl