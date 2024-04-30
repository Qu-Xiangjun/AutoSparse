// #include <iostream>
// #include <algorithm>
// #include <vector>
// #include <string.h>

// #include "backend_api.hpp"

// using namespace std;

// int main()
// {
//     string computation_desc = "4 A 1 2 i 49152 0 j 49152 1 B 0 2 i 49152 0 k 256 0 C 0 2 k 256 0 j 49152 0 D 1 2 i 49152 0 j 49152 1 float";
//     string schedule = "2 i 2 i0 32768 i1 1 j 2 j0 128 j1 256 2 A 4 j0 i0 j1 i1 D 4 j0 i0 j1 i1 A 4 i0 1 i1 0 j0 1 j1 1 D 4 i0 1 i1 0 j0 1 j1 1 1 k 2 k0 128 k1 2 6 k0 j0 i0 j1 i1 k1 None None k0 128 12 1";
//     vector<string> filepaths;
//     filepaths.push_back("/home/qxj/AutoSparse/dataset/demo_dataset/cca.csr");
//     filepaths.push_back("/home/qxj/AutoSparse/dataset/demo_dataset/cca.csr");

//     BackEndAPI device(computation_desc, filepaths);

//     cout << device.origin_time << endl;

//     double excute_time = device.Compute(schedule, 5, 5);

//     cout << excute_time << endl;
//     return 0;
// }

// // icpc -std=c++17 -O3 -march=native -qopenmp -DICC -DNUMCORE=${NUMCORE} test_backend_api.cpp -o ./bin/test_backend_api -lm -ldl