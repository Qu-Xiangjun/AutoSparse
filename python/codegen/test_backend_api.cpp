// #include <iostream>
// #include <algorithm>
// #include <vector>
// #include <string.h>

// #include "backend_api.hpp"

// using namespace std;

// int main()
// {
//     string computation_desc = "3 A 1 2 i 4096 0 k 8192 1 B 0 2 k 8192 0 j 256 0 C 0 2 i 4096 0 j 256 0 float";
//     string schedule = "2 k 2 k0 32 k1 256 i 2 i0 128 i1 32 1 A 4 i1 k1 i0 k0 A 4 i0 1 i1 2 k0 0 k1 2 1 j 2 j0 32 j1 8 6 i1 j1 k1 i0 j0 k0 i1 None None 0 12 32";
//     vector<string> filepaths;
//     filepaths.push_back("/home/qxj/AutoSparse/dataset/demo_dataset/nemspmm1_16x4_0.csr");

//     BackEndAPI device(computation_desc, filepaths);
//     double excute_time = device.Compute(schedule, 100, 200);

//     cout << excute_time << endl;
//     return 0;
// }