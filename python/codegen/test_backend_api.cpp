// #include <iostream>
// #include <algorithm>
// #include <vector>
// #include <string.h>

// #include "backend_api.hpp"

// using namespace std;

// int main()
// {
//     string computation_desc = "3 A 1 2 i 64 0 k 128 1 B 0 2 k 128 0 j 256 0 C 0 2 i 64 0 j 256 0 float";
//     string schedule = "3 i 2 i0 8 i1 8 k 2 k0 2 k1 64 j 2 j0 32 j1 8 1 A 4 k1 i1 k0 i0 A 4 i0 3 i1 2 k0 2 k1 1 1 j0 2 j00 4 j01 8 7 k1 i1 k0 i0 j00 j01 j1 i1 None None 0 12 64";
//     vector<string> filepaths;
//     filepaths.push_back("/home/qxj/AutoSparse/dataset/demo_dataset/__test_matrix.csr");

//     BackEndAPI device(computation_desc, filepaths);
//     double excute_time = device.Compute(schedule, 100, 200);

//     cout << excute_time << endl;
//     return 0;
// }