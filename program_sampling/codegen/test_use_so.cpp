#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <sstream>

#include <dlfcn.h>

#include "tensor.hpp"

using namespace std;

typedef int (*compute2)(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B);
compute2 func2;

// void run()
// {

// }


int main()
{
    void *lib_handle = dlopen("./taco_kernel.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib_handle)
    {
        stringstream ss;
        ss << "[ERROR][Compile] DLOPEN - " << dlerror() << endl;
        throw std::runtime_error(ss.str());
    }
    func2 = (compute2)dlsym(lib_handle, "compute");
    taco_tensor_t *T1 = new taco_tensor_t;
    int num_rank = 2;
    T1->order = num_rank;
    T1->dimensions = new int32_t[num_rank];
    T1->mode_types = new taco_mode_t[num_rank];
    T1->indices    = new uint8_t**[num_rank]; 
    for (int rank = 0; rank < num_rank; rank++)
    {
        T1->dimensions[rank] = 32;
        T1->mode_types[rank] = taco_mode_dense;
        T1->indices[rank] = new uint8_t*[2];
        vector<float> T_vals(32*32, 1);
        T1->indices[rank][0] = (uint8_t *)(T_vals.data()); 
        T1->indices[rank][1] = (uint8_t *)(T_vals.data()); 
        T1->vals = (uint8_t *)(T_vals.data());
        T1->vals_size = T_vals.size();
    }
    func2(T1, T1, T1);
    return 0;
}