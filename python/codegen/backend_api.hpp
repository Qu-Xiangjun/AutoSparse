#include <iostream>
#include <string>
#include <random>
#include <set>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <map>
#include <vector>
#include <fstream>
#include <cmath>
#include <assert.h>

#include "tensor.hpp"
#include "execution_manager.hpp"
#include "time.hpp"
#include "utils.hpp"

// #define __DEBUG__

using namespace std;

default_random_engine generator(
    chrono::system_clock::now().time_since_epoch().count());
uniform_real_distribution<float> uniform(-1.0, 1.0);


/**
 * Read CSR format sparse matrix from .csr file
 */
void ReadCSR2D(string filepath, Compressed_Coo& coo)
{
    coo.clear();

    int num_row, num_col, num_nonzero;
    fstream csr(filepath);
    // Load mnatrix info
	csr.read((char *)&num_row, sizeof(int));
	csr.read((char *)&num_col, sizeof(int));
	csr.read((char *)&num_nonzero, sizeof(int));

#ifdef __DEBUG__
    cout << "Read csr file from " << filepath << " with matirx ";
    cout << "NUMROW: " << num_row << " / NUMCOL: " << num_col;
    cout << " / NNZ: " << num_nonzero << endl;
#endif

    // Load matrix
	vector<int> A_crd(num_nonzero);
	vector<int> A_pos(num_row + 1);
	vector<float> A_val(num_nonzero);
	for (int i = 0; i < num_row + 1; i++)
	{
		int data;
		csr.read((char *)&data, sizeof(data));
		A_pos[i] = data;
	}
	for (int i = 0; i < num_nonzero; i++)
	{
		int data;
		csr.read((char *)&data, sizeof(data));
		A_crd[i] = data;
		A_val[i] = 1.0; // TODO: Now only consider sparse matrix layout.
	}
	csr.close();

	coo.resize(num_nonzero);
	int col_digit2 = (int)(ceil(log2(num_col)));
	#pragma omp parallel for schedule(dynamic, 4)
	for (int i = 0; i < num_row; i++)
	{
		for (int j = A_pos[i]; j < A_pos[i + 1]; j++)
		{
			uint64_t i_ = (((uint64_t)i) << col_digit2);
			uint64_t t = i_ | A_crd[j];
			coo[j] = {t, A_val[j]}; 
		}
	}
}

void MSplitHelp(ExecutionManager& M, bool is_fsplit, string axis_name,
    vector<string>& new_axes_name, vector<int>& new_axes_size)
{
    auto SPLIT = [](ExecutionManager& M,
        bool is_fsplit, string& axis_name, 
        string& outer_var, string& inner_var, int factor)
    {
        if (is_fsplit) M.fsplit(axis_name, outer_var, inner_var, factor);
        else M.lsplit(axis_name, outer_var, inner_var, factor);
    };

    assert (new_axes_name.size() == new_axes_size.size());
    assert (new_axes_name.size() > 1);

#ifdef __DEBUG__
    cout << "In [MSplitHelp] " << (is_fsplit ? "format split" : "loop split");
    cout << " for axis " << axis_name << endl;
#endif

    int num = new_axes_name.size();
    string current_axis = axis_name;
    for (int i = 1; i < num; i++)
    {
        int factor = 1;
        string new_axis_name = "";
        for (int j = i; j < num; j++)
        {
            factor *= new_axes_size[j];
            new_axis_name += new_axes_name[j];
        }
        SPLIT(M, is_fsplit, current_axis,
                new_axes_name[i-1], new_axis_name, factor);
#ifdef __DEBUG__
    cout << "Split " << current_axis << " to " << new_axes_name[i-1];
    cout << " and " << new_axis_name << " with factor " << factor << endl; 
#endif
        current_axis = new_axis_name;
    }
    assert(current_axis == new_axes_name[num-1]);
}

class BackEndAPI
{
public:
    ExecutionManager M;
    float origin_time; // Run with no schedule in build.

    /**
     * Construct Excution Manager.
     * @param init_info:
     *      "tensor_count, 
     *      tensor_list[tensor_name, is_sparse, axes_count, 
     *                  axes_list[axis_name, size, mode]],
     *      dtype"
     * @param filepaths:
     *      All sparse tensor data file path string.
     */
    BackEndAPI(string init_info, vector<string> filepaths)
    {
        assert (init_info.size());
        
        stringstream ss(init_info);

        int tensor_count;
        ss >> tensor_count;
        assert (tensor_count);
        int sparse_count = 0;
        for (int i = 0; i < tensor_count; ++i)
        {
            string tensor_name;
            int is_sparse;
            int axes_count;
            ss >> tensor_name >> is_sparse >> axes_count;
            vector<FormatInfo> tensor_format;
            for (int j = 0; j < axes_count; ++j)
            {
                string axis_name;
                int axis_size;
                int mode;
                ss >> axis_name >> axis_size >> mode;
                tensor_format.push_back(
                    {axis_name, axis_size, mode_type_array[mode]}
                );
            }
            bool is_lhs = (i == tensor_count - 1);
            if (is_sparse)
            {
                Compressed_Coo tmp_coo;
                assert (sparse_count < filepaths.size());
                ReadCSR2D(filepaths[sparse_count], tmp_coo);
                M.add_tensor(tensor_name, tensor_format, tmp_coo, is_lhs);
                sparse_count++;
            }
            else
            {
                int len = 1;
                for (int j = 0; j < axes_count; ++j) len *= tensor_format[j].dimension;
                vector<float> tensor_data(len, is_lhs ? 0 : 1);
                M.add_tensor(tensor_name, tensor_format, tensor_data, is_lhs);
            }
        }

#ifdef __DEBUG__
        M.print_all();
#endif

        M.reset_all();
        // Excute to save correct result
        M.parallelize(M.lh_tensor->format[0].var);
        M.compile(NUMCORE, 32);
        bool verify = false;
        origin_time = M.run(100, 100, verify, false, true);
#ifdef __DEBUG__
        cout << "Origin test time = " << origin_time << endl;
#endif
    }


    /**
     * Add format and schedule command. Run computation.
     * @param scheudle
     *      "fsplit_count, 
     *      fsplit_list[axis_name, new_axes_count, new_axes_list[axis_name, size]],
     *      sparse_tensor_count, 
     *      freorder_list[tensor_name, axes_size, axes_list[axis_name]], 
     *      fmode_list[tensor_name, axes_size, axes_list[axis_name, fmode]],
     *      lsplit_count, 
     *      lsplit_list[axis_name, new_axes_count, new_axes_list[axis_name, size]],
     *      lreorder_vars_count, lreorder_list[axis_name]
     *      parallize_axis_name, vectorize_axis_name, unroll_axis_name, unroll_factor, 
     *      thread_num, parchunk]"
     *      Notice: If the item of 'x_count' is 0, there will no next item of 
     *          'x_list'. 'dtype' can't be none. 'parallize' and 'vectorize'
     *          'unroll' item will fill None string if there have not action. 
     *           And 'unroll_factor' will be ignored if 'unroll' is None.
     * @param warm Warm times
     * @param round Test times.
     * @param time_policy Using average time test or middle or best time.
     * @return the function excution time
     */
    double Compute(string schedule, int warm = 10, 
                    int round = 50, string time_policy = "avg")
    {
        M.reset_all();
        stringstream ss(schedule);

        // fsplit
        int fsplit_count;
        ss >> fsplit_count;
        for (int i = 0; i < fsplit_count; i++)
        {
            string axis_name;
            int new_axes_count;
            ss >> axis_name >> new_axes_count;
#ifdef __DEBUG__
            cout << "Format split axis " << axis_name << " to ";
            cout << new_axes_count << "new axis" << endl;
#endif
            vector<string> new_axes_name(new_axes_count);
            vector<int> new_axes_size(new_axes_count);
            for (int j = 0 ; j < new_axes_count; j++)
                ss >> new_axes_name[j] >> new_axes_size[j];
            MSplitHelp(M, true, axis_name, new_axes_name, new_axes_size);
        }

        // freorder
        int sparse_count;
        ss >> sparse_count;
        for (int i = 0; i < sparse_count; i++)
        {
            string tensor_name;
            int axes_size;
            ss >> tensor_name >> axes_size;
            vector<string> freordered_vars(axes_size);
            for (int j = 0; j < axes_size; j++) ss >> freordered_vars[j];
            M.freorder(tensor_name, freordered_vars);
#ifdef __DEBUG__
            cout << "Format reorder " << tensor_name << ": ";
            for (int j = 0; j < axes_size; j++) cout << freordered_vars[j] << " ";
            cout << endl;
#endif
        }

        // fmode
        for (int i = 0; i < sparse_count; i++)
        {
            string tensor_name;
            int axes_size;
            ss >> tensor_name >> axes_size;
#ifdef __DEBUG__
            cout << "Format axis mode set for " << tensor_name << ": ";
#endif
            for (int j = 0; j < axes_size; j++)
            {
                string axis_name;
                int mode;
                ss >> axis_name >> mode;
                M.fmode(tensor_name, axis_name, mode_type_array[mode]);
#ifdef __DEBUG__
                cout << axis_name << "=" << mode << ", ";
#endif
            }
#ifdef __DEBUG__
            cout << endl;
#endif
        }

        // lsplit
        int lsplit_count;
        ss >> lsplit_count;
        for (int i = 0; i < lsplit_count; i++)
        {
            string axis_name;
            int new_axes_count;
            ss >> axis_name >> new_axes_count;
#ifdef __DEBUG__
            cout << "Loop split axis " << axis_name << " to ";
            cout << new_axes_count << "new axis" << endl;
#endif
            vector<string> new_axes_name(new_axes_count);
            vector<int> new_axes_size(new_axes_count);
            for (int j = 0 ; j < new_axes_count; j++)
                ss >> new_axes_name[j] >> new_axes_size[j];
            MSplitHelp(M, false, axis_name, new_axes_name, new_axes_size);
        }
        
        // lreorder
        int lreorder_vars_count;
        ss >> lreorder_vars_count;
        if (lreorder_vars_count)
        {
            vector<string> lreordered_vars(lreorder_vars_count);
            for (int i = 0; i < lreorder_vars_count; i++) ss >> lreordered_vars[i];
            M.lreorder(lreordered_vars);
#ifdef __DEBUG__
            cout << "Loop reorder: ";
            for (int i = 0; i < lreorder_vars_count; i++) 
                cout << lreordered_vars[i] << " ";
            cout << endl;
#endif
        }


        // parallel
        string parallize_axis_name;
        ss >> parallize_axis_name;
        if (parallize_axis_name != "None") M.parallelize(parallize_axis_name);
#ifdef __DEBUG__
        cout << "Parallel axis: " << parallize_axis_name << endl;
#endif

        // vectorize
        string vectorize_axis_name;
        ss >> vectorize_axis_name;
        if (vectorize_axis_name != "None") M.vectorize(vectorize_axis_name);
#ifdef __DEBUG__
        cout << "Vectorize axis: " << vectorize_axis_name << endl;
#endif

        // unroll
        string unroll_axis_name;
        int unroll_factor;
        ss >> unroll_axis_name >> unroll_factor;
        if (unroll_axis_name != "None") M.unroll(unroll_axis_name, unroll_factor);
#ifdef __DEBUG__
        cout << "Unrool axis: " << unroll_axis_name << endl;
#endif

        // OpenMP parameters
        int thread_num;
        int parchunk;
        ss >> thread_num >> parchunk;
#ifdef __DEBUG__
        cout << "OpenMP thread_num=" << thread_num << ", parchunk=" << parchunk << endl;
#endif

        // Compile
        try{
            M.compile(thread_num, parchunk);
        }
        catch (const invalid_argument& e)
        {
            string errorMsg = e.what();
            if (errorMsg.find("Format storage size overleaf") != string::npos) {
#ifdef __DEBUG__
                cout << "[ERROR][Tensor] Format storage size overleaf when pack." << endl;
#endif
                return -1.0;
            }
            else{
                cout << schedule << endl;
                
            }
        }

        // Run
        float test_time = -1;
        bool verify_res = true;
        try{
            test_time = M.run(warm, round, verify_res, true, false, time_policy, origin_time*2);
        }
        catch (const runtime_error& e)
        {
            cout << schedule << endl;
            throw;
        }

#ifdef __DEBUG__
        cout << "---------------- Run " << " -----------------" << endl;
        cout << "Excution time = " << test_time << ", Correct = " << verify_res << endl;
#endif
        if (verify_res) return test_time;
        return -1.0;
    }
};