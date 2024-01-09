#ifndef EXCUTION_MANAGE_HPP
#define EXCUTION_MANAGE_HPP

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <map>
#include <algorithm>
#include <iomanip>
#include <dlfcn.h>

#include "tensor.hpp"

using namespace std;
using namespace std::chrono_literals;


/* Define aliases for function Pointers. */
typedef int (*compute1)(taco_tensor_t *A, taco_tensor_t *B);
typedef int (*compute2)(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C);
typedef int (*compute3)(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, 
                        taco_tensor_t *D);
typedef int (*compute4)(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, 
                        taco_tensor_t *D, taco_tensor_t *E);

struct compute_func_t
{
    compute1 func1;
    compute2 func2;
    compute3 func3;
    compute4 func4;
};


class ExecutionManager
{
public:
    Tensor *lh_tensor;                  // The tensor of equotion left handle.
    string lh_tensor_name;              // The tensor name of eqution left.
    map<string, Tensor *> rhs_tensor;   // The tensors of all equotion right handles.
    compute_func_t compute_func;        // The compute func pointer, which will reinterpret
                                        // func pointer and pointe to dynamic lib func.
    vector<string> vars;                // All the axes name vector, which will apply lsplit,
                                        // lreorder, fsplit, freorder. And the vector indicate
                                        // final loop order after final reorder.
    vector<string> vars_rst;            // vars vectors' backup for using in reset.
    vector<string> freorder_vars;       // The format reorder scheduler result.
    bool is_lreorder;                   // Whether apply lreorder.
    vector<pair<vector<string>, int> > lsplit_record;
                                        // Record loop split schedule application.
                                        // First vector contains three elements, containing var, 
                                        // outer_var and inner_var. Second is the split factor.
    map<string, string> parallel_vars;  // Parallel scheduler record, first indicate axis name,
                                        // and second is taco schedule command field `hardware`.
    map<string, int> unroll_vars;       // First is unroll axis var name and second indicates
                                        // unroll factor.
    map<string, string> precompute_vars;// Precompute the axes. First is var, and second is
                                        // precomputed expression. 
    
    vector<float> corret_val;           // Correct result, which store in running and using 
                                        // to verify other optional computation.
    string taco_command;                // Compile the taco_command
    bool compile_success;               // Compile result.
    string global_kernel;               // Record the kernel tensor expression.
    

public:
    ExecutionManager() 
    {         
        lh_tensor           = nullptr;
        is_lreorder         = false;
        compile_success     = false;
        compute_func.func1  = nullptr;
        compute_func.func2  = nullptr;
        compute_func.func3  = nullptr;
        compute_func.func4  = nullptr;
    }
	ExecutionManager(vector<float> &val) : corret_val(val)
    {         
        lh_tensor           = nullptr;
        is_lreorder         = false;
        compile_success     = false;
        compute_func.func1  = nullptr;
        compute_func.func2  = nullptr;
        compute_func.func3  = nullptr;
        compute_func.func4  = nullptr;
    }
    ~ExecutionManager()
	{
		delete lh_tensor;
		for (auto &it : rhs_tensor)
		{
			delete it.second;
		}
	}

    /* Reset all the schedule and format optimization. */
    void reset_all()
    {
        compute_func.func1 = nullptr;
        compute_func.func2 = nullptr;
        compute_func.func3 = nullptr;
        compute_func.func4 = nullptr;

        is_lreorder = false;
        vars = vars_rst;
        freorder_vars.clear();
        lsplit_record.clear();
        parallel_vars.clear();
        unroll_vars.clear();
        precompute_vars.clear();
        taco_command.clear();
        compile_success = false;
        global_kernel.clear(); 
        
        lh_tensor->reset();
        for (auto &it : rhs_tensor)
        {
            it.second->reset();
        }
    }

    /* Add sparse tensor. */
    void add_tensor(
        string tensor_name, vector<FormatInfo> &tensor_format, 
        vector<pair<uint64_t, float>> &coo, bool is_lhs
    )
    {
        Tensor *tensor = new Tensor(coo, tensor_format);
        if (is_lhs) lh_tensor = tensor, lh_tensor_name = tensor_name;
        else rhs_tensor[tensor_name] = tensor;

        for (auto f : tensor_format)
        {
            // Add axis var name.
            if (find(vars.begin(), vars.end(), f.var) == vars.end())
            {
                vars.push_back(f.var);
                vars_rst.push_back(f.var);
            }
        }
    }

    /* Add dense tensor. */
    void add_tensor(
        string tensor_name, vector<FormatInfo> &tensor_format, 
        vector<float> &dense, bool is_lhs
    )
    {
        Tensor *tensor = new Tensor(dense, tensor_format);
        if (is_lhs) lh_tensor = tensor, lh_tensor_name = tensor_name;
        else rhs_tensor[tensor_name] = tensor;

        for (auto f : tensor_format)
        {
            // Add axis var name.
            if (find(vars.begin(), vars.end(), f.var) == vars.end())
            {
                vars.push_back(f.var);
                vars_rst.push_back(f.var);
            }
        }
    }

    /* Get a tensor handle by tensor var name. */
    Tensor &get_tensor(string tensor_name)
    {
        if (tensor_name == lh_tensor_name) return *lh_tensor;
        auto t = rhs_tensor.find(tensor_name);
        if (t != rhs_tensor.end()) return *(t->second);

        throw std::runtime_error("[ERROR] Tensor not found: " + tensor_name);
    }

    /* Print all the tensor info */
    void print_all()
    {
		cout << "############## left handle ############## "<< endl;
		cout << ">>>>>>>>>>>>>> " << lh_tensor_name << " <<<<<<<<<<<<<<" << endl;
		cout << lh_tensor->print_format();
		
        cout << "############## right handle ############## "<< endl;
		for (auto &it : rhs_tensor)
		{
			cout << ">>>>>>>>>>>>>> " << it.first << " <<<<<<<<<<<<<<" << endl;
			cout << it.second->print_format();
		}
    }

    /* Print one tensor */
    string print_tensor(string tensor_name)
	{
        if(lh_tensor_name == tensor_name) 
            return lh_tensor->print_format();
		auto t = rhs_tensor.find(tensor_name);
		if (t != rhs_tensor.end())
		{
			return t->second->print_format();
		}
        throw std::runtime_error("[ERROR] Tensor not found: " + tensor_name);
	}

    //////////////////////////////////////
    // Scheduler apply.
    //////////////////////////////////////

    /**
     * Split the axis for strorage format in tensor. And all the tensor related with
     * the axis will apply fsplit operation. 
     */ 
    void fsplit(string var, string outer_var, string inner_var, int factor)
    {
        auto it = find(vars.begin(), vars.end(), var);
        if (it == vars.end()) // The axis var don't exist.
        {
            throw std::runtime_error("[ERROR][fsplit] var not found: " + var);
        }
        // Replace new axes var name.
        int index = distance(vars.begin(), it);
        vars[index] = outer_var;
        vars.insert(vars.begin() + index + 1, inner_var);

        if (lh_tensor->is_var_exist(var))
        {
            lh_tensor->split(var, outer_var, inner_var, factor);
        }
        for (auto &it : rhs_tensor)
        {
            if (it.second->is_var_exist(var)) 
            {
                it.second->split(var, outer_var, inner_var, factor);
            }
        }
    }

    /* Change a tensor's storage format axes order. */
    void freorder(string tensor_name, vector<string> reordered_vars)
    {
        if (lh_tensor_name == tensor_name)
        {
            lh_tensor->reorder(reordered_vars);
            freorder_vars = reordered_vars;
            return ;
        }
        else 
        {
            auto t = rhs_tensor.find(tensor_name);
            if (t != rhs_tensor.end())
            {
                t->second->reorder(reordered_vars);
                freorder_vars = reordered_vars;
                return ;
            }
        }
        throw std::runtime_error("[ERROR][freorder] Tensor not found: " + tensor_name);
    }

    /* Change tensor's axis mode. */
    void fmode(string tensor_name, string var, mode_type mode)
    {
        if (lh_tensor_name == tensor_name) 
        {
            lh_tensor->mode(var, mode);
            return ;
        }
        else
        {
            auto t = rhs_tensor.find(tensor_name);
            if (t != rhs_tensor.end())
            {
                t->second->mode(var, mode);
                return ;
            }
        }
        throw std::runtime_error("[ERROR][fmode] Tensor not found: " + tensor_name);
    }

    /* Split axis with factor for loop, which don't change storage format. */
    void lsplit(string var, string outer_var, string inner_var, int factor)
    {
        auto it = find(vars.begin(), vars.end(), var);
        if (it == vars.end()) // The axis var don't exist.
        {
            throw std::runtime_error("[ERROR][lsplit] var not found: " + var);
        }
        // Replace new axes var name.
        int index = distance(vars.begin(), it);
        vars[index] = outer_var;
        vars.push_back(inner_var);

        vector<string> temp_vars = {var, outer_var, inner_var};
        lsplit_record.push_back({temp_vars, factor});
    }

    /* Reorder all the axis of loop, which don't change storage format. */
    void lreorder(vector<string> &reordered_vars)
    {
        if(reordered_vars.size() != vars.size())
        {
            throw std::runtime_error(
                "[ERROR][lreorder] The reorder var vector must contain all the axes."
            );
        }
        for (string var : reordered_vars)
        {
            auto it = find(vars.begin(), vars.end(), var);
            if (it == vars.end()) // The axis var don't exist.
            {
                throw std::runtime_error("[ERROR][lreorder] var not found: " + var);
            }
        }
        vars = reordered_vars;
        is_lreorder = true;
    }

    /* parallelize a loop axis. */
    void parallelize(string var, string hardware = "CPUThread")
    {
        auto it = find(vars.begin(), vars.end(), var);
        if (it == vars.end()) // The axis var don't exist.
        {
            throw std::runtime_error("[ERROR][parallelize] var not found: " + var);
        }
        auto it2 = unroll_vars.find(var);
        if (it2 != unroll_vars.end()) // Conflict.
        {
            throw std::runtime_error("[ERROR][parallelize] var conflict with unroll: " + var);
        }
        parallel_vars[var] = hardware;
    }

    /* Unroll a loop axis. */
    void unroll(string var, int factor)
    {
        auto it = find(vars.begin(), vars.end(), var);
        if (it == vars.end()) // The axis var don't exist.
        {
            throw std::runtime_error("[ERROR][unroll] var not found: " + var);
        }
        auto it2 = parallel_vars.find(var);
        if (it2 != parallel_vars.end()) // Conflict.
        {
            throw std::runtime_error("[ERROR][unroll] var conflict with parallel_vars: " + var);
        }
        unroll_vars[var] = factor;
    }
    
    void vectorize(string var)
    {
        parallelize(var, "CPUVector");
    }

    void precompute(string var, string expr = "")
    {
        auto it = find(vars.begin(), vars.end(), var);
        if (it == vars.end()) // The axis var don't exist.
        {
            throw std::runtime_error("[ERROR][precompute] var not found: " + var);
        }
        precompute_vars[var] = expr;
    }

    //////////////////////////////////////
    // Build the taco program and running.
    //////////////////////////////////////
    /**
     * Generate taco command about sparse tensor program description.
     * Examples
     * --------
     * `"C(i1,i0,j)=A(i1,k1,k0,i0)*B(k1,k0,j)" -f=C:ddd:0,1,2 -f=A:ssds:0,1,2,3 
     * -f=B:ddd:0,1,2 -t=C:float -t=A:float -t=B:float`
     */
    string gen_command()
    {
        string kernel = "\"";
        string format = "";
        string pricision = "";

        /* left handle tensor */ 
        format += " -f=" + lh_tensor_name + ":";
        kernel += lh_tensor_name + "(";
        pricision += " -t=" + lh_tensor_name + ":float";

        vector<FormatInfo> lh_format = lh_tensor->get_format();
        string format_order = "";
        for (int rank = 0; rank < lh_format.size(); rank++)
        {
            format_order += to_string(rank) + ",";
            switch (lh_format[rank].mode)
            {
            case COMPRESSED:
                format += "s";
                break;
            case COMPRESSED_NU:
                format += "u";
                break;
            case SINGLETON:
                format += "q";
                break;
            case SINGLETON_NU:
                format += "c";
                break;
            default: 
                /* UNCOMPRESSED axis don't need indices info */
                format += "d";
                break;
            }
            kernel += lh_format[rank].var;
            if (rank == lh_format.size() - 1)
				kernel += ")";
			else
				kernel += ",";
        }
        format_order.pop_back();
        format += ":" + format_order;
        kernel += "=";

        /* Right handle tensor */
        for (auto &it : rhs_tensor)
        {
            format += " -f=" + it.first + ":";
            kernel += it.first + "(";
            pricision += " -t=" + it.first + ":float";

            vector<FormatInfo> rh_format = it.second->get_format();
            format_order = "";
            for (int rank = 0; rank < rh_format.size(); rank++)
            {
                format_order += to_string(rank) + ",";
                switch (rh_format[rank].mode)
                {
                case COMPRESSED:
                    format += "s";
                    break;
                case COMPRESSED_NU:
                    format += "u";
                    break;
                case SINGLETON:
                    format += "q";
                    break;
                case SINGLETON_NU:
                    format += "c";
                    break;
                default: 
                    /* UNCOMPRESSED axis don't need indices info */
                    format += "d";
                    break;
                }
                kernel += rh_format[rank].var;
                if (rank == rh_format.size() - 1)
                    kernel += ")";
                else
                    kernel += ",";
            }
            format_order.pop_back();
            format += ":" + format_order;
			kernel += "*"; // TODO: 这里计算负只能是乘法了？
        }

        kernel.pop_back(); // Discard extra '*'.
        kernel += "\"";

        global_kernel = kernel;
        
        return kernel + format + pricision;
    }

    /**
     * Generate taco command about schedule.
     * Examples
     * --------
     * `-s="bound(j,jb,256, MaxExact)" -s="bound(k0,k0b,4, MaxExact)" 
     * -s="reorder(i1,k1,k0b,i0,jb)" -s="parallelize(i1,CPUThread,NoRaces)`
     */
    string gen_schedule()
    {
        string schedules = "";

        map<string, bool> var_is_compressed; // Whether the var isn't only about uncompressed.
        map<string, int> dimensions; // Every axis' length.
        for (auto &f : lh_tensor->get_format())
        {
            // Whether first see the axis var.
            if(var_is_compressed.find(f.var) == var_is_compressed.end())
            {
                var_is_compressed[f.var] = (f.mode == UNCOMPRESSED);
                dimensions[f.var] = f.dimension;
            }
            else // map had added the axis var.
            {
                var_is_compressed[f.var] &= (f.mode == UNCOMPRESSED);
            }
        }
        for (auto &rh_tensor : rhs_tensor)
        {
            for (auto &f : rh_tensor.second->get_format())
            {
                // Whether first see the axis var.
                if(var_is_compressed.find(f.var) == var_is_compressed.end())
                {
                    var_is_compressed[f.var] = (f.mode == UNCOMPRESSED);
                    dimensions[f.var] = f.dimension;
                }
                else // map had added the axis var.
                {
                    var_is_compressed[f.var] &= (f.mode == UNCOMPRESSED);
                }
            }
        }

        // lsplit.
        for (int i = 0; i < lsplit_record.size(); i++)
        {
            string var = lsplit_record[i].first[0];
            string outer_var = lsplit_record[i].first[1];
            string inner_var = lsplit_record[i].first[2];
            int factor = lsplit_record[i].second;

            if(var_is_compressed[var]) // Change the axis record.
            {
                var_is_compressed[var] = false;
                var_is_compressed[outer_var] = true;
                var_is_compressed[inner_var] = true;
                dimensions[outer_var] = roundup(dimensions[var], factor);
                dimensions[inner_var] = factor;
            }

            schedules += " -s=\"split(" + var + "," + outer_var + "," + inner_var;
            schedules += "," + to_string(factor) + ")\"";
        }

        /* Add bound for uncompressed axis, that is precondition of some schedules. */
        for (auto &it : var_is_compressed)
        {
            if (it.second) // UNCOMPRESSED or the axis had not changed by lsplit.
            {
                // Notice the var will rename to {name}Bound.
                schedules += " -s=\"bound(" + it.first + "," + it.first + "Bound,";
                schedules += to_string(dimensions[it.first]) + ",MaxExact)\"";
            } 
        }

        // reorder
        if (is_lreorder)
        {
            schedules += " -s=\"reorder(";
            for (int i = 0; i < vars.size(); i++)
            {
                if (var_is_compressed[vars[i]]) schedules += vars[i] + "Bound,";
                else schedules += vars[i] + ",";
            }
            schedules.pop_back(); // Discard extra ','.
            schedules += ")\"";
        }
        

        // precompute
        for (auto &it : precompute_vars)
        {
            if (global_kernel.empty())
            {
                throw std::runtime_error(
                    "[ERROR] Kernel of tensor expression is empty."
                );
            }
            int equalPos = global_kernel.find('=');
            string afterEqual = global_kernel.substr(equalPos + 1);
            afterEqual.pop_back();
            schedules += " -s=\"precompute(" + afterEqual + ",";
            if(var_is_compressed[it.first])
                schedules += it.first + "Bound," + it.first + "Bound)\"";
            else
                schedules += it.first + "," + it.first + ")\"";
        }

        // parallel
        for (auto &it : parallel_vars)
        {
            string var = it.first;
            bool is_reduction_var = !(lh_tensor->is_var_exist(var));
            if(var_is_compressed[var])
                schedules += " -s=\"parallelize(" + var + "Bound,";
            else    
                schedules += " -s=\"parallelize(" + var + ",";
            schedules += it.second + ",";
            schedules += (is_reduction_var ? "Atomics" : "NoRaces");
            schedules += ")\"";
        }

        // unroll
        for (auto &it : unroll_vars)
        {
            int unroll_factor = min(it.second, dimensions[it.first]);
            if(var_is_compressed[it.first])
                schedules += " -s\"unroll(" + it.first + "Bound," + to_string(unroll_factor) + ")\"";
            else
                schedules += " -s\"unroll(" + it.first + "," + to_string(unroll_factor) + ")\"";
        }

        return schedules;
    }

    /**
     * Gnerate lower compution kernel, and compile it.
     * Prameters
     * ---------
     * arg1 : num_thread
     *   Set omp num thread, which usually is number of core.
     * arg2 : chunk_size
     *   Set omp schedule's chunk size.
     */
    string compile(int num_thread, int chunk_size)
    {
        omp_set_num_threads(num_thread);
		omp_set_schedule(omp_sched_dynamic, chunk_size);

        // Pack all the tensor
        lh_tensor->pack();
        for (auto &it : rhs_tensor) it.second->pack();

        char *env_val = getenv("AUTOSPARSE_HOME");
        if (env_val == NULL)
        {
            throw std::runtime_error(
                "[ERROR][compile] Environment variable AUTOSPARSE_HOME not defined"
            );
        }

        // delete last kernel
        ifstream kernel_file(string("./taco_kernel.c").c_str());
        if(kernel_file.good())
        {
            system("rm -f ./taco_kernel.c");
        }

        taco_command = string(env_val) + "/taco/build/bin/taco ";
        taco_command += gen_command();
        taco_command += gen_schedule();
        taco_command += " -write-compute=taco_kernel.c";

        // Add the header to kernel.c
        string header = "#include <stdint.h> \\n"
						"typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;\\n"
						"typedef struct {\\n"
						"  int32_t      order;\\n"
						"  int32_t*     dimensions;\\n"
						"  int32_t      csize;\\n"
						"  int32_t*     mode_ordering;\\n"
						"  taco_mode_t* mode_types;\\n"
						"  uint8_t***   indices;\\n"
						"  uint8_t*     vals;\\n"
						"  int32_t      vals_size;\\n"
						"} taco_tensor_t;\\n";
		string taco_header_command = "sed -i '1s/^/" + header + "/' taco_kernel.c";
        cout << taco_command << endl;
		compile_success = executeCommand(taco_command);
        // cout<<taco_header_command<<endl;
		compile_success = executeCommand(taco_header_command);

        // Compile kernel.c
        string cc_command;
		#ifdef ICC // `-DICC` in compile command
		cc_command = "icc -march=native -mtune=native -O3 -ffast-math -qopenmp -fPIC -shared taco_kernel.c -o taco_kernel.so -lm";
		#elif GCC  // `-DGCC` in compile command
		cc_command = "gcc -march=native -mtune=native -O3 -fopenmp -ffast-math -fPIC -shared taco_kernel.c -o taco_kernel.so -lm";
		#endif
		compile_success = executeCommand(cc_command);

        return taco_command;
    }

    /**
     * Run the compution function and verify result.
     * Parameters
	 * ----------
	 * arg1 : warm 
	 *   Warming excution times.
	 * arg2 : round 
	 *   Test excution times.
	 * arg3 : verify_res bool
	 * 	 Verify the excution result.
     * arg4 : verify bool
	 * 	 Whether verify the excution result.
	 * arg5 : store
	 * 	 Whether store current excution result as correct result.
	 * arg6 : avg_test
     *   The test time select average or middle number.
     * arg7 : time_limit
     *   Execution program time limit.
	 * Return
	 * ------
	 * ret : elapsed_time
     *   Excution times (ms).
     */
    double run(
        int warm, int round,bool &verify_res, bool verify = true, 
        bool store = false, bool avg_test = true, double time_limit = 1000000.0
    )
    {
        verify_res = false;
        double elapsed_time = 0.;
        vector<double> elapsed;

        if (!compile_success) 
        {
            return -1;
        }

        /* Open the dynamic lib, Take a func pointer point to kernel compute function in kernel.c */ 
        void *lib_handle = dlopen("./taco_kernel.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib_handle)
        {
            stringstream ss;
            ss << "[ERROR][Compile] DLOPEN - " << dlerror() << endl;
            throw std::runtime_error(ss.str());
        }
        switch (rhs_tensor.size())
        {
        case 1:
            compute_func.func1 = (compute1)dlsym(lib_handle, "compute");
            break;
        case 2:
            compute_func.func2 = (compute2)dlsym(lib_handle, "compute");
            break;
        case 3:
            compute_func.func3 = (compute3)dlsym(lib_handle, "compute");
            break;
        case 4:
            compute_func.func4 = (compute4)dlsym(lib_handle, "compute");
            break;
        default:
            throw std::runtime_error(
                "[ERROR][Compile] Don't support operator count."
            );
            break;
        }
        if (dlerror() != NULL)
		{
            stringstream ss;
            ss << "[ERROR][Compile] " << dlerror() << endl;
            throw std::runtime_error(ss.str());
		}
        
        /* Run the func. */ 
        vector<taco_tensor_t *> T;
        T.push_back(lh_tensor->T);
        for (auto &it : rhs_tensor)
        {
            T.push_back(it.second->T);
        }
        // fwrite2file((float *)T[0]->vals, T[0]->vals_size, "******* T0 *******");
        // fwrite2file((float *)T[1]->vals, T[1]->vals_size, "******* T1 *******");
        // fwrite2file((float *)(rhs_tensor["A"]->T_pos[1].data()), rhs_tensor["A"]->T_pos[1].size(), "******* T1 pos *******");
        // fwrite2file((float *)(rhs_tensor["A"]->T_crd[1].data()), rhs_tensor["A"]->T_crd[1].size(), "******* T1 crd *******");
        // fwrite2file((float *)T[2]->vals, T[2]->vals_size, "******* T2 *******");

        // Warmup.
        while (warm--)
        {
            auto t1 = Clock::now();
            switch (rhs_tensor.size())
            {
            case 1:
                compute_func.func1(T[0], T[1]);
                break;
            case 2:
                compute_func.func2(T[0], T[1], T[2]);
                break;
            case 3:
                compute_func.func3(T[0], T[1], T[2], T[3]);
                break;
            case 4:
                compute_func.func4(T[0], T[1], T[2], T[3], T[4]);
                break;
            default:
                break;
            }
            elapsed_time = compute_clock(Clock::now(), t1);
			if (elapsed_time > time_limit)
				goto verifyStart;
        }

        // Test.
        while (round--)
        {
            auto start_time = Clock::now();
            switch (rhs_tensor.size())
            {
            case 1:
                start_time = Clock::now();
                compute_func.func1(T[0], T[1]);
                elapsed.push_back(compute_clock(Clock::now(), start_time));
                break;
            case 2:
                start_time = Clock::now();
                compute_func.func2(T[0], T[1], T[2]);
                elapsed.push_back(compute_clock(Clock::now(), start_time));
                break;
            case 3:
                start_time = Clock::now();
                compute_func.func3(T[0], T[1], T[2], T[3]);
                elapsed.push_back(compute_clock(Clock::now(), start_time));
                break;
            case 4:
                start_time = Clock::now();
                compute_func.func4(T[0], T[1], T[2], T[3], T[4]);
                elapsed.push_back(compute_clock(Clock::now(), start_time));
                break;
            default:
                break;
            }
        }
        if (avg_test)
        {
            for(int tt = 0; tt < elapsed.size(); tt++)
				elapsed_time += elapsed[tt];
			elapsed_time /= elapsed.size();
        }
        else // Middle number of time.
        {
            sort(elapsed.begin(), elapsed.end());
			elapsed_time = elapsed[elapsed.size() / 2];
        }

        // Verify and Store
        verifyStart:
        if (!verify and !store){
            dlclose(lib_handle);
            return elapsed_time;
        }

        lh_tensor->fill_val(0);
        switch (rhs_tensor.size())
        {
        case 1:
            compute_func.func1(T[0], T[1]);
            break;
        case 2:
            // fwrite2file((float *)T[0]->vals, T[0]->vals_size, "******* T0 *******");
            // fwrite2file((float *)T[1]->vals, T[1]->vals_size, "******* T1 *******");
            // fwrite2file((float *)T[2]->vals, T[2]->vals_size, "******* T2 *******");
            compute_func.func2(T[0], T[1], T[2]);
            break;
        case 3:
            compute_func.func3(T[0], T[1], T[2], T[3]);
            break;
        case 4:
            compute_func.func4(T[0], T[1], T[2], T[3], T[4]);
            break;
        default:
            break;
        }
        
        if (verify && corret_val.size())
        {
            vector<float> &res = lh_tensor->get_vals();
            bool flag = true;
            for (int i = 0; i < res.size(); i++)
            {
                if (abs(corret_val[i] - res[i] > 0.01))
                {
                    flag = false;
                    break;
                }
            }
            verify_res = flag;
        }

        if (store)
        {
            corret_val.clear();
            float *res = (float*)(T[0]->vals);
            int res_size = T[0]->vals_size;
			// fwrite2file(res, res_size, "******* T0 *******");
            corret_val.resize(res_size);
            #pragma omp parallel for
            for (int i = 0; i < res_size; i++)
				corret_val[i] = res[i];
        }

        dlclose(lib_handle);
        return elapsed_time;
    }

};

#endif
