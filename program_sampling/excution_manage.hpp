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

#include "tensor.hpp"

using namespace std;
using namespace std::chrono_literals;


/* Debug func. */
void fwrite2file(float* val, int data_size, string title = "*******")
{
	ofstream outputFile("Debug_data.txt", ios::app);
	outputFile << title << endl;
	for(int tt = 0; tt < data_size; tt++) 
	{
		outputFile << fixed << setprecision(2) << val[tt] << endl;
	}
	outputFile << endl;
	outputFile.close();
	cout<< "[Debug] Debug data write successed." <<endl;
}

/* Safe func to excute shell command. */
bool executeCommand(const char* cmd) {
    FILE* pipe = popen(cmd, "r"); // Open a pipe to execute the command
    if (!pipe) 
    {
		cerr << "[ERROR][ExecuteCommand] " << cmd << endl;
	}
    char buffer[128];
    string result = "";
    while (!feof(pipe)) 
    {
        if (fgets(buffer, 128, pipe) != nullptr)
            result += buffer;
    }
    pclose(pipe);
	bool compile_success = 1; // Flag to indicate successful compilation.

	if (result.find("error") != string::npos || result.find("Error") != string::npos) 
    {
        cerr << "[ERROR][ExecuteCommand] " << result << endl;
		compile_success = 0; // Set flag to indicate compilation failure
    }
    return compile_success;
}

/* Define aliases for function Pointers. */
typedef int (*compute1)(taco_tensor_t *A, taco_tensor_t *B);
typedef int (*compute2)(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C);
typedef int (*compute3)(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, 
                        taco_tensor_t *D);
typedef int (*compute4)(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, 
                        taco_tensor_t *D, taco_tensor_t *E);


class ExcutionManager
{
public:
    Tensor *lh_tensor;                  // The tensor of equotion left handle.
    string lh_tensor_name;              // The tensor name of eqution left.
    map<string, Tensor *> rhs_tensor;   // The tensors of all equotion right handles.
    void *compute_func;                 // The compute func pointer, which will reinterpret
                                        // func pointer and pointe to dynamic lib func.
    vector<string> vars;                // All the axes name vector, which will apply lsplit,
                                        // lreorder, fsplit, freorder. And the vector indicate
                                        // final loop order after final reorder.
    vector<string> vars_rst;            // vars vectors' backup for using in reset.
    vector<string> freorder_vars;       // The format reorder scheduler result.
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
    

public:
    ExecutionManager() {}
	ExecutionManager(vector<float> &val) : corret_val(val) {}
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
        var = var_rst;
        freorder_vars.clear();
        lsplit_record.clear()
        parallel_vars.clear();
        unroll_vars.clear();
        precompute_vars.clear();
        taco_command.clear();
        compile_success = false;

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
		auto t = tensor_lhs.find(tensor_name);
		if (t != tensor_lhs.end())
		{
			return t->second->print_format();
		}
		else
		{
			t = rhs_tensor.find(tensor_name);
			if (t != rhs_tensor.end())
			{
				return t->second->print_format();
			}
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
        vars.push_back(inner_var);

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
            if (t != rhs_tensor)
            {
                t->second->reorder(reordered_vars);
                freorder_vars = reordered_vars;
                return ;
            }
        }
        throw std::runtime_error("[ERROR][freorder] Tensor not found: " + tensor_name);
    }

    /* Change tensor's axis mode. */
    void fmode(string tensor_name, string var, mode_t mode)
    {
        if (lh_tensor_name == tensor_name) 
        {
            lh_tensor->mode(var, mode);
            return ;
        }
        else
        {
            auto t = rhs_tensor.find(tensor_name);
            if (t != rhs_tensor)
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
            throw std::exception(
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
    }

    /* Parallize a loop axis. */
    void parallize(string var, string hardware)
    {
        auto it = find(vars.begin(), vars.end(), var);
        if (it == vars.end()) // The axis var don't exist.
        {
            throw std::runtime_error("[ERROR][parallize] var not found: " + var);
        }
        auto it = find(unroll_vars.begin(), unroll_vars.end(), var);
        if (it != vars.end()) // Conflict.
        {
            throw std::runtime_error("[ERROR][parallize] var conflict with unroll: " + var);
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
        auto it = find(parallel_vars.begin(), parallel_vars.end(), var);
        if (it != vars.end()) // Conflict.
        {
            throw std::runtime_error("[ERROR][parallize] var conflict with parallel_vars: " + var);
        }
        unroll_vars[var] = factor;
    }
    
    void vectorize(string var)
    {
        parallize(var, "CPUVector");
    }

    void precompute(string expr, string var)
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
        precision += " -t=" + lh_tensor_name + ":float";

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
        format += format_order;
        kernel += "=";

        /* Right handle tensor */
        for (auto &it : rhs_tensor)
        {
            format += " -f=" + it.first + ":";
            kernel += it.first + "(";
            precision += " -t=" + it.first + ":float";

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
            format += format_order;
        }

        kernel += "\""
        
        return kernel + format + precision;
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
        for (auto rh_tensor : rhs_tensor)
        {
            for (auto &f : rh_tensor->get_format())
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
        map<string, int> is_lsplit; // Second is index of lsplit_record
        for (int i = 0; i < lsplit_record.size(); i++)
        {
            string var = lsplit_record[i].first[0];
            string outer_var = lsplit_record[i].first[1];
            string inner_var = lsplit_record[i].first[2];
            int factor = lsplit_record[i].second;

            if(var_is_compressed[var])
            {
                var_is_compressed[var] = false;
            }
            is_lsplit[var] = i;

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
            auto it = is_lsplit.find(it.first);
            if (it != is_split.end()) // The var had been lsplit.
            {
                int index = distance(is_split.begin(), it);
                string var = lsplit_record[index].first[0];
                string outer_var = lsplit_record[index].first[1];
                string inner_var = lsplit_record[index].first[2];
                int factor = lsplit_record[index].second;
                int outer_dimension = roundup(dimensions[var], factor);
                schedules += " -s=\"bound(" + it.first + "," + it.first + "Bound,";
                schedules += to_string(dimensions[it.first]) + ",MaxExact)\"";
            }
        }
        


    }

    string compile_success()
    {

    }

    string run()
    {

    }

}