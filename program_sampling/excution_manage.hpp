#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <map>

using namespace std;
using namespace std::chrono_literals;

/**
 * Debug func
 */
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

/**
 * Safe func to excute shell command.
 */
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

// Define aliases for function Pointers.
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
    vector<string> var;                 // All the axes name vector, which will apply lsplit,
                                        // lreorder, fsplit, freorder.
    vector<string> freorder_var;        // The format reorder scheduler result.
    map<string, string> parallel_var;   // Parallel scheduler record, first indicate axis name,
                                        // and second is taco schedule command field `hardware`.
    map<string, int> unroll_val;        // First is unroll axis var name and second indicates
                                        // unroll factor.
    vector<string> precompute_var;      // Precompute the axes. 
    
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
		for (auto &it : tensor_rhs)
		{
			delete it.second;
		}
	}


    /* Add sparse tensor. */
    void add_tensor(
        string tensor_name, vector<FormatInfo> tensor_format, 
        vector<pair<uint64_t, float>> &coo, bool is_lhs
    )
    {

    }

    /* Add dense tensor. */
    void add_tensor(
        string tensor_name, vector<FormatInfo> tensor_format, 
        vector<float> &dense, bool is_lhs
    )
    {

    }

    /* Get a tensor handle by tensor var name. */
    Tensor &get_tensor(string tensor_name)
    {

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
    string print_tensor(string tensorname)
	{
		auto t = tensor_lhs.find(tensorname);
		if (t != tensor_lhs.end())
		{
			return t->second->print_format();
		}
		else
		{
			t = tensor_rhs.find(tensorname);
			if (t != tensor_rhs.end())
			{
				return t->second->print_format();
			}
		}
	}

    void reset()
    {

    }

    //////////////////////////////////////
    // Scheduler apply.
    //////////////////////////////////////
    
    void fsplit()
    {

    }

    void freorder()
    {

    }

    /* Change tensor's axis mode. */
    void fmode(string tensor_name, string axis_var, mode_t mode)
    {

    }

    void lsplit()
    {

    }

    void lreorder()
    {

    }

    void parallize()
    {

    }

    void unroll()
    {

    }
    
    void vectorize()
    {

    }

    void precompute()
    {

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

    }

    string compile_success()
    {

    }

    string run()
    {

    }






}