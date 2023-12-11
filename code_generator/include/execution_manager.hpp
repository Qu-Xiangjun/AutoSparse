#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <string>
#include <fstream>
#include <dlfcn.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <cstdlib>

extern string suffix = "";
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
	cout<< "Debug 数据写入文件成功 " <<endl;
}

/**
 * Safe func to excute shell command.
 */
int executeCommand(const char* cmd) {
    FILE* pipe = popen(cmd, "r"); // Open a pipe to execute the command
    if (!pipe) {
		cout << "[ERROR] executeCommand: " << cmd << endl;
		return -1; // Return "ERROR" if the pipe couldn't be opened
	}
    char buffer[128];
    string result = "";
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != nullptr)
            result += buffer;
    }
    pclose(pipe); // Close the pipe
	int compile_success = 0; // Flag to indicate successful compilation.

	if (result.find("error") != string::npos || result.find("Error") != string::npos) {
        cout << "[ERROR] executeCommand: " << result << endl;
		compile_success = -1; // Set flag to indicate compilation failure
    }
    return compile_success; // Return the command output
}


// 定义函数指针的别名
// typedef int (*compute)(taco_tensor_t* C, taco_tensor_t* A, taco_tensor_t* B, float* C_vals, float* A_vals, float* B_vals);
// typedef int (*compute2)(taco_tensor_t* C, taco_tensor_t* A, taco_tensor_t* B, taco_tensor_t* D, float* C_vals, float* A_vals, float* B_vals, float* D_vals);
typedef int (*compute)(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B);
typedef int (*compute2)(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *D);

class ExecutionManager
{
private:
	map<string, FormatScheduler *> tensor_lhs; // 计算等式左侧的张量，存储张量的名称和格式调度信息类
	map<string, FormatScheduler *> tensor_rhs; // 计算等式右侧的张量，存储张量的名称和格式调度信息类
	compute func;
	compute2 func2;
	void *lib_handle = NULL;  // 打开动态链接库 dlopen 返回的句柄
	vector<float> ref;	// 该计算的正确结果，按照T_val 所有非零值存储
	string parallel_var;
	bool is_parallel = false;
	vector<string> reorder_var;
	float timeout = 0;

	string taco_command;
	bool compile_success;

	/**
	 * 生成TACO的命令
	 * taco的命令可以通过 $WACO_HOME/code_generator/taco/build/bin/taco 进行查看
	 * 调度的命令可以通过 $WACO_HOME/code_generator/taco/build/bin/taco -help=scheduling 进行查看
	 */
	string gen_command()
	{
		string kernel = "\"";
		string format = "";
		string precision = "";
		for (auto &it : tensor_lhs)
		{
			format += " -f=";
			format += it.first;
			format += ":";

			kernel += it.first;
			kernel += "(";

			precision += " -t=" + it.first + ":float"; // 这里只支持float类型？
			
			vector<FormatInfo> TensorFormat = it.second->get_format();
			for (int rank = 0; rank < TensorFormat.size(); rank++)
			{
				if (TensorFormat[rank].mode == UNCOMPRESSED)
				{
					format += "d";
				}
				else if (TensorFormat[rank].mode == COMPRESSED)
				{
					format += "s";
				}
				kernel += TensorFormat[rank].var;
				if (rank == TensorFormat.size() - 1)
					kernel += ")";
				else
					kernel += ",";
			}
			format += ":";
			for (int rank = 0; rank < TensorFormat.size(); rank++)
			{	// TODO: 感觉这个有问题，只能是按照0,1,2,3...顺序写，但如csc B:ds:1,0 无法实现
				// TODO: 因此是否这里可以实现为rank的顺序也随机？
				// TODO: 但是，通过freorder改变的是如A(i,j)=B(i,k)*C(k,j) 到 A(i,j)=B(k,i)*C(k,j), 也能实现csc
				//			则这与 kernel 中的值相关
				format += to_string(rank) + ",";
			}
			format.pop_back();
		}
		kernel += "=";
		for (auto &it : tensor_rhs)
		{
			format += " -f=";
			format += it.first;
			format += ":";
			kernel += it.first;
			kernel += "(";
			precision += " -t=" + it.first + ":float";
			vector<FormatInfo> TensorFormat = it.second->get_format();
			for (int rank = 0; rank < TensorFormat.size(); rank++)
			{
				if (TensorFormat[rank].mode == UNCOMPRESSED)
				{
					format += "d";
				}
				else if (TensorFormat[rank].mode == COMPRESSED)
				{
					format += "s";
				}
				kernel += TensorFormat[rank].var;
				if (rank == TensorFormat.size() - 1)
					kernel += ")";
				else
					kernel += ",";
			}
			format += ":";
			for (int rank = 0; rank < TensorFormat.size(); rank++)
			{
				format += to_string(rank) + ",";
			}
			format.pop_back();
			kernel += "*"; // TODO: 这里计算负只能是乘法了？
		}
		kernel.pop_back();
		kernel += "\"";

		return kernel + format + precision;
	}

	bool IsPowerOfTwo(int x) // 是否为2的指数倍
	{
		return (x & (x - 1)) == 0;
	}

	/**
	 * 生成调度命令字符串
	 */
	string gen_sched()
	{
		map<string, bool> vars;	// 该轴是否为 UNCOMPRESSED，只要一个张量中不是就false
		map<string, int> dims;	// 轴的长度
		for (auto &it : tensor_lhs)
		{
			FormatScheduler *t = it.second;
			for (auto &rank : t->get_format())
			{
				if (vars.find(rank.var) == vars.end()) // 第一次出现这个轴
				{
					vars[rank.var] = rank.mode == UNCOMPRESSED;
					dims[rank.var] = rank.dimension;
				}
				else // 已收录过该轴，则与上结果
				{
					vars[rank.var] &= (rank.mode == UNCOMPRESSED);
				}
			}
		}
		for (auto &it : tensor_rhs)
		{
			FormatScheduler *t = it.second;
			for (auto &rank : t->get_format())
			{
				if (vars.find(rank.var) == vars.end())
				{
					vars[rank.var] = rank.mode == UNCOMPRESSED;
					dims[rank.var] = rank.dimension;
				}
				else
				{
					vars[rank.var] &= (rank.mode == UNCOMPRESSED);
				}
			}
		}

		string schedule = "";
		for (auto &it : vars) // 对于 UNCOMPRESSED 的边都加上准确的bound
		{
			if (it.second == true) // UNCOMPRESSED
			{	// TODO: 这里对稠密轴是固定这个循环的大小为其轴长，为什么要多此一举呢？
				schedule += " -s=\"bound(" + it.first + "," + it.first + "b," + to_string(dims[it.first]) + ", MaxExact)\"";
			}
		}

		// Reorder
		if (reorder_var.size() > 0)
		{
			schedule += " -s=\"reorder(";
			for (int i = 0; i < reorder_var.size(); i++)
			{
				string var = reorder_var[i];
				schedule += var + (vars[var] ? "b" : ""); // 是UNCOMPRESS 的注意上面bound处理时候将变量名转换为添加了b后缀
				if (i < reorder_var.size() - 1)
				{
					schedule += ",";
				}
				else
				{
					schedule += ")\"";
				}
			}
		}

		// Parallelize
		if (is_parallel)
		{
			bool is_reduction = true;
			for (auto t : tensor_lhs) // 遍历结果张量的每一个轴，查看是否有标记为并行的轴，则能说明并行轴是否为reduce的轴
			{
				FormatScheduler *t_format = t.second;
				is_reduction &= !(t_format->is_var_exist(parallel_var));
			}
			// 是UNCOMPRESS 的轴注意上面bound处理时候将变量名转换为添加了b后缀
			// 注意，这里可以通过parallel 命令支持vectorize
			schedule += " -s=\"parallelize(" + parallel_var + (vars[parallel_var] ? "b" : "")
						+ ",CPUThread," 									// 只支持CPU的并行
						+ (is_reduction ? "Atomics" : "NoRaces") + ")\""; 	// reduce 的轴需要 Atomics（原子累加）
		}

		return schedule;
	}

public:
	ExecutionManager() {}
	ExecutionManager(vector<float> &ref) : ref(ref) {}
	ExecutionManager(vector<float> &ref, float timeout) : ref(ref), timeout(timeout) {}
	~ExecutionManager()
	{
		for (auto &it : tensor_lhs)
		{
			delete it.second;
		}
		for (auto &it : tensor_rhs)
		{
			delete it.second;
		}
		if (lib_handle)
		{
			dlclose(lib_handle);
		}
	}

	/**
	 * 设置某一个张量的coo格式的临时存储buffer内容
	 */
	void mod_tensor(string tensorname, vector<pair<uint64_t, float>> &coo)
	{
		auto t = tensor_lhs.find(tensorname);
		if (t != tensor_lhs.end())
		{
			t->second->set_coo(coo);
		}
		else
		{
			t = tensor_rhs.find(tensorname);
			if (t != tensor_rhs.end())
			{
				t->second->set_coo(coo);
			}
		}
	}


	/**
	 * 两种添加计算中涉及张量的函数
	 */
	void add_tensor(string tensorname, vector<FormatInfo> tensorformat, vector<pair<uint64_t, float>> &coo, bool lhs)
	{
		/**
		 * 添加操作数
		 * arg1 : tensorname 张量字符串名称
		 * arg2 : tensorformat 张量的各个维度格式信息
		 * arg3 : dense 稀疏格式下的值存储
		 * arg4 : lhs 是否为等号左侧
		 */
		FormatScheduler *T = new FormatScheduler(coo, tensorformat);
		if (lhs)
		{
			tensor_lhs[tensorname] = T;
		}
		else
		{
			tensor_rhs[tensorname] = T;
		}
	}

	void add_tensor(string tensorname, vector<FormatInfo> tensorformat, vector<float> &dense, bool lhs)
	{
		/**
		 * arg3 : dense 稠密张量的所有元素存储
		 */
		FormatScheduler *T = new FormatScheduler(dense, tensorformat);
		if (lhs)
		{
			tensor_lhs[tensorname] = T;
		}
		else
		{
			tensor_rhs[tensorname] = T;
		}
	}


	/**
	 * 根据张量的名称 查找 该计算中张量的格式调度信息
	 */
	FormatScheduler &get_tensor(string tensorname)
	{
		auto t = tensor_lhs.find(tensorname);
		if (t != tensor_lhs.end())
		{
			return *(t->second);
		}
		else
		{
			t = tensor_rhs.find(tensorname);
			if (t != tensor_rhs.end())
			{
				return *(t->second);
			}
		}
	}


	/**
	 * 应用split对某一个轴，将涉及该轴的所有张量 格式中轴进行split
	 * 
	 * 以下 tensorname 不起作用，因为会对所有与此轴相关的所有张量都处理，而不是值对于某一个张量
	 */
	ExecutionManager &fsplit(string tensorname, string var, string outer_var, string inner_var, int split_size)
	{
		
		for (auto &it : tensor_lhs)
		{
			it.second->split(var, outer_var, inner_var, split_size);
		}
		for (auto &it : tensor_rhs)
		{
			it.second->split(var, outer_var, inner_var, split_size);
		}
		return *this;
	}

	
	/** format reorder
	 * 存储顺序的reorder
	 * 将某一个张量的存储顺序进行进行变换，如影响了行主序还是列主序存储方式
	 */
	ExecutionManager &freorder(string tensorname, vector<string> reordered_vars)
	{
		auto t = tensor_lhs.find(tensorname);
		if (t != tensor_lhs.end())
		{
			t->second->reorder(reordered_vars);
		}
		else
		{
			t = tensor_rhs.find(tensorname);
			if (t != tensor_rhs.end())
			{
				t->second->reorder(reordered_vars);
			}
		}
		return *this;
	}

	/** schedule reorder
	 * 更改计算顺序的reorder ，即将计算for循环的顺序改变
	 * 此调换顺序放在计算manager类中，待生成 schedule 命令时候使用
	 */
	ExecutionManager &lreorder(vector<string> reordered_vars)
	{
		reorder_var = reordered_vars;
		return *this;
	}

	/**
	 * 更改某个张量的某个轴的属性，压缩 or 不压缩
	 */
	ExecutionManager &fmode(string tensorname, string var, taco_mode_t mode)
	{
		auto t = tensor_lhs.find(tensorname);
		if (t != tensor_lhs.end())
		{
			t->second->mode(var, mode);
		}
		else
		{
			t = tensor_rhs.find(tensorname);
			if (t != tensor_rhs.end())
			{
				t->second->mode(var, mode);
			}
		}
		return *this;
	}

	/**
	 * 设置某个轴并行化，注意 只会有一个轴被并行化
	 */
	ExecutionManager &parallelize(string var, int num_thread, int chunk_size)
	{
		// Test Variable Existence
		bool is_exist = false;
		for (auto t : tensor_lhs)
		{
			FormatScheduler *t_format = t.second;
			is_exist |= (t_format->is_var_exist(var));
		}
		for (auto t : tensor_rhs)
		{
			FormatScheduler *t_format = t.second;
			is_exist |= (t_format->is_var_exist(var));
		}
		if (is_exist == false)
		{
			cerr << "[Paralleize] There is no " << var << " in notation" << endl;
			exit(-1);
		}
		// omp_set_num_threads(num_thread);
		omp_set_num_threads(NUMCORE);
		omp_set_schedule(omp_sched_dynamic, chunk_size);
		is_parallel = NUMCORE > 1; // num_thread > 1; // TODO: 这里的是不是应该直接用 NUMCORE
		parallel_var = var; // 记录被并行化的轴
		return *this;
	}

	/**
	 * 将所有的张量pack到实际存储区域
	 */
	void pack_all()
	{
		for (auto &it : tensor_lhs)
		{
			it.second->pack();
		}
		for (auto &it : tensor_rhs)
		{
			it.second->pack();
		}
	}

	/**
	 * 输出所有的张量信息
	 */
	void print_all()
	{
		for (auto &it : tensor_lhs)
		{
			cout << ">>>>>>>>>>>>>> " << it.first << " <<<<<<<<<<<<<<" << endl;
			cout << it.second->print_format();
		}
		for (auto &it : tensor_rhs)
		{
			cout << ">>>>>>>>>>>>>> " << it.first << " <<<<<<<<<<<<<<" << endl;
			cout << it.second->print_format();
		}
	}

	/**
	 * 返回某一个张量的信息
	 */
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


	/**
	 * 初始化或重置整个执行管理，开启下一个计算格式调度管理
	 */
	void init_all()
	{
		reorder_var.clear();
		parallel_var = "";
		is_parallel = false;
		// 初始化重置所有张量的格式调度信息
		for (auto &it : tensor_lhs)
		{
			it.second->init();  
		}
		for (auto &it : tensor_rhs)
		{
			it.second->init();
		}

		func = nullptr;
		func2 = nullptr;

	}


	/**
	 * 编译生成计算内核
	 */
	string compile()
	{
		char *env_val = getenv("WACO_HOME");
		if (env_val == NULL)
		{
			std::cout << "ERR : Environment variable WACO_HOME not defined" << std::endl;
			exit(1);
		}
		std::string waco_prefix = env_val;

		taco_command = waco_prefix + "/code_generator/taco/build/bin/taco "; // taco 命令地址
		string kernel = gen_command();	// 核心计算描述命令
		string source = " -write-compute=taco_kernel" + suffix + ".c";	// 输出taco生成的计算内核地址
		string schedule = gen_sched();	// 生成调度命令
		taco_command += kernel + source + schedule;	// 完成执行命令组装
		string header = "#include <stdint.h> \\n"
						"typedef enum { COMPRESSED, UNCOMPRESSED } taco_mode_t;\\n"
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
		string taco_header_command = "sed -i '1s/^/" + header + "/' taco_kernel" + suffix + ".c"; // 添加头文件和代码到该文件
		int taco_compile = system(taco_command.c_str());
		int taco_add_header = system(taco_header_command.c_str());
		if (tensor_rhs.size() == 3) // 只有在右侧是3个操作数的时候执行下面 sddmm_patch.py 脚本
		{
			string patch_command = "python " + waco_prefix + "/code_generator/include/sddmm_patch.py ./taco_kernel" + suffix + ".c";
			system(patch_command.c_str());
		}
		
		// 只有在并行的时候才编译生成 taco_kernel.c 文件的.so动态链接库
		// TODO: 这力由于is_parallel 赋值来源于 NUMCORE>1 即可，所以必然会开启，但注意必须调用parallel函数才行
		// if (is_parallel) 
		// {	
		string cc_command;
		#ifdef ICC
		cc_command = "icc -march=native -mtune=native -O3 -ffast-math -qopenmp -fPIC -shared taco_kernel" + suffix + ".c -o taco_kernel" + suffix + ".so -lm";
		#elif GCC
		cc_command = "gcc -march=native -mtune=native -O3 -fopenmp -ffast-math -fPIC -shared taco_kernel" + suffix + ".c -o taco_kernel" + suffix + ".so -lm";
		#endif
		compile_success = executeCommand(cc_command.c_str());

		if(!compile_success) 
			return taco_command;
		// }

		if (lib_handle) // 打开动态链接库 dlopen 返回的句柄是否存在
		{
			dlclose(lib_handle); // 关闭之前使用 dlopen() 或者类似函数打开的动态链接库（shared library）
		}
		string taco_kernel = "./taco_kernel" + suffix + ".so";
		// RTLD_NOW 表示在 dlopen() 调用时立即解析库中的所有符号。
		// RTLD_LOCAL 表示动态库中的符号不会被动态加载的其他库使用，而是局限于当前动态库。
		lib_handle = dlopen(taco_kernel.c_str(), RTLD_NOW | RTLD_LOCAL);
		if (!lib_handle)
		{
			cout << "DLOPEN - " << dlerror() << endl;
		}

		// 给定计算的函数，从动态库中调取
		if (tensor_rhs.size() == 2)
			func = (compute)dlsym(lib_handle, "compute");
		else if (tensor_rhs.size() == 3)
		{
			func2 = (compute2)dlsym(lib_handle, "compute");
		}

		if (dlerror() != NULL)
		{
			cout << "DLSYM ERROR" << endl;
		}

		return taco_command;
	}

	/**
	 * 执行该调度格式下的核函数
	 * Parameters
	 * ----------
	 * arg1 : warm 
	 *   热身轮数
	 * arg2 : warm 
	 *   实际测试运行次数
	 * arg3 : verify
	 * 	 是否验证程序正确性，与类变量ref（外面传入的）进行比较
	 * arg4 : store
	 * 	 是否存储当前计算的结果
	 * 
	 * Return
	 * ------
	 * ret : elapsed_time 执行时间 ms
	 */
	double run(int warm, int round, bool &verify, bool store = false, bool avg_test = true)
	{
		if(!compile_success) {
			verify = false;
			return -1;
		}

		vector<taco_tensor_t *> T; // 等式左右所有的张量
		for (auto &it : tensor_lhs)
		{
			T.push_back(it.second->get_taco_tensor()); // 提取张量到 执行句柄 中
		}
		for (auto &it : tensor_rhs)
		{
			T.push_back(it.second->get_taco_tensor());
		}
		double elapsed_time;
		if (tensor_rhs.size() == 2) // 右侧张量数量为两个的
		{
			// 热身
			for (int r = 0; r < warm; r++) 
			{
				auto t1 = Clock::now();
				// func(T[0], T[1], T[2], (float*)(T[0]->vals), (float*)(T[1]->vals), (float*)(T[2]->vals));
				func(T[0], T[1], T[2]); // 调用compute函数
				double tt = compute_clock(Clock::now(), t1);
				if (tt > 100)
					return tt;
			}
			vector<double> elapsed;
			
			// 实际测量
			for (int r = 0; r < round; r++) 
			{
				auto t1 = Clock::now();
				func(T[0], T[1], T[2]);
				double tt = compute_clock(Clock::now(), t1);
				elapsed.push_back(tt);
			}

			if(avg_test) {
				for(int tt = 0; tt < elapsed.size(); tt++)
					elapsed_time += elapsed[tt];
				elapsed_time /= elapsed.size();
			}
			else
			{
				sort(elapsed.begin(), elapsed.end());
				elapsed_time = elapsed[elapsed.size() / 2]; // 取中位数时间
			}

			// 计算真实值用于后面
			for (auto &it : tensor_lhs)
			{
				FormatScheduler *t = it.second;
				vector<float> &res = t->get_vals();
				fill(res.begin(), res.end(), 0);
				func(T[0], T[1], T[2]);
				// float *T0 = (float*)(T[0]->vals);
				// fwrite2file(T0, T[0]->vals_size, "******* T0 *******");
			}

			// 验证正确性
			if (verify)
			{
				float *res = (float*)(T[0]->vals);
				bool flag = true;
				// #pragma omp parallel for
				for (int i = 0; i < ref.size(); i++) // 与类变量（外面初始化的）比较结果
				{
					if (abs(ref[i] - res[i]) > 0.01)
					{
						// string error;
						// error = "Wrong " + to_string(i) + " " + to_string(ref[i]) + " " + to_string(res[i]);
						// #pragma omp critical
						// {
						// 	cout << error << endl;
						// }
						// exit(-1);
						flag = false;
						break;
					}
				}
				verify = flag;
			}

			// Store current result
			if(store) 
			{
				ref.clear();
				float *res = (float*)(T[0]->vals);
				int res_size = T[0]->vals_size;
				fwrite2file(res, res_size, "******* T0 *******");
				ref.resize(res_size);
				for (int i = 0; i < res_size; i++)
				{
					ref[i] = res[i];
					// cout << ref[i] << " ";
				}
				// cout<<endl;
			}
		}
		else if (tensor_rhs.size() == 3)
		{ // MTTKRP
			for (int r = 0; r < warm; r++)
			{
				func2(T[0], T[1], T[2], T[3]);
			}
			vector<double> elapsed;
			for (int r = 0; r < round; r++)
			{
				auto t1 = Clock::now();
				// func2(T[0], T[1], T[2], T[3], (float*)(T[0]->vals), (float*)(T[1]->vals), (float*)(T[2]->vals), (float*)(T[3]->vals));
				func2(T[0], T[1], T[2], T[3]);
				double tt = compute_clock(Clock::now(), t1);
				if (tt > 1000)
					return tt;
				elapsed.push_back(tt);
			}
			sort(elapsed.begin(), elapsed.end());
			elapsed_time = elapsed[round / 2];

			if (verify)
			{
				for (auto &it : tensor_lhs)
				{
					FormatScheduler *t = it.second;
					vector<float> &res = t->get_vals();
					fill(res.begin(), res.end(), 0);
					func2(T[0], T[1], T[2], T[3]);
					bool flag = true;
					for (int i = 0; i < res.size(); i++)
					{
						if (abs(ref[i] - res[i]) > 0.01)
						{
							// string error;
							// error = "Wrong " + to_string(i) + " " + to_string(ref[i]) + " " + to_string(res[i]);
							// #pragma omp critical
							// {
							// 	cout << error << endl;
							// }
							// exit(-1);
							flag = false;
							break;
						}
					}
					verify = flag;
				}
			}
		}

		return elapsed_time;
	}
};
