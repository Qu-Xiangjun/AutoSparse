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

#include "tensor.hpp"
#include "execution_manager.hpp"
#include "time.hpp"
#include "utils.hpp"

using namespace std;

default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
uniform_real_distribution<float> uniform(-1.0, 1.0);


int main(int argc, char *argv[])
{
    if (argc != 4)
	{
		cout << "Wrong arguments" << endl;
		exit(-1);
	}
	string mtx_name(argv[1]); // 矩阵文件的地址
	cout << mtx_name << endl;

    //////////////////////////
	// Reading CSR A from file
	//////////////////////////
	int num_row, num_col, num_nonzero;
	fstream csr(argv[1]); // 矩阵文件的file stream 流初始化
	csr.read((char *)&num_row, sizeof(int));
	csr.read((char *)&num_col, sizeof(int));
	csr.read((char *)&num_nonzero, sizeof(int));
	// 加载CSR存储的矩阵
	vector<int> A_crd(num_nonzero);
	vector<int> A_pos(num_row + 1);
	vector<float> A_val(num_nonzero);
	for (int i = 0; i < num_row + 1; i++) // 读取A_pos
	{
		int data;
		csr.read((char *)&data, sizeof(data));
		A_pos[i] = data;
	}
	for (int i = 0; i < num_nonzero; i++) // 读取A_crd， 
	{
		int data;
		csr.read((char *)&data, sizeof(data));
		A_crd[i] = data;
		A_val[i] = 1.0;	// TODO: A_val直接设置为1，因为这里只使用用稀疏模式不考虑值？？？
	}
	csr.close();
	cout << "NUMROW : " << num_row << " / NUMCOL : " << num_col << " / NNZ : " << num_nonzero << " ";
	cout << endl;

	Compressed_Coo coo(num_nonzero);
	int col_digit2 = (int)(ceil(log2(num_col)));
	#pragma omp parallel for schedule(dynamic, 4)
	for (int i = 0; i < num_row; i++) // 行号
	{
		for (int j = A_pos[i]; j < A_pos[i + 1]; j++) // crd数组和val数组的索引
		{
			uint64_t i_ = (((uint64_t)i) << col_digit2);
			uint64_t t = i_ | A_crd[j];
			// coo存储格式，
			// j 表示该坐标的在csr格式的crd数组中的坐标
			// t  高位存的是行号，低位存的是列号，
			// 		实际对应的就是t用于拼接各个维度的坐标到一个uint64
			// A_val[j] 代表的是该坐标下的值
			coo[j] = {t, A_val[j]}; 
		}
	}

    	////////////////////////////
	// Generating Random Dense B
	////////////////////////////
	int N = 256; // Number of Column in B(k,j)
	vector<float> B = vector<float>(roundup(num_col, 1024) * N, 1); // 1初始化数组，列数padding到1024,实际用不了这么多

	////////////////////////////
	// Generating Dense Output C
	////////////////////////////
	vector<float> C1 = vector<float>(roundup(num_row, 1024) * N, 0); // 0初始化数组，列数padding到1024
	
  	ExecutionManager M;

	// 初始化每个张量的格式信息
	vector<FormatInfo> TensorA, TensorB, TensorC;
	TensorC.push_back({"i", num_row, UNCOMPRESSED}); // {name, dimention, mode_type}
	TensorC.push_back({"j", N, UNCOMPRESSED});
	TensorA.push_back({"i", num_row, UNCOMPRESSED});
	TensorA.push_back({"k", num_col, COMPRESSED});
	TensorB.push_back({"k", num_col, UNCOMPRESSED});
	TensorB.push_back({"j", N, UNCOMPRESSED});
	M.add_tensor("C", TensorC, C1, true);
	M.add_tensor("A", TensorA, coo, false);
	M.add_tensor("B", TensorB, B, false);

	cout << "Use " << NUMCORE << " Threads" << endl;
    M.reset_all();
	M.parallelize("i"); // 为什么不使用NUMCORE 而是48？
	M.compile(48, 32);
	stringstream fixedCSR;
	bool verify = false;
	double fix_time = -1;
	fix_time = M.run(10, 50, verify, false, true);
	fixedCSR << "FixedCSR : " << fix_time << " ms" << endl;
	cout << fixedCSR.str() << endl;

	string schedule;
	
    // Run the extension schedule design space.
	float bestTime = 1000000000;
    string arg3(argv[2]);
    fstream arg3_file(arg3);
    string best_autosparse_schedule;
    for (; getline(arg3_file, schedule); )
    {
        stringstream ss(schedule);
        int i_fsplit, k_fsplit, j_fsplit;
        vector<string> fr(4); // reordered vars for format.
        vector<int> vm(4); // vars mode
        int i_lsplit1, i_lsplit0, k_lsplit1, k_lsplit0, j_lsplit1, j_lsplit0;
		int lreordered_vars_size;
        vector<string> lreordered_vars;
        string parallel_var, vectorize_var, unroll_var, precompute_var;
        int unroll_factor;
        int thread_num, parachunk;
        ss >> i_fsplit >> k_fsplit >> j_fsplit;
	    ss >> fr[0] >> fr[1] >> fr[2] >> fr[3];
	    ss >> vm[0] >> vm[1] >> vm[2] >> vm[3]; 
        ss >> i_lsplit1 >> i_lsplit0 >> k_lsplit1 >> k_lsplit0 >> j_lsplit1 >> j_lsplit0;
		ss >> lreordered_vars_size;
		lreordered_vars.resize(lreordered_vars_size);
        for (int i = 0; i < lreordered_vars_size; i++) ss >> lreordered_vars[i];
        ss >> parallel_var >> vectorize_var >> unroll_var >> unroll_factor >> precompute_var;
        ss >> thread_num >> parachunk;

        try
        {
            M.reset_all();
            M.fsplit("i", "i1", "i0", i_fsplit);
            M.fsplit("j", "j1", "j0", k_fsplit);
            M.fsplit("k", "k1", "k0", j_fsplit);
            M.freorder("A", fr);
            M.fmode("A", "i1", mode_type_array[vm[0]]);
            M.fmode("A", "i0", mode_type_array[vm[1]]);
            M.fmode("A", "k1", mode_type_array[vm[2]]);
            M.fmode("A", "k0", mode_type_array[vm[3]]);

			if (i_lsplit1 > 1)
				M.lsplit("i1", "i11", "i10", i_lsplit1);
			if (i_lsplit0 > 1)
				M.lsplit("i0", "i01", "i00", i_lsplit0);
			if (k_lsplit1 > 1)
				M.lsplit("k1", "k11", "k10", k_lsplit1);
			if (k_lsplit0 > 1)
				M.lsplit("k0", "k01", "k00", k_lsplit0);
			if (j_lsplit1 > 1)
				M.lsplit("j1", "j11", "j10", j_lsplit1);
			if (j_lsplit0 > 1)
				M.lsplit("j0", "j01", "j00", j_lsplit0);
            M.lreorder(lreordered_vars);
            if (parallel_var != "None")
            {
                M.parallelize(parallel_var);
            }
            if (vectorize_var != "None")
            {
                M.vectorize(vectorize_var);
            }
            if (unroll_var != "None")
            {
                M.unroll(unroll_var, unroll_factor);
            }
			if (precompute_var != "None")
            {
				M.precompute(precompute_var);
			}

            M.compile(thread_num, parachunk);
            verify = true;
            float avgtime = M.run(10, 50, verify, true, false, true, fix_time * 3);
			cout << "correct:" << verify << ", " << fixed << setprecision(5) << avgtime;
			cout << " ms" << ", Schedules:" << schedule << endl;
            if (bestTime > avgtime)
			{
				bestTime = avgtime;
				schedule.pop_back();
				best_autosparse_schedule = schedule;
			}
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    double bestTime2 = 1000000000;
	string bestSuperSchedule;
	string arg(argv[3]);	// waco 调度文件地址
	fstream arg_file(arg);
    for (; getline(arg_file, schedule);) // 从arg_file 读取一行存储到schedule 字符串中
	{
		stringstream ss(schedule);
		int isplit, ksplit, jsplit;
		vector<string> r(6); // reorder的6个轴
		vector<int> f(4);	// 稀疏张量split的四个轴 的格式类型
		string pidx;	// 选中需要并行的轴
		int pnum, pchunk;	// 并行num 和 chunk 大小
		ss >> isplit >> ksplit >> jsplit;
		ss >> r[0] >> r[1] >> r[2] >> r[3] >> r[4] >> r[5];
		ss >> f[0] >> f[1] >> f[2] >> f[3]; 
		ss >> pidx >> pnum >> pchunk;

		// Erase Split Size = 1 in loop order "r"
		// 如果split大小为1说明不需要split
		if (isplit == 1)
		{
			r.erase(find(r.begin(), r.end(), "i0"));
			auto itr = find(r.begin(), r.end(), "i1");
			*itr = "i";
		}
		if (ksplit == 1)
		{
			r.erase(find(r.begin(), r.end(), "k0"));
			auto itr = find(r.begin(), r.end(), "k1");
			*itr = "k";
		}
		if (jsplit == 1)
		{
			r.erase(find(r.begin(), r.end(), "j0"));
			auto itr = find(r.begin(), r.end(), "j1");
			*itr = "j";
		}

		// Extract format order of A
		vector<string> rA;
		for (string idx : r)
		{
			if (idx[0] == 'i' || idx[0] == 'k')
			{
				rA.push_back(idx);
			}
		}

		try
		{
			M.reset_all();
			if (isplit != 1) // 不为1则划分为两个轴了的
			{
				M.fsplit("i", "i1", "i0", isplit);
			}
			if (ksplit != 1)
			{
				M.fsplit("k", "k1", "k0", ksplit);
			}
			if (jsplit != 1)
			{
				M.fsplit("j", "j1", "j0", jsplit);
			}
			M.lreorder(r);
			M.freorder("A", rA); // 只需要对A矩阵进行存储重排序

			// 设置稀疏矩阵的 轴的压缩或未压缩属性
			if (isplit != 1) 
			{
				M.fmode("A", "i1", f[0] == 0 ? COMPRESSED : UNCOMPRESSED);
				M.fmode("A", "i0", f[1] == 0 ? COMPRESSED : UNCOMPRESSED);
				M.parallelize(pidx);
			}
			else
			{
				M.fmode("A", "i", f[0] == 0 ? COMPRESSED : UNCOMPRESSED);
				M.parallelize("i");
			}
			if (ksplit != 1)
			{
				M.fmode("A", "k1", f[2] == 0 ? COMPRESSED : UNCOMPRESSED);
				M.fmode("A", "k0", f[3] == 0 ? COMPRESSED : UNCOMPRESSED);
			}
			else
			{
				M.fmode("A", "k", f[2] == 0 ? COMPRESSED : UNCOMPRESSED);
			}
			string schedule_command = M.compile(pnum, pchunk);	// 编译生成kernel
			verify = true;
			float avgtime = M.run(10, 50, verify, true, false, true, fix_time * 3); // 运行并返回时间，这里不需要验证
			cout << "correct:" << verify << ", " << fixed << setprecision(5) << avgtime;
			cout << " ms" << ", Schedules:" << schedule << endl;
			if (bestTime2 > avgtime)
			{
				bestTime2 = avgtime;
				bestSuperSchedule = schedule;
			}
		}
		catch (...)
		{
		}
	}

	cout << endl;
	cout << "Best Schedule found by AutoSparse : " << best_autosparse_schedule << endl;
	cout << "SuperSchedule found by WACO       : " << bestSuperSchedule << endl;
	cout << "AutoSparse : " << bestTime << " ms" << endl;
	cout << "WACO       : " << bestTime2 << " ms" << endl;
	cout << fixedCSR.str() << endl;

	return 0;
}