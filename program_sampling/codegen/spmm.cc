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


/**
 * argc = 3
 * argv[1]: The input matrix filepath.
 * argv[2]: The schedule filepath.
 */
int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		cout << "Wrong arguments" << endl;
		exit(-1);
	}
	string mtx_name(argv[1]); // 矩阵文件的地址

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

	// 首先用固定值并行化 chunk = 48的干净csr进行执行
	M.reset_all();
	M.parallelize("i"); // 为什么不使用NUMCORE 而是48？
	M.compile(48, 32);
	stringstream fixedCSR;
	bool verify = false;
	fixedCSR << "FixedCSR : " << M.run(10, 50, verify, verify, true) << " ms" << endl;

	string arg(argv[2]);	// 调度文件地址
	fstream arg_file(arg);
	string schedule;
	string bestSuperSchedule;
	float bestTime = 1000000000;
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
			float avgtime = M.run(10, 50, verify); // 运行并返回时间，这里不需要验证
			cout << "correct:" << verify << ", " << avgtime << " ms" << ", Schedules:" << schedule << endl;
			if (bestTime > avgtime)
			{
				bestTime = avgtime;
				schedule.pop_back();
				bestSuperSchedule = schedule;
			}
		}
		catch (...)
		{
		}
	}
	cout << endl;
	cout << "SuperSchedule found by WACO : " << bestSuperSchedule << endl;
	cout << "WACO : " << bestTime << " ms" << endl;
	cout << fixedCSR.str() << endl;

	return 0;
}

//{
//  vector<float> elapsed;
//  sparse_matrix_t SA;
//  MKL_INT M = num_row;
//  MKL_INT K = num_col;
//  MKL_INT *mkl_A_pos = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * A_pos.size(), 64);
//  MKL_INT *mkl_A_crd = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * A_crd.size(), 64);
//  float   *mkl_A_val = (float*)  mkl_malloc(sizeof(float) * A_val.size(), 64);
//  vector<float> mklC = vector<float>(roundup(num_row,1024)*N,0);
//  for (int i = 0; i<A_pos.size(); i++) mkl_A_pos[i] = A_pos[i];
//  for (int i = 0; i<A_crd.size(); i++) mkl_A_crd[i] = A_crd[i];
//  for (int i = 0; i<A_val.size(); i++) mkl_A_val[i] = A_val[i];

//  sparse_status_t status = mkl_sparse_s_create_csr(&SA, SPARSE_INDEX_BASE_ZERO, M, K, mkl_A_pos, &(mkl_A_pos[1]), mkl_A_crd, mkl_A_val);
//  matrix_descr descr;
//  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
//
//  auto ts = Clock::now();
//  mkl_sparse_set_memory_hint(SA, SPARSE_MEMORY_AGGRESSIVE);
//  auto ts1 = Clock::now();
//  status = mkl_sparse_set_mm_hint(SA, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_ROW_MAJOR, N, atoi(argv[4]));
//  auto ts2 = Clock::now();
//  status = mkl_sparse_optimize(SA);
//  auto te = Clock::now();
// 	cout << "Start " << endl;
//  for(int niter=0; niter<3; niter++) status = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, SA, descr, SPARSE_LAYOUT_ROW_MAJOR, B.data(), N, N, 0.0, mklC.data(), N);
//  for (int niter = 0; niter < 10; niter++) {
//    auto t1 = Clock::now();
//    status = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, SA, descr, SPARSE_LAYOUT_ROW_MAJOR, B.data(), N, N, 0.0, mklC.data(), N);
//    elapsed.push_back(compute_clock(Clock::now(), t1));
//  }
//  sort(elapsed.begin(), elapsed.end());
//  cout << "MKL " << elapsed[elapsed.size()/2] << endl;
//  mkl_free(mkl_A_pos);
//  mkl_free(mkl_A_crd);
//}
