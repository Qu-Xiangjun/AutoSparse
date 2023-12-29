/**
 * A test file for taco_kernel.c
 * Read matrix meta data from file.
 * Handle write schedule for loop and fromat.
 */
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

/**
 * argc = 2
 * argv[1]: The input matrix filepath.
 */
int main(int argc, char *argv[])
{
	if (argc == 1)
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

	vector<pair<uint64_t, float>> coo(num_nonzero);
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
			// t 高位存的是行号，低位存的是列号，实际对应的就是t用于拼接各个维度的坐标到一个uint64
			// A_val[j] 代表的是该坐标下的值
			coo[j] = {t, A_val[j]}; 
		}
	}

	////////////////////////////
	// Generating Random Dense B
	////////////////////////////
	int N = 256; // Number of Column in B(k,j)
	vector<float> B = vector<float>(roundup(num_col, 1024) * N, 1);

	////////////////////////////
	// Generating Dense Output C
	////////////////////////////
	vector<float> C = vector<float>(roundup(num_row, 1024) * N, 0);


	////////////////////////////
	// Initial Excution Manager
	////////////////////////////
  	ExecutionManager M;

	vector<FormatInfo> TensorA, TensorB, TensorC;
	TensorC.push_back({"i", num_row, UNCOMPRESSED}); // {name, dimention, mode_type}
	TensorC.push_back({"j", N, UNCOMPRESSED});
	TensorA.push_back({"i", num_row, UNCOMPRESSED});
	TensorA.push_back({"k", num_col, COMPRESSED});
	TensorB.push_back({"k", num_col, UNCOMPRESSED});
	TensorB.push_back({"j", N, UNCOMPRESSED});
	M.add_tensor("C", TensorC, C, true);
	M.add_tensor("A", TensorA, coo, false);
	M.add_tensor("B", TensorB, B, false);
	M.reset_all();

	///////////////////////////////
	// Excute Puring Correct Result
	///////////////////////////////
	vector<string> reordered_vars = {"i", "k", "j"};
	M.lreorder(reordered_vars);
	M.parallelize("i");
	M.compile(48, 32);
	stringstream fixedCSR;
	bool verify = false;
	fixedCSR << "FixedCSR : " << M.run(10, 50, verify, verify, true) << " ms" << endl;

	////////////////////////////
	// Add Test Schedule
	////////////////////////////
	M.reset_all();
	vector<string> rB;
	rB.push_back("i");
	rB.push_back("j");
	rB.push_back("k");
	M.lreorder(rB);

	////////////////////////////
	// Build and Excution
	////////////////////////////
	M.compile(48, 32);
	verify = true;
	float avgtime = M.run(10, 50,verify, verify); // 运行并返回时间，这里不需要验证


	// cout << fixedCSR.str() << endl;
	cout << "Testing :" << avgtime << " ms" << ",  correct:" << verify << endl;

	return 0;
}
