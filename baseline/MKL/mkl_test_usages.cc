#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <time.h>
#include <sys/time.h>

#include "mkl.h"
#include "mkl_types.h"
#include "mkl_lapacke.h"
#include "mkl_spblas.h"

#include "time.hpp"

using namespace std;

using namespace std::chrono_literals;

double SpMM_S(string filepath, MKL_INT n)
{
    // Load sparse matrix info
    int num_row, num_col, num_nonzero;
    fstream csr(filepath);
	csr.read((char *)&num_row, sizeof(int));
	csr.read((char *)&num_col, sizeof(int));
	csr.read((char *)&num_nonzero, sizeof(int));
    
    float *A_val = (float*)mkl_malloc(num_nonzero * sizeof(float), 64);
    MKL_INT *A_pos = (MKL_INT*)mkl_malloc((num_row + 1) * sizeof(MKL_INT), 64);
    MKL_INT *A_crd = (MKL_INT*)mkl_malloc(num_nonzero * sizeof(MKL_INT), 64);
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
		A_val[i] = (float)(rand()%1048576)/1048576; // TODO: Now only consider sparse matrix layout.
	}
    csr.close();

    MKL_INT m, k;
    m = num_row;
    k = num_col;

    // Consturct dense matrix B
    float *matrixB = (float*)mkl_malloc(n*k * sizeof(float), 64);
    for (int i = 0; i < n*k; i++) matrixB[i] = 1.0;

    // Construct csr sparse matrix A
    sparse_matrix_t csrA;
    sparse_status_t status = mkl_sparse_s_create_csr(
        &csrA,
        SPARSE_INDEX_BASE_ZERO,
        m,
        k,
        A_pos,
        &(A_pos[1]),
        A_crd,
        A_val
    );
    cout << status << endl;
    assert (status == SPARSE_STATUS_SUCCESS);

    // spmm mkl api
    double alpha = 1.0;
	double beta = 0.0;
    struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT };
    
    double elapsed_time = 0.;
    struct timeval starttime, endtime;
    int warm = 50;
    int round = 100;
    // Warm
    for (int cnt = 0; cnt < warm; cnt++){
        float *matrixC = (float*)mkl_malloc(m*n * sizeof(float), 64);
        for (int i = 0; i < m*n; i++) matrixC[i] = 0.;
        status = mkl_sparse_s_mm(
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha,
            csrA,
            descr,
            SPARSE_LAYOUT_COLUMN_MAJOR,
            matrixB,
            n,
            n,
            beta,
            matrixC,
            m
        );
        assert (status == SPARSE_STATUS_SUCCESS);
    }

    // Test
    for (int cnt = 0; cnt < round; cnt++){
        float *matrixC = (float*)mkl_malloc(m*n * sizeof(float), 64);
        for (int i = 0; i < m*n; i++) matrixC[i] = 0.;

        // auto t1 = Clock::now();
        gettimeofday(&starttime,NULL);
        status = mkl_sparse_s_mm(
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha,
            csrA,
            descr,
            SPARSE_LAYOUT_COLUMN_MAJOR,
            matrixB,
            n,
            n,
            beta,
            matrixC,
            m
        );
        gettimeofday(&endtime,NULL);
        elapsed_time += ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000.0;
        // elapsed_time += compute_clock(Clock::now(), t1);
        assert (status == SPARSE_STATUS_SUCCESS);
        if (cnt==0) cout << matrixC[1] << endl;
    }

    return elapsed_time / round;
}


int main()
{
    char *env_val = getenv("AUTOSPARSE_HOME");
    double res = 0.;
    string filename = "strides_mask";
    string filepath = string(env_val) + "/dataset/demo_dataset/" + filename + ".csr";
    cout << filepath << endl;
    res = SpMM_S(filepath, 256);
    cout << filename << " = " << res << endl;

    filename = "encoder.layer.9.intermediate.dense.weight";
    filepath = string(env_val) + "/dataset/demo_dataset/" + filename + ".csr";
    cout << filepath << endl;
    res = SpMM_S(filepath, 3072);
    cout << filename << " = " << res << endl;

    filename = "encoder.layer.8.output.dense.weight";
    filepath = string(env_val) + "/dataset/demo_dataset/" + filename + ".csr";
    cout << filepath << endl;
    res = SpMM_S(filepath, 768);
    cout << filename << " = " << res << endl;

    filename = "encoder.layer.9.output.dense.weight";
    filepath = string(env_val) + "/dataset/demo_dataset/" + filename + ".csr";
    cout << filepath << endl;
    res = SpMM_S(filepath, 768);
    cout << filename << " = " << res << endl;

    filename = "encoder.layer.10.output.dense.weight";
    filepath = string(env_val) + "/dataset/demo_dataset/" + filename + ".csr";
    cout << filepath << endl;
    res = SpMM_S(filepath, 768);
    cout << filename << " = " << res << endl;

    filename = "encoder.layer.11.output.dense.weight";
    filepath = string(env_val) + "/dataset/demo_dataset/" + filename + ".csr";
    cout << filepath << endl;
    res = SpMM_S(filepath, 768);
    cout << filename << " = " << res << endl;

    return 0;
}

