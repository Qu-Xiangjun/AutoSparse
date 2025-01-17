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

double SpMMAdd(string filepath)
{
    // Load sparse matrix info
    int num_row, num_col, num_nonzero;
    fstream csr(filepath);
    csr.read((char *)&num_row, sizeof(int));
    csr.read((char *)&num_col, sizeof(int));
    csr.read((char *)&num_nonzero, sizeof(int));

    float *A_val = (float *)mkl_malloc(num_nonzero * sizeof(float), 64);
    MKL_INT *A_pos = (MKL_INT *)mkl_malloc((num_row + 1) * sizeof(MKL_INT), 64);
    MKL_INT *A_crd = (MKL_INT *)mkl_malloc(num_nonzero * sizeof(MKL_INT), 64);
    float *B_val = (float *)mkl_malloc(num_nonzero * sizeof(float), 64);
    MKL_INT *B_pos = (MKL_INT *)mkl_malloc((num_row + 1) * sizeof(MKL_INT), 64);
    MKL_INT *B_crd = (MKL_INT *)mkl_malloc(num_nonzero * sizeof(MKL_INT), 64);
    float *C_val = (float *)mkl_malloc(num_nonzero * sizeof(float), 64);
    MKL_INT *C_pos = (MKL_INT *)mkl_malloc((num_row + 1) * sizeof(MKL_INT), 64);
    MKL_INT *C_crd = (MKL_INT *)mkl_malloc(num_nonzero * sizeof(MKL_INT), 64);
    for (int i = 0; i < num_row + 1; i++)
    {
        int data;
        csr.read((char *)&data, sizeof(data));
        A_pos[i] = data;
        B_pos[i] = data;
        C_pos[i] = data;
    }
    for (int i = 0; i < num_nonzero; i++)
    {
        int data;
        csr.read((char *)&data, sizeof(data));
        A_crd[i] = data;
        A_val[i] = 1.0; //(float)(rand()%1048576)/1048576; // TODO: Now only consider sparse matrix layout.
        B_crd[i] = data;
        B_val[i] = 1.0;
        C_crd[i] = data;
        C_val[i] = 0.0;
    }
    csr.close();

    MKL_INT m, n;
    m = num_row;
    n = num_col;

    // Construct csr sparse matrix A
    sparse_matrix_t csrA;
    sparse_status_t status = mkl_sparse_s_create_csr(
        &csrA,
        SPARSE_INDEX_BASE_ZERO,
        m,
        n,
        A_pos,
        &(A_pos[1]),
        A_crd,
        A_val);
    // cout << status << endl;
    assert(status == SPARSE_STATUS_SUCCESS);
    // Consturct dense matrix B
    sparse_matrix_t csrB;
    status = mkl_sparse_s_create_csr(
        &csrB,
        SPARSE_INDEX_BASE_ZERO,
        m,
        n,
        B_pos,
        &(B_pos[1]),
        B_crd,
        B_val);
    // cout << status << endl;
    assert(status == SPARSE_STATUS_SUCCESS);
    // Consturct dense matrix C
    sparse_matrix_t csrC;
    // status = mkl_sparse_s_create_csr(
    //     csrC,
    //     SPARSE_INDEX_BASE_ZERO,
    //     m,
    //     n,
    //     C_pos,
    //     &(C_pos[1]),
    //     C_crd,
    //     C_val
    // );
    // cout << status << endl;
    // assert (status == SPARSE_STATUS_SUCCESS);

    // spmm mkl api
    double alpha = 1.0;

    double elapsed_time = 0.;
    struct timeval starttime, endtime;
    int warm = 50;
    int round = 100;
    // sparse_status_t mkl_sparse_s_add( const sparse_operation_t operation,
    //                                   const sparse_matrix_t    A,
    //                                   const float              alpha,
    //                                   const sparse_matrix_t    B,
    //                                   sparse_matrix_t          *C );
    // Warm
    for (int cnt = 0; cnt < warm; cnt++)
    {
        status = mkl_sparse_s_add(
            SPARSE_OPERATION_NON_TRANSPOSE,
            csrA,
            alpha,
            csrB,
            &csrC);
        // cout << status << endl;
        assert(status == SPARSE_STATUS_SUCCESS);
        // if (cnt==0) {
        //     sparse_index_base_t index_base;
        //     MKL_INT rows, cols, nonzeros;
        //     MKL_INT *rows_start = NULL, *rows_end = NULL, *col_indx = NULL;
        //     float *values = NULL;

        //     // Export the CSR structure from the sparse matrix handle.
        //     mkl_sparse_s_export_csr(csrC, &index_base, &rows, &cols, &rows_start,
        //                             &rows_end, &col_indx, &values);

        //     cout << "Element: " << ((float*)values)[0] << " " << ((float*)values)[1] << endl;
        // }
    }

    // // Test
    for (int cnt = 0; cnt < round; cnt++)
    {
        // auto t1 = Clock::now();
        gettimeofday(&starttime, NULL);
        status = mkl_sparse_s_add(
            SPARSE_OPERATION_NON_TRANSPOSE,
            csrA,
            alpha,
            csrB,
            &csrC);
        gettimeofday(&endtime, NULL);
        elapsed_time += ((endtime.tv_sec - starttime.tv_sec) * 1000000 + endtime.tv_usec - starttime.tv_usec) / 1000.0;
        // elapsed_time += compute_clock(Clock::now(), t1);
        assert(status == SPARSE_STATUS_SUCCESS);
    }

    return elapsed_time / round;
}

double SpPlus3(string filepath)
{
    // Load sparse matrix info
    int num_row, num_col, num_nonzero;
    fstream csr(filepath);
    csr.read((char *)&num_row, sizeof(int));
    csr.read((char *)&num_col, sizeof(int));
    csr.read((char *)&num_nonzero, sizeof(int));

    float *A_val = (float *)mkl_malloc(num_nonzero * sizeof(float), 64);
    MKL_INT *A_pos = (MKL_INT *)mkl_malloc((num_row + 1) * sizeof(MKL_INT), 64);
    MKL_INT *A_crd = (MKL_INT *)mkl_malloc(num_nonzero * sizeof(MKL_INT), 64);
    float *B_val = (float *)mkl_malloc(num_nonzero * sizeof(float), 64);
    MKL_INT *B_pos = (MKL_INT *)mkl_malloc((num_row + 1) * sizeof(MKL_INT), 64);
    MKL_INT *B_crd = (MKL_INT *)mkl_malloc(num_nonzero * sizeof(MKL_INT), 64);
    float *C_val = (float *)mkl_malloc(num_nonzero * sizeof(float), 64);
    MKL_INT *C_pos = (MKL_INT *)mkl_malloc((num_row + 1) * sizeof(MKL_INT), 64);
    MKL_INT *C_crd = (MKL_INT *)mkl_malloc(num_nonzero * sizeof(MKL_INT), 64);
    for (int i = 0; i < num_row + 1; i++)
    {
        int data;
        csr.read((char *)&data, sizeof(data));
        A_pos[i] = data;
        B_pos[i] = data;
        C_pos[i] = data;
    }
    for (int i = 0; i < num_nonzero; i++)
    {
        int data;
        csr.read((char *)&data, sizeof(data));
        A_crd[i] = data;
        A_val[i] = 1.0; //(float)(rand()%1048576)/1048576; // TODO: Now only consider sparse matrix layout.
        B_crd[i] = data;
        B_val[i] = 1.0;
        C_crd[i] = data;
        C_val[i] = 1.0;
    }
    csr.close();

    MKL_INT m, n;
    m = num_row;
    n = num_col;

    // Construct csr sparse matrix A
    sparse_matrix_t csrA;
    sparse_status_t status = mkl_sparse_s_create_csr(
        &csrA,
        SPARSE_INDEX_BASE_ZERO,
        m,
        n,
        A_pos,
        &(A_pos[1]),
        A_crd,
        A_val);
    // cout << status << endl;
    assert(status == SPARSE_STATUS_SUCCESS);
    // Consturct dense matrix B
    sparse_matrix_t csrB;
    status = mkl_sparse_s_create_csr(
        &csrB,
        SPARSE_INDEX_BASE_ZERO,
        m,
        n,
        B_pos,
        &(B_pos[1]),
        B_crd,
        B_val);
    // cout << status << endl;
    assert(status == SPARSE_STATUS_SUCCESS);
    // Consturct dense matrix C
    sparse_matrix_t csrC;
    status = mkl_sparse_s_create_csr(
        &csrC,
        SPARSE_INDEX_BASE_ZERO,
        m,
        n,
        C_pos,
        &(C_pos[1]),
        C_crd,
        C_val
    );
    // cout << status << endl;
    assert (status == SPARSE_STATUS_SUCCESS);

    sparse_matrix_t csrTemp;
    sparse_matrix_t csrD;

    // spmm mkl api
    double alpha = 1.0;

    double elapsed_time = 0.;
    struct timeval starttime, endtime;
    int warm = 50;
    int round = 100;
    // sparse_status_t mkl_sparse_s_add( const sparse_operation_t operation,
    //                                   const sparse_matrix_t    A,
    //                                   const float              alpha,
    //                                   const sparse_matrix_t    B,
    //                                   sparse_matrix_t          *C );
    // Warm
    for (int cnt = 0; cnt < warm; cnt++)
    {
        status = mkl_sparse_s_add(
            SPARSE_OPERATION_NON_TRANSPOSE,
            csrA,
            alpha,
            csrB,
            &csrTemp);
        // cout << status << endl;
        assert(status == SPARSE_STATUS_SUCCESS);
        status = mkl_sparse_s_add(
            SPARSE_OPERATION_NON_TRANSPOSE,
            csrTemp,
            alpha,
            csrC,
            &csrD);
        // cout << status << endl;
        assert(status == SPARSE_STATUS_SUCCESS);

        // if (cnt==0) {
        //     sparse_index_base_t index_base;
        //     MKL_INT rows, cols, nonzeros;
        //     MKL_INT *rows_start = NULL, *rows_end = NULL, *col_indx = NULL;
        //     float *values = NULL;

        //     // Export the CSR structure from the sparse matrix handle.
        //     mkl_sparse_s_export_csr(csrD, &index_base, &rows, &cols, &rows_start,
        //                             &rows_end, &col_indx, &values);

        //     cout << "Element: " << ((float*)values)[0] << " " << ((float*)values)[1] << endl;
        // }
    }

    // // Test
    for (int cnt = 0; cnt < round; cnt++)
    {
        // auto t1 = Clock::now();
        gettimeofday(&starttime, NULL);
        status = mkl_sparse_s_add(
            SPARSE_OPERATION_NON_TRANSPOSE,
            csrA,
            alpha,
            csrB,
            &csrTemp);
        // cout << status << endl;
        assert(status == SPARSE_STATUS_SUCCESS);
        status = mkl_sparse_s_add(
            SPARSE_OPERATION_NON_TRANSPOSE,
            csrTemp,
            alpha,
            csrC,
            &csrD);
        // cout << status << endl;
        assert(status == SPARSE_STATUS_SUCCESS);
        gettimeofday(&endtime, NULL);
        elapsed_time += ((endtime.tv_sec - starttime.tv_sec) * 1000000 + endtime.tv_usec - starttime.tv_usec) / 1000.0;
        // elapsed_time += compute_clock(Clock::now(), t1);
        assert(status == SPARSE_STATUS_SUCCESS);
    }

    return elapsed_time / round;
}

int main()
{
    char *env_val = getenv("AUTOSPARSE_HOME");
    string inputFileName = string(env_val) + "/dataset/validation_demo.txt";
    string outputFileName = string(env_val) + "/baseline/MKL/result.txt";

    ifstream inputFile(inputFileName);
    if (!inputFile)
    {
        cerr << "Error: Unable to open input file." << endl;
        return 1;
    }

    cout << outputFileName << endl;
    ofstream resultFile(outputFileName);
    if (!resultFile)
    {
        cerr << "Error: Unable to create result file." << endl;
        return 1;
    }

    // Simple Test
    // string filepath = string(env_val) + "/dataset/demo_dataset/nemspmm1_16x4_0.csr";
    // cout << SpMM_S(filepath) << endl;
    // cout << SpMV_S(filepath) << endl;

    string filename;

    // cout << "---------SpMMAdd---------" << endl;
    // resultFile << "---------SpMMAdd---------" << endl;
    // while (getline(inputFile, filename))
    // {
    //     // filename.pop_back();
    //     cout << filename << endl;
    //     string filepath = string(env_val) + "/dataset/demo_dataset/" + filename + ".csr";
    //     cout << filepath << endl;
    //     double res = SpMMAdd(filepath);
    //     resultFile << filename << " = " << res << endl;
    //     cout << filename << " = " << res << endl;
    // }
    // inputFile.close();

    cout << "---------SpPlus3---------" << endl;
    inputFile = ifstream(inputFileName);
    if (!inputFile) {
        cerr << "Error: Unable to open input file." << endl;
        return 1;
    }
    resultFile << "---------SpPlus3---------" << endl;
    while(getline(inputFile, filename))
    {
        // filename.pop_back();
        cout << filename << endl;
        string filepath = string(env_val) + "/dataset/demo_dataset/" + filename + ".csr";
        cout << filepath << endl;
        double res = SpPlus3(filepath);
        resultFile << filename << " = " << res << endl;
        cout << filename << " = " << res << endl;
    }

    inputFile.close();
    // resultFile.close();

    cout << "Processing complete. Results are saved in the 'result' folder." << endl;

    return 0;
}
