#include <iostream>
#include <chrono>
#define M 64
#define N 128
#define K 256
float A[M][K], B[K][N], C[M][N];
int main() {

    
    // 获取当前时间点（开始时间）
    auto start_time = std::chrono::high_resolution_clock::now();

    float acc = 0;
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
                acc = 0.;
            for(int k=0; k<K; k++){
                acc += A[i][k] * B[k][j]; 
            }
            C[i][j] = acc;
        }
    }

    // 获取当前时间点（结束时间）
    auto end_time = std::chrono::high_resolution_clock::now();
    // 计算时间差异
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // 打印执行时间
    std::cout << "Execution Time: " << duration.count() << " microseconds" << std::endl;

    return 0;
}
