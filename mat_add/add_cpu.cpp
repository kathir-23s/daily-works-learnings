#include <iostream>
// #include <stdio.h>
#include <chrono>

using namespace std;

extern "C"{
    void add_matrix_cpu(const float* A, const float* B, float* C, int N){
        auto start = std::chrono::high_resolution_clock::now();


        for (int i=0; i<N; i++){
            C[i] = A[i] + B[i];
        }
        // for (int i = 0; i < M; i++){
        //     for (int j = 0; i < N; j++){
        //         C[i][j] = A[i][j] + B[i][j];
        //     }
        // }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[CPU] Matrix addition took" << ms << "ms\n";
    }
}