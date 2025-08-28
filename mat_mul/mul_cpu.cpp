// cpu_matmul.cpp
#include <iostream>
#include <chrono>

extern "C" {

void matmul_cpu(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB, double* time_ms) {

 auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; k++) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    *time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    // std::cout << "[CPU] Matrix Multiplication took" << time_ms << "ms\n";
}

}
