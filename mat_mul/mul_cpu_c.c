// cpu_matmul.c
#include <stdio.h>
#include <time.h>

void matmul_cpuc(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB, double* time_ms) {
    clock_t start, end;
    start = clock();

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; k++) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }

    end = clock();

    // Calculate time in milliseconds
    *time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
}
