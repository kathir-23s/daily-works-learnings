#include <stdio.h>
#include <time.h>

void add_matrix_cpu(const float* A, const float* B, float* C, int N){
    clock_t start = clock();

    for (int i = 0; i < N; i++){
        C[i] = A[i] + B[i];
    }

    clock_t end = clock();
    double ms = ((double)(end-start)/CLOCKS_PER_SEC) * 1000.0;
    printf("[CPU] Matrix addition took %.3f ms in C \n", ms);
}