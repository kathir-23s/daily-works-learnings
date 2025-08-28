#include <iostream>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* A, const float* B, float* C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" {
    void add_matrix_gpu(const float* A, const float* B, float* C, int N){
        float *d_A, *d_B, *d_C;

        cudaMalloc((void**)&d_A, N * sizeof(float));
        cudaMalloc((void**)&d_B, N * sizeof(float));
        cudaMalloc((void**)&d_C, N * sizeof(float));

        cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(160);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        add_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "[GPU] Matrix addition took " << ms << " ms\n";

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}