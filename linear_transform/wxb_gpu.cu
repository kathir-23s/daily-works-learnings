#include <cuda_runtime.h>
#include <iostream>

extern "C" {

__global__ void wxb_kernel(const float* W, const float* X, const float* b, float* Y,
                           int out_features, int in_features, int batch_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output feature index

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += X[row * in_features + k] * W[col * in_features + k];
        }
        Y[row * out_features + col] = sum + b[col];
    }
}

void wxb_gpu(const float* W, const float* X, const float* b, float* Y,
             int out_features, int in_features, int batch_size, float* time_ms) {
    float *d_W, *d_X, *d_b, *d_Y;
    size_t sizeW = out_features * in_features * sizeof(float);
    size_t sizeX = batch_size * in_features * sizeof(float);
    size_t sizeB = out_features * sizeof(float);
    size_t sizeY = batch_size * out_features * sizeof(float);

    cudaMalloc(&d_W, sizeW);
    cudaMalloc(&d_X, sizeX);
    cudaMalloc(&d_b, sizeB);
    cudaMalloc(&d_Y, sizeY);

    cudaMemcpy(d_W, W, sizeW, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((out_features + 15) / 16, (batch_size + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    wxb_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_W, d_X, d_b, d_Y,
                                                   out_features, in_features, batch_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(time_ms, start, stop);
    cudaMemcpy(Y, d_Y, sizeY, cudaMemcpyDeviceToHost);

    cudaFree(d_W);
    cudaFree(d_X);
    cudaFree(d_b);
    cudaFree(d_Y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

}
