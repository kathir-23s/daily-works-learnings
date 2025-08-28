#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CPU & GPU function declarations
void wxb_cpu(const float* W, const float* X, const float* b, float* Y,
             int out_features, int in_features, int batch_size, double* time_ms);
void wxb_gpu(const float* W, const float* X, const float* b, float* Y,
             int out_features, int in_features, int batch_size, float* time_ms);

int main() {
    int out_features = 10;
    int in_features = 6;
    int batch_size = 100000000;

    float* W = (float*)malloc(out_features * in_features * sizeof(float));
    float* X = (float*)malloc(batch_size * in_features * sizeof(float));
    float* b = (float*)malloc(out_features * sizeof(float));
    float* Y_cpu = (float*)malloc(batch_size * out_features * sizeof(float));
    float* Y_gpu = (float*)malloc(batch_size * out_features * sizeof(float));

    // Random init
    for (int i = 0; i < out_features * in_features; i++) W[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < batch_size * in_features; i++) X[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < out_features; i++) b[i] = (float)rand() / RAND_MAX;

    double t_cpu;
    float t_gpu;

    // CPU
    wxb_cpu(W, X, b, Y_cpu, out_features, in_features, batch_size, &t_cpu);
    printf("CPU wx+b took %.6f ms\n", t_cpu);

    // GPU
    wxb_gpu(W, X, b, Y_gpu, out_features, in_features, batch_size, &t_gpu);
    printf("GPU wx+b took %.6f ms\n", t_gpu);

    // Difference
    float max_diff = 0.0f;
    for (int i = 0; i < batch_size * out_features; i++) {
        float diff = fabsf(Y_cpu[i] - Y_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max Difference CPU vs GPU: %.10f\n", max_diff);

    free(W);
    free(X);
    free(b);
    free(Y_cpu);
    free(Y_gpu);

    return 0;
}
