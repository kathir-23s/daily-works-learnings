#include <time.h>

void wxb_cpu(const float* W, const float* X,
             const float* b, float* Y,
             int out_features, int in_features, int batch_size,
             double* time_ms) {

    clock_t start = clock();

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < out_features; j++) {
            float sum = 0.0f;
            for (int k = 0; k < in_features; k++) {
                sum += X[i * in_features + k] * W[j * in_features + k];
            }
            Y[i * out_features + j] = sum + b[j];
        }
    }

    clock_t end = clock();
    *time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
}
