#include <time.h>

void wxb_cpuc(const float* W, const float* x,
             const float* b, float* Y, int out_features,
             int in_features, int batch_size, double* time_ms) {

    clock_t start, end;
    start = clock();

    // Loop over batch
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < out_features; j++) {
            float sum = 0.0f;
            for (int k = 0; k < in_features; k++) {
                sum += x[i * in_features + k] * W[j * in_features + k];
            }
            Y[i * out_features + j] = sum + b[j];
        }
    }

    end = clock();
    *time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;  // time in milliseconds
}
