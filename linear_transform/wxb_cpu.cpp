#include <iostream>
#include <chrono>
#include <vector>

extern "C" {
    void wxb_cpu (const float* W, const float* x, 
        const float* b, float* Y, int out_features, 
        int in_features, int batch_size, double* time_ms){

            auto start = std::chrono::high_resolution_clock::now();

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

        auto end = std::chrono::high_resolution_clock::now();
        *time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    }
}
