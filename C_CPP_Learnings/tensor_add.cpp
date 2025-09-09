// tensor_add.cpp
#include <cstdint>
#include <iostream>

// --------------------
// 1. C++ templated kernel
// --------------------
template <typename T>
void add_kernel(const T* a, const T* b, T* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

// --------------------
// 2. C ABI dispatcher
// --------------------
extern "C" void tensor_add(void* a, void* b, void* out,
                           int dtype, int64_t n) {
    switch (dtype) {
        case 0: // float
            add_kernel<float>((float*)a, (float*)b, (float*)out, n);
            break;
        case 1: // double
            add_kernel<double>((double*)a, (double*)b, (double*)out, n);
            break;
        case 2: // int64
            add_kernel<int64_t>((int64_t*)a, (int64_t*)b, (int64_t*)out, n);
            break;
        default:
            std::cerr << "Unsupported dtype!\n";
            break;
    }
}
