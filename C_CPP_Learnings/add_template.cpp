// file: add_template.cpp
#include <cstdint>
#include <iostream>

// Generic template
template <typename T>
T add(T a, T b) {
    return a + b;
}

int main() {
    // Floating-point
    std::cout << "float32: " << add<float>(1.1f, 2.2f) << "\n";
    std::cout << "float64: " << add<double>(1.1, 2.2) << "\n";

    // Signed ints
    std::cout << "int8: "   << (int)add<int8_t>((int8_t)100, (int8_t)27) << "\n";
    std::cout << "int16: "  << add<int16_t>((int16_t)30000, (int16_t)1000) << "\n";
    std::cout << "int32: "  << add<int32_t>((int32_t)100000, (int32_t)200000) << "\n";
    std::cout << "int64: "  << add<int64_t>((int64_t)1000000000LL, (int64_t)2000000000LL) << "\n";

    // Unsigned ints
    std::cout << "uint8: "  << (unsigned int)add<uint8_t>((uint8_t)200, (uint8_t)55) << "\n";
    std::cout << "uint16: " << add<uint16_t>((uint16_t)60000, (uint16_t)1234) << "\n";
    std::cout << "uint32: " << add<uint32_t>((uint32_t)4000000000U, (uint32_t)123456789U) << "\n";
    std::cout << "uint64: " << add<uint64_t>((uint64_t)4000000000ULL, (uint64_t)123456789ULL) << "\n";

    return 0;
}
