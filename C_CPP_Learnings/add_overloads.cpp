// file: add_overloads.cpp
#include <cstdint>
#include <iostream>

// -------- Floating-point --------
float  add(float a, float b)       { return a + b; }
double add(double a, double b)     { return a + b; }

// -------- Signed integers --------
int8_t   add(int8_t a, int8_t b)     { return a + b; }
int16_t  add(int16_t a, int16_t b)   { return a + b; }
int32_t  add(int32_t a, int32_t b)   { return a + b; }
int64_t  add(int64_t a, int64_t b)   { return a + b; }

// -------- Unsigned integers --------
uint8_t  add(uint8_t a, uint8_t b)   { return a + b; }
uint16_t add(uint16_t a, uint16_t b) { return a + b; }
uint32_t add(uint32_t a, uint32_t b) { return a + b; }
uint64_t add(uint64_t a, uint64_t b) { return a + b; }

// -------- Demo main --------
int main() {
    // Floating-point
    std::cout << "float32: " << add(1.1f, 2.2f) << "\n";
    std::cout << "float64: " << add(1.1, 2.2) << "\n";

    // Signed ints
    std::cout << "int8: "   << (int)add((int8_t)100, (int8_t)27) << "\n";
    std::cout << "int16: "  << add((int16_t)30000, (int16_t)1000) << "\n";
    std::cout << "int32: "  << add((int32_t)100000, (int32_t)200000) << "\n";
    std::cout << "int64: "  << add((int64_t)1000000000LL, (int64_t)2000000000LL) << "\n";

    // Unsigned ints
    std::cout << "uint8: "  << (unsigned int)add((uint8_t)200, (uint8_t)55) << "\n";
    std::cout << "uint16: " << add((uint16_t)60000, (uint16_t)1234) << "\n";
    std::cout << "uint32: " << add((uint32_t)4000000000U, (uint32_t)123456789U) << "\n";
    std::cout << "uint64: " << add((uint64_t)4000000000ULL, (uint64_t)123456789ULL) << "\n";

    return 0;
}
