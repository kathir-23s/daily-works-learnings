// file: add_all.c
#include <stdint.h>   // for intX_t and uintX_t
#include <stdio.h>

// -------- Floating-point --------
float  add_float32(float a, float b)       { return a + b; }
double add_float64(double a, double b)     { return a + b; }


// -------- Signed integers --------
int8_t   add_int8(int8_t a, int8_t b)       { return a + b; }
int16_t  add_int16(int16_t a, int16_t b)    { return a + b; }
int32_t  add_int32(int32_t a, int32_t b)    { return a + b; }
int64_t  add_int64(int64_t a, int64_t b)    { return a + b; }

// -------- Unsigned integers --------
uint8_t  add_uint8(uint8_t a, uint8_t b)    { return a + b; }
uint16_t add_uint16(uint16_t a, uint16_t b) { return a + b; }
uint32_t add_uint32(uint32_t a, uint32_t b) { return a + b; }
uint64_t add_uint64(uint64_t a, uint64_t b) { return a + b; }

// -------- Demo main --------
int main() {
    // Floating-point
    printf("float32: %f\n", add_float32(1.1f, 2.2f));
    printf("float64: %f\n", add_float64(1.1, 2.2));


    // Signed ints
    printf("int8: %d\n", add_int8(100, 27));
    printf("int16: %d\n", add_int16(30000, 1000));
    printf("int32: %d\n", add_int32(100000, 200000));
    printf("int64: %lld\n", (long long)add_int64(1000000000LL, 2000000000LL));

    // Unsigned ints
    printf("uint8: %u\n", add_uint8(200, 55));
    printf("uint16: %u\n", add_uint16(60000, 1234));
    printf("uint32: %u\n", add_uint32(4000000000U, 123456789U));
    printf("uint64: %llu\n", (unsigned long long)add_uint64(4000000000ULL, 123456789ULL));

    return 0;
}
