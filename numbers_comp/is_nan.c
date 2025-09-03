#include <stdio.h>
#include <stdint.h>

// 16-bit half-precision union for raw bit manipulation
typedef union {
    uint16_t d;
} fp16_t;

// Detect if raw 16-bit value is a NaN (IEEE 754 half precision)
int half_nan_isnan(uint16_t bits) {
    return (((bits >> 10) & 0x1F) == 0x1F) && ((bits & 0x3FF) != 0);
}

// Set NaN payload (payload limited to 10 bits)
uint16_t half_nan_set_payload(uint16_t nan, uint16_t payload) {
    // Clear mantissa, set only lower 10 bits from payload
    nan = (nan & ~0x03FF) | (payload & 0x03FF);
    return nan;
}

// Retrieve payload (lower 10 bits)
uint16_t half_nan_get_payload(uint16_t nan) {
    return nan & 0x03FF;
}

// Print binary for debugging
void print16(uint16_t v) {
    for (int i = 15; i >= 0; i--) {
        printf("%d", (v >> i) & 1);
        if (i == 15 || i == 10) printf(" "); // sign | exponent
    }
}

int main() {
 
    uint16_t base_nan = 0x7E00; 

    uint16_t payload = 0x0155;  
    uint16_t nan_with_payload = half_nan_set_payload(base_nan, payload);

    printf("base NaN raw bits      : "); print16(base_nan); printf("\n");
    printf("nan with payload bits  : "); print16(nan_with_payload); printf("\n");

    printf("Is NaN?                : %d\n", half_nan_isnan(nan_with_payload));
    printf("Payload                : 0x%03x\n", half_nan_get_payload(nan_with_payload));
    return 0;
}
