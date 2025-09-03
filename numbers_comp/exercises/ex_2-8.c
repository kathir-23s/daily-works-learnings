#include <stdio.h>

unsigned int hammingDistance(unsigned short x, unsigned short y) {
    unsigned short xor_val = x ^ y;   
    unsigned int distance = 0;

    while (xor_val) {
        distance += (xor_val & 1);   
        xor_val >>= 1;               }

    return distance;
}


int main() {
    unsigned short a = 0b1011;  // 11 decimal
    unsigned short b = 0b0010;  // 2 decimal
    printf("Hamming distance: %u\n", hammingDistance(a, b));  // Output: 2
    return 0;
}