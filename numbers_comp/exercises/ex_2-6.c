#include <stdio.h>

unsigned char bit_reverse(unsigned char n) {
    unsigned char reversed = 0;
    for (int i = 0; i < 8; i++) {
        reversed <<= 1;
        reversed |= (n & 1);
        n >>= 1;
    }
    return reversed;
}

int main() {
    unsigned char x = 0b11010010; 
    unsigned char r = bit_reverse(x);
    printf("Reversed bits: %X\n", r);
    return 0;
}
