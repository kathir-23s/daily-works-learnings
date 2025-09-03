#include <stdio.h>

int main() {
    printf("---Ex 1---\n");
    if (0xBD & 0x08){
        printf("bit 3 is on(true)\n\n");
    }
    printf("---Ex 2---\n");
    unsigned char z = 0xB5;
    z = z | 0x08; // set bit 3
    printf("%x\n", z);
    z |= 0x40;
    // set bit 6
    printf("%x\n", z);
    return 0;
}