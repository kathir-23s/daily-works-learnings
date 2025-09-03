#include <stdio.h>

unsigned int countOnes(unsigned int n) {
    unsigned int count = 0;
    while (n) {
        count += (n & 1);
        n >>= 1;
    }
    return count;
}

int main() {
    unsigned int x = 0b11010101100101010101010101101010;
    unsigned int c = countOnes(x);
    printf("Number of 1 bits: %u\n", c);
    return 0;
}
