#include <stdio.h>

// Question 24 - Restoring division
unsigned char restore_div(unsigned char dvd, unsigned char dvs, unsigned char *r){
    *r = 0;
    unsigned char i = 0;
    for (i = 0; i < 8; i++){
        *r = (*r << 1) | ((dvd & 0x80) != 0);
        dvd <<= 1;
        if ((*r >= dvs)){
            dvd |= 1;
            *r -= dvs;
        }
    }
    return dvd;
}

// Question 23 - Multiplication using addition and shifts
unsigned short int multiply(unsigned short int a, unsigned short int b){
    unsigned short int res = 0;
    char i=0;
    while (b != 0){
        unsigned char mbit = (b & 1) == 1;
        if (mbit == 1){
            res += (a << i);
            i++;
            b>>=1;
        }
        else {
            res += (0 << i);
            i++;
            b>>=1;
        }
    }
    return res;
}

int main() {
    unsigned char dividend = 123;
    unsigned char divisor = 4;
    unsigned char remainder = 0;

    unsigned char quotient = restore_div(dividend, divisor, &remainder);

    printf("Dividend: %u\n", dividend);
    printf("Divisor: %u\n", divisor);
    printf("Quotient: %u\n", quotient);
    printf("Remainder: %u\n", remainder);

    unsigned char a = 13, b = 12;
    unsigned char product = multiply(a, b);
    printf("Product of a and b: %d\n", product);

    return 0;
}