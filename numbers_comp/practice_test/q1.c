#include <stdio.h>
// #include <math.h>

unsigned short int base7_to_decimal(unsigned short int a){
    unsigned short int power = 1;
    unsigned short int res=0;
    while (a > 0){
        res += (a%10) * power;
        power *= 7;
        a /= 10;
    }
    return res;
}

int main(){
    unsigned short int n = 1563;
    unsigned short int decimal = base7_to_decimal(n);
    printf("Base 7 to Base 10: %u\n", decimal);
}

// Input: 1563 base 7
// Output: 633 base 10