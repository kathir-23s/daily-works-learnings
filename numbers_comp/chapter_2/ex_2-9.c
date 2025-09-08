#include <stdio.h>

unsigned short bin_to_gray(unsigned short n){
    unsigned short gray_val = n ^ (n>>1);
    return gray_val;
}

unsigned short gray_to_bin(unsigned short n){
    unsigned short mask;
    for (mask = n >> 1; mask != 0; mask = mask >> 1) {
        n ^= mask;
    }
    return n;
}

int main(){
    unsigned short inp = 0b00110011;
    printf("%d\n", inp);
    unsigned short res = bin_to_gray(inp);
    printf("%d\n", res);
    unsigned short res2 = gray_to_bin(res);
    printf("%d\n", res2);
}