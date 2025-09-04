#include <stdio.h>

unsigned short bin_to_gray(unsigned short n){
    unsigned short gray_val = n ^ (n>>1);
    return gray_val;
}

int main(){
    unsigned short inp = 0b0011;
    unsigned short res = bin_to_gray(inp);
    printf("%x", res);
}