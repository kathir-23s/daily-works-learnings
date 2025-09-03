#include <stdio.h>

int main(){
    unsigned char x, y, z;

    x = 0x35;
    y = 0xee;
    z = x - y;

    printf("%x\n", z);
}