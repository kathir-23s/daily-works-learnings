#include <stdio.h>

unsigned char div1(unsigned char n, unsigned m, unsigned *r){
    unsigned char q=0;

    *r = n;
    while (*r > m){
        q++;
        *r -= m;
    }
    return q;
}

unsigned char div2(unsigned char n, unsigned char m, unsigned char *r){
    
    unsigned char i;
    *r = 0;

    for (i=0; i<8; i++){
        *r = (*r << 1) + ((n & 0x80) != 0);
        n <<= 1;
        if ((*r-m) >= 0){
            n |= 1;
            *r -= m;
        }
    }
    return n;
}