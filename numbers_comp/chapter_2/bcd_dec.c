#include <stdio.h>

unsigned short bin2bcd8 (unsigned char b){
    unsigned int p=0;
    unsigned char i,t;

    p = (unsigned int)b;

    for (i=0; i<8; i++){
        t = (p & 0xf00) >> 8;
        if (t>=5){
            t+=3;
            p = ((p>>12)<<12) | (t<<8) | (p&0xff);
        }
        t = (p & 0xf000) >> 12;
        if (t>=5){
            t+=3;
            p = ((p>>16)<<16) | (t<<12) | (p&0xfFf);
        }
        t = (p & 0xf0000) >> 16;
        if (t>=5){
            t+=3;
            p = ((p>>20)<<20) | (t<<16) | (p&0xffFF);
        }
        p <<= 1;
    }
    return (unsigned short)(p>>8);
}

unsigned char bcd2bin8 (unsigned short b){
    unsigned char n;

    n = (b & 0x0f);
    b >>= 4;
    n += 10 * (b & 0x0f);
    b >>=4;
    n += 100 * (b & 0x0f);

    return n;
}
unsigned int binary_to_bcd(unsigned int x) {
    unsigned int bcd = 0, shift = 0;
    while (x > 0) {
        unsigned int digit = x % 10;
        bcd |= (digit << (shift * 4));
        x /= 10;
        shift++;
    }
    return bcd;
}
int main(){
    unsigned char a = 14;
    unsigned short bcd_a = binary_to_bcd(a);
    printf("FROM DECIMAL TO BCD:%b\n", bcd_a);
    unsigned char dec_a = bcd2bin8(bcd_a);
    printf("BACK TO DECIMAL FROM BCD: %b\n", dec_a);
}