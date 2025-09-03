#include <stdio.h>

signed short booth_mul(signed char m, signed char r){
    signed int A, P, S;
    unsigned char i;

    A = m << 9;
    S = (-m) << 9;
    P = (r & 0xff) << 1;

    for (i=0; i<8; i++){
        switch (P & 3)
        {
            // by masking with 3, the values we can get are 0,1,2,3
            case 1:  // case for 01 which is binary for 1
                P += A;
                break;
            case 2: // case for 10 which is 2 in binary
                P += S;
                break;
            default: // case for 0 or 3 (binary 00 or 11)
                break;
        }
        P >>= 1;
    }
    return P >> 1;
}

int main(){
    signed char a = 10, b = -4;
    signed short res = booth_mul(a, b);
    printf("%hd\n", res);
}