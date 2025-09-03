#include <stdio.h>

signed short signed_mult1(signed char n, signed char m){
    unsigned char i, s=0;
    signed short ans=0;

    if((n>0) && (m<0)){
        s = 1;
        m = -m;
    }
    if((n<0) && (m>0)){
        s = 1;
        n = -n;
    }
    if ((n<0) && (m<0)){
        n = -n;
        m = -m;
    }    

    for (i=0; i<8; i++){
        if (m&1) ans += n<<i;
        m >>= i;
    }

    if (s) ans = -ans;
    return ans;

}

signed short signed_mult2(signed char m, signed char r){
    signed int A, S, P;
    unsigned char i;

    A = m << 9;
    S = (-m) << 9;
    P = ( r & 0xff) << 1;

    for (i=0; i<8; i++){
        switch (P & 3) {
            case 1:
                P += A; 
                break;
            case 2:
                P += S; 
                break;
            default:
                break;
        }
        P >> 1;
    }
    return P >> 1;
}