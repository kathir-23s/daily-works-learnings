#include <stdio.h>

signed char bset(signed char v, signed char n){
    return (v & (1 << n)) != 0;
}

signed char scomp(signed char a, signed char b){
    unsigned char i;

    if ((bset(a,7)==0) && (bset(b, 7)==1)) return 1;
    if ((bset(a,7)==1) && (bset(b, 7)==0)) return -1;

    for (i=0; i<7; i++){
        if ((bset(a, 6-i)==0) && (bset(b, 6-i)==1)) return -1;
        if ((bset(a, 6-i)==1) && (bset(b, 6-i)==0)) return 1;
    }
    return 0;
}

unsigned char isEqual(signed char a, signed char b){
    return scomp(a,b)==0;
}

unsigned char isLessThan(signed char a, signed char b){
    return scomp(a,b) == -1;
}

unsigned char isGreaterThan(signed char a, signed char b){
    return scomp(a,b) == 1;
}