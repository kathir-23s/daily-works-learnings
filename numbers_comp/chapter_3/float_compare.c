#include <stdio.h>

typedef union{
    float f;
    int d;
} sp_t;

int fp_compare (float a, float b){
    sp_t x,y;

    x.f = a;
    y.f = b;

    // if ((x.d == (int)0x80000000) && (y.d==0)) return 0;
    // if ((y.d == (int)0x80000000) && (x.d==0)) return 0;
    // if (x.d == y.d) return 0;

    if ((x.d == (int)0x80000000) && (y.d==0) || 
    (y.d == (int)0x80000000) && (x.d==0) || (x.d == y.d)) return 0; 

    if (x.d < y.d) return -1;
    return 1;
}

int fp_eq(float a, float b){
    return fp_compare(a,b) == 0;
}

int fp_lessthan(float a, float b){
    return fp_compare(a,b) == -1;
}
int fp_greaterthan(float a, float b){
    return fp_compare(a,b) == 1;
}


int main(){
    float a, b;

    a = 3.13, b = 3.13;
    printf("3.13 == 3.13: %d %d\n", fp_eq(a,b), a==b);

    a = 3.1; b = 3.13;
    printf("3.1 < 3.13: %d %d\n", fp_lessthan(a,b), a<b);

    a = 3.2;b = 3.13;
    printf("3.2 > 3.13: %d %d\n", fp_greaterthan(a,b), a>b);

    a = -1; b = 1;
    printf("-1 < 1: %d %d\n", fp_lessthan(a,b), a<b);
}