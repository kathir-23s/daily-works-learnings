#include <stdio.h>

typedef union{
    float f;
    unsigned int d;
}fp_t;

float bin_to_fden(unsigned int a){
    float fden = 0;
    for (unsigned char i = 1; i<24; i++){
        unsigned char lsb = (a & 0x400000) == 0x400000;
        fden += lsb * (1.0f / (1 << i));
        a <<= 1;
    }
        return fden;

}

float ieee_to_fp(float a){
    fp_t fp;
    fp.f = a;

    unsigned int mant = fp.d & 0x7fffff;
    unsigned int exp = fp.d >> 23 & 0xFF;
    unsigned int sign = fp.d >> 31 & 0x1;
    
    // printf("Mantissa part: %X\n", mant);
    
    unsigned int exp_bias = exp - 127;
    // printf("Exponent part: %X\n", exp);
    signed char sign_factor = (sign == 0) ?  1 : -1;
    // printf("Sign part: %X\n", sign);
    // printf("Negative or Positive: %d\n", sign_factor);
    
    // printf("Exponent Bias: %X\n", exp_bias);
    float fraction = bin_to_fden(mant);
    // printf("Decimal Value of Mantissa: %f\n", fraction);
    return sign_factor * (1 + bin_to_fden(mant)) * (1 << exp_bias);
}

int main() {
    // binary: 0 10000000 10010010000111111011011 (π in IEEE-754 single-precision)
    fp_t fp;
    fp.d = 0x40490FDB;  // Hex equivalent of the given bit pattern
    float ans = ieee_to_fp(fp.f);
    printf("Floating point value: %f\n", ans);
    return 0;
}
