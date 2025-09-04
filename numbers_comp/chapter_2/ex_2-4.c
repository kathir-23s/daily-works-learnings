#include <stdio.h>

int main(){
    unsigned char v = 0xc4;
    unsigned char l_nib=0, r_nib=0;
    l_nib = v & 0xf0;
    r_nib = v & 0x0f;
    unsigned char res = (r_nib << 4) | (l_nib >> 4);

    printf("Before Nibble Switch: %X\n", v);
    printf("After Nibble switch: %X\n", res);

    unsigned char i;
    unsigned char m = 5;
    unsigned short ans = 0;

    for (i=0; i<8; i++){
        if (m&1){
            ans+=v << i;
    }
        m >>= 1;
    }

    printf("v * 5 : %X\n\n", ans);
}