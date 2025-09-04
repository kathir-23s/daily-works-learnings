#include <stdio.h>


void decimal_to_base(unsigned short int n, unsigned char base){
    if (n==0){
        printf("0");
        return;
    }
    short int res[32], i = 0;
    while (n>0){
        res[i++] = n % base;
        n /= base;
    }
    for (char j=i-1; j>=0; j--)
        printf("%d", res[j]);
    printf("\n");
}

void decimal_to_hex(unsigned short int n){
    if (n == 0) {
        printf("0\n");
        return;
    }
    char hex[5];
    int i = 0;
    while (n > 0) {
        unsigned char digit = n % 16;
        if (digit < 10)
            hex[i++] = '0' + digit;
        else
            hex[i++] = 'A' + (digit - 10);
        n /= 16;
    }
    for (int j = i - 1; j >= 0; j--)
        printf("%c", hex[j]);
    printf("\n");
}

int main(){
    unsigned short int a = 1023;
    unsigned char toBinary, toOctal, toHex;
    decimal_to_base(a, 2);
    decimal_to_base(a, 8);
    decimal_to_hex(a);
}
