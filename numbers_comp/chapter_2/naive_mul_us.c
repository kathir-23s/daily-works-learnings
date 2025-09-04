#include <stdio.h>

unsigned short mult1 (unsigned char n, unsigned char m){
    unsigned char i;
    unsigned short ans = 0;

    if (n<m){
        for (i=0; i<n; i++) ans += m;
    } else {
        for (i=0; i<m; i++){
            ans += n;
        }
    }
    return ans;
}

unsigned short mult2 (unsigned char n, unsigned char m){
    unsigned char i;
    unsigned short ans = 0;

    for (i=0; i<8; i++){
        if (m&1){
            ans += n << i;
        }
        m >>= 1;
    }
    return ans;
}