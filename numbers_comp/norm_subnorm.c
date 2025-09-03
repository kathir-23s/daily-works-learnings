#include <stdio.h>
#include <string.h>

const char *ppx(unsigned int x) {
    static char b[36];
    unsigned int z, i=0;
    b[0] = '\0';
    for (z = (1<<31); z > 0; z >>= 1) {
        strcat(b, ((x & z) == z) ? "1" : "0");
        if ((i == 0) || (i == 8))
            strcat(b, " ");
        i++;
    }
    return b;
}

typedef union {
    float f;
    unsigned int d;
} fp_t;

int main() {
    fp_t x;
    int i;
    x.f = 1.0;
    for(i=0; i < 151; i++) {
        printf("x = (%s) %0.8g\n", ppx(x.d), x.f);
        x.f /= 2.0;
    }
}
