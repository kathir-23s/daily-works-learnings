#include <stdio.h>

int main() {
    unsigned short *p;
    unsigned short n = 256;

    p = (unsigned short*) &n;
    // printf("address of first byte = %p\n", &p[0]);
    printf("address of second and third bit = %p\t%p", &p[0], &p[1]);
}